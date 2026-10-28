"""Self-play PPO training over the deck pool.

    python -m ocgforge.train --out runs/base-s0 --size base --seed 0

Every iteration plays `--steps` decisions with AsyncCollector, runs one PPO update and appends a
line to <out>/metrics.jsonl. A `--league-share` of the envs play frozen earlier checkpoints instead
of the current model: the last `--league-size` arena checkpoints plus anchors (the first and every
`--anchor-every` iterations), picked by PFSP (rollout.League). League envs keep only one seat's
decisions, so their share of the training data is lower (logged as leagueDataShare). Every
`--arena-every` iterations the model plays `--arena-duels` duels against a random policy, its
previous and its first arena checkpoint (both seats, sampled moves); win rates, how many duels they
rest on and an Elo estimate (random = 1500) go to <out>/arena.jsonl, how often the model took each
card's actions since the last arena to <out>/cards.jsonl and the model to
<out>/checkpoints/<iteration>. <out>/latest holds the checkpoint plus optimizer state to resume
from with --resume. Run several --seed values to see how much a result depends on luck.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from . import ladder, rollout
from .cards import CardFeatures, CardTable
from .embed_cards import load_text_embeddings
from .model import Model, ModelConfig, load_checkpoint, save_checkpoint
from .ppo import PPOConfig, build_steps, ppo_update

SIZES = {
    "small": ModelConfig(d_model=128, heads=4, layers=3, ff=256),
    "base": ModelConfig(d_model=256, heads=8, layers=6, ff=1024),
    "large": ModelConfig(d_model=384, heads=8, layers=8, ff=1536),
}
STATE_FILE = "trainer.pt"
# glibc malloc fragments under many engine threads (RSS keeps growing) and is slower: with jemalloc
# the engine runs ~25% faster and memory stays flat. `sudo apt install libjemalloc2`.
JEMALLOC = Path("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2")


def use_jemalloc(argv: list[str]) -> None:
    """Restart this process with jemalloc preloaded, if it is installed and not loaded yet."""
    if not JEMALLOC.exists() or str(JEMALLOC) in os.environ.get("LD_PRELOAD", ""):
        return
    env = {**os.environ, "LD_PRELOAD": ":".join(p for p in (os.environ.get("LD_PRELOAD"), str(JEMALLOC)) if p)}
    os.execve(sys.executable, [sys.executable, "-m", "ocgforge.train", *argv], env)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", choices=sorted(SIZES), default="base")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decks", type=Path, nargs="+", default=sorted((rollout.REPO / "decks").glob("*.ydk")))
    p.add_argument("--text-embeddings", type=Path, default=rollout.REPO / ".cache" / "card_text_emb.npz",
                   help="embed_cards.py output; pass 'none' to train without card text embeddings")
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--threads", type=int, default=0, help="engine threads, 0 = all cores")
    p.add_argument("--segment", type=int, default=128)
    p.add_argument("--steps", type=int, default=32768, help="decisions per iteration")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--hours", type=float, default=0.0, help="stop after this long, 0 = no limit")
    p.add_argument("--lr", type=float, default=PPOConfig.lr)
    p.add_argument("--epochs", type=int, default=PPOConfig.epochs)
    p.add_argument("--batch-size", type=int, default=PPOConfig.batch_size)
    p.add_argument("--micro-batch", type=int, default=PPOConfig.micro_batch)
    p.add_argument("--entropy", type=float, default=PPOConfig.entropy_coef)
    p.add_argument("--league-share", type=float, default=0.4, help="share of envs (0.4 = ~25%% of the data)")
    p.add_argument("--league-size", type=int, default=6, help="newest checkpoints in the league")
    p.add_argument("--anchor-every", type=int, default=250, help="checkpoints kept as long-term league anchors")
    p.add_argument("--arena-every", type=int, default=40)
    p.add_argument("--arena-duels", type=int, default=200, help="per matchup; 200 gives about +-7%% on a win rate")
    p.add_argument("--arena-envs", type=int, default=128)
    p.add_argument("--save-every", type=int, default=5, help="iterations between saves of <out>/latest")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp32", action="store_true", help="no bf16 autocast (slower, exact)")
    p.add_argument("--no-jemalloc", action="store_true", help="keep glibc malloc")
    return p.parse_args(argv)


def deck_codes(path: Path) -> set[int]:
    """Passcodes in a .ydk file's main and extra deck (the side deck is ignored)."""
    codes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("!side"):
            break
        if line.isdigit():
            codes.add(int(line))
    return codes


def build_model(args: argparse.Namespace) -> Model:
    # Only cards in the training decks get a learned name embedding; every other card (e.g. in the
    # hold-out decks) is read from its text and stats alone.
    table = CardTable.from_cdb()
    trained = {card.code for deck in args.decks for code in deck_codes(deck) if (card := table.card(table.index(code)))}
    card_features = CardFeatures.build(table, trained_codes=trained)
    config = replace(SIZES[args.size], dropout=0.0)
    embeddings = None
    if str(args.text_embeddings) != "none":
        embeddings = load_text_embeddings(args.text_embeddings, card_features)
        config = replace(config, text_embedding_dim=embeddings.shape[1])
    return Model(card_features, config, text_embeddings=embeddings)


def card_names(card_features: CardFeatures) -> dict[int, str]:
    """Card index -> name."""
    table = CardTable.from_cdb()
    by_code = {card.code: card.name for card in table.cards}
    return {i + 2: by_code.get(code, str(code)) for i, code in enumerate(card_features.codes)}


def card_report(usage: rollout.CardUsage, names: dict[int, str]) -> list[dict]:
    """Every card that had an action, most played first: offered (decisions with one of its
    actions), chosen, and the share of those decisions in which the model took it."""
    rows = [{"card": names.get(int(i), str(int(i))), "offered": int(usage.offered[i]), "chosen": int(usage.chosen[i]),
             "rate": round(float(usage.chosen[i] / usage.offered[i]), 4)}
            for i in np.flatnonzero(usage.offered)]
    return sorted(rows, key=lambda r: -r["chosen"])


def arena(model: Model, opponents: dict[str, Model], args: argparse.Namespace, iteration: int) -> dict:
    """Duels against a random policy and each of `opponents`; returns win rates and Elo."""
    names = [d.stem for d in args.decks]
    env = rollout.vector_env(model.card_features, args.decks, num_envs=args.arena_envs,
                             seed=args.seed * 1_000_003 + iteration, threads=args.threads)
    me = rollout.ModelPolicy(model, args.device, seed=iteration, bf16=not args.fp32)
    name = f"it{iteration}"
    results = rollout.match(env, me, rollout.RandomPolicy(iteration), args.arena_duels, name, "random", names)
    for k, (label, other) in enumerate(opponents.items(), start=1):
        policy = rollout.ModelPolicy(other, args.device, seed=iteration + k, bf16=not args.fp32)
        results += rollout.match(env, me, policy, args.arena_duels, name, label, names)
    # Duels cut off (loop protection, Lua budget) are not counted, so a matchup can have no games.
    summary: dict = {}
    for label in ("random", *opponents):
        summary.update({f"vs_{label}": None, f"games_{label}": 0})
    for (a, b), stats in ladder.matchups(results).items():
        if a == name:
            summary[f"vs_{b}"] = round(stats.win_rate, 4)
            summary[f"games_{b}"] = stats.games
    summary["elo"] = round(ladder.elo(results, anchor="random").get(name, 0.0), 1) if results else None
    return summary


def save_latest(model: Model, optimizer: torch.optim.Optimizer, out: Path, iteration: int) -> None:
    """<out>/latest, replaced only once the new copy is complete (latest.old survives a crash midway)."""
    tmp, latest, old = out / "latest.tmp", out / "latest", out / "latest.old"
    shutil.rmtree(tmp, ignore_errors=True)
    save_checkpoint(model, tmp)
    torch.save({"optimizer": optimizer.state_dict(), "iteration": iteration}, tmp / STATE_FILE)
    shutil.rmtree(old, ignore_errors=True)
    if latest.exists():
        latest.rename(old)
    tmp.rename(latest)
    shutil.rmtree(old, ignore_errors=True)


def drop_after(path: Path, iteration: int) -> None:
    """Remove log lines of iterations after `iteration` (run again after a resume)."""
    if path.exists():
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if json.loads(l)["iteration"] <= iteration]
        path.write_text("".join(l + "\n" for l in lines), encoding="utf-8")


def error_reasons(messages: list[str], top: int = 3) -> dict[str, int]:
    return dict(Counter(m.split(" (seed")[0].splitlines()[0][:120] for m in messages).most_common(top))


def main(argv=None) -> None:
    args = parse_args(argv)
    out: Path = args.out
    if not args.resume and (out / "metrics.jsonl").exists():
        raise SystemExit(f"{out} already holds a run: pass --resume to continue it, or choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))  # kill: save first
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # rollouts and updates alike, so logprobs stay comparable
    torch.backends.cudnn.allow_tf32 = True

    ppo = PPOConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, micro_batch=args.micro_batch,
                    entropy_coef=args.entropy, kl_coef=0.0, bf16=not args.fp32)
    iteration = 0
    saved = next((d for d in (out / "latest", out / "latest.old") if (d / STATE_FILE).exists()), None)
    if args.resume and saved is not None:
        model = load_checkpoint(saved, args.device, dropout=0.0)
        state = torch.load(saved / STATE_FILE, map_location=args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=ppo.lr)
        optimizer.load_state_dict(state["optimizer"])
        iteration = state["iteration"]
        for log in ("metrics.jsonl", "arena.jsonl", "cards.jsonl"):
            drop_after(out / log, iteration)
    else:
        model = build_model(args).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=ppo.lr)
    run = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "decks": [str(d) for d in args.decks], "model": model.config.to_json(), "ppo": asdict(ppo),
           "parameters": sum(p.numel() for p in model.parameters())}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"{run['parameters'] / 1e6:.1f}M parameters, {args.size}, seed {args.seed}, from iteration {iteration}",
          flush=True)

    env = rollout.vector_env(model.card_features, args.decks, num_envs=args.envs,
                             seed=args.seed * 1_000_003 + iteration, threads=args.threads)
    league = rollout.League(args.league_share, args.league_size, seed=args.seed)
    first: Model | None = None  # first and latest arena checkpoint, for the arena
    previous: Model | None = None
    anchored = -1  # anchor_every bucket of the latest anchor

    def admit(checkpoint: Path) -> Model:
        """Load an arena checkpoint into the league; the first one, and the first one in each new
        block of --anchor-every iterations, stay as anchors."""
        nonlocal first, previous, anchored
        frozen = load_checkpoint(checkpoint, args.device, dropout=0.0)
        bucket = int(checkpoint.name) // args.anchor_every
        anchor = first is None or bucket > anchored
        anchored = max(anchored, bucket) if anchor else anchored
        league.add(checkpoint.name, rollout.ModelPolicy(frozen, args.device, seed=args.seed + int(checkpoint.name),
                                                        bf16=ppo.bf16), anchor=anchor)
        first = frozen if first is None else first
        previous = frozen
        return frozen

    for checkpoint in sorted((out / "checkpoints").glob("*")) if args.resume else []:  # the league drops old ones
        if int(checkpoint.name) <= iteration:
            admit(checkpoint)
    usage = rollout.CardUsage(model.card_features.size)
    collector = rollout.AsyncCollector(env, batch_size=max(1, args.envs // 2), segment=args.segment,
                                       league=league if args.league_share > 0 else None, card_usage=usage)
    names = card_names(model.card_features)
    policy = rollout.ModelPolicy(model, args.device, seed=args.seed, bf16=ppo.bf16)
    deadline = time.time() + args.hours * 3600 if args.hours > 0 else None
    saved_iteration, failing = iteration, 0
    try:
        while iteration < args.iterations and (deadline is None or time.time() < deadline):
            start = time.time()
            segments, stats = collector.collect(policy, args.steps)
            collected = time.time()
            steps = build_steps(segments, ppo.gamma, ppo.lam)
            history = ppo_update(model, optimizer, steps, ppo, args.device, seed=args.seed * 1_000_003 + iteration + 1)
            iteration += 1
            done = time.time()
            finished = stats.duels + stats.league_duels + stats.errors
            line = {"iteration": iteration, "time": round(done, 1), "decisions": len(steps),
                    "decisionsPerSecond": round(stats.decisions / (collected - start)),
                    "collectSeconds": round(collected - start, 2), "updateSeconds": round(done - collected, 2),
                    "duels": stats.duels, "leagueDuels": stats.league_duels, "errors": stats.errors,
                    "errorReasons": error_reasons(stats.error_messages),
                    "leagueDataShare": round(stats.league_decisions / max(1, stats.learner_decisions), 3),
                    "wins": stats.wins, "league": league.scores(),
                    **{k: round(v, 5) if isinstance(v, float) else v for k, v in history[-1].items()}}
            with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
            print(json.dumps(line), flush=True)
            failing = failing + 1 if finished >= 10 and stats.errors > finished / 2 else 0
            if failing >= 3:
                print(f"WARNING: more than half of the finished duels failed for {failing} iterations: "
                      f"{line['errorReasons']}", flush=True)

            if iteration % args.arena_every == 0:
                checkpoint = out / "checkpoints" / f"{iteration:06d}"
                save_checkpoint(model, checkpoint)
                opponents = {"previous": previous} if previous is not None else {}
                if first is not None and first is not previous:
                    opponents["first"] = first
                result = {"iteration": iteration, **arena(model, opponents, args, iteration)}
                with open(out / "arena.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)
                with open(out / "cards.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iteration": iteration, "cards": card_report(usage, names)}) + "\n")
                usage.reset()
                admit(checkpoint)
            if iteration % args.save_every == 0 or iteration % args.arena_every == 0:
                save_latest(model, optimizer, out, iteration)
                saved_iteration = iteration
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        if iteration > saved_iteration:  # the deadline, the last iterations or an interrupt
            save_latest(model, optimizer, out, iteration)
            print(f"saved iteration {iteration} to {out / 'latest'}", flush=True)


if __name__ == "__main__":
    if "--no-jemalloc" not in sys.argv[1:]:
        use_jemalloc(sys.argv[1:])
    main()
