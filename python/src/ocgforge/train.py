"""Self-play PPO training over the deck pool.

    python -m ocgforge.train --out runs/base-s0 --size base --seed 0

Every iteration plays `--steps` decisions with AsyncCollector, runs one PPO update and appends a
line to <out>/metrics.jsonl. A `--league-share` of the envs play frozen earlier checkpoints (the
last `--league-size` arena checkpoints, picked by PFSP) instead of the current model. Every `--arena-every` iterations the model plays `--arena-duels` duels
against a random policy and against its previous arena checkpoint (both seats, sampled moves); the
results and an Elo estimate (random = 0) go to <out>/arena.jsonl, how often the model took each
card's actions since the last arena to <out>/cards.jsonl and the model to
<out>/checkpoints/<iteration>. <out>/latest holds the checkpoint plus optimizer state to resume
from with --resume. Run several --seed values to see how much a result depends on luck.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    p.add_argument("--league-share", type=float, default=0.25)
    p.add_argument("--league-size", type=int, default=6)
    p.add_argument("--arena-every", type=int, default=10)
    p.add_argument("--arena-duels", type=int, default=256)
    p.add_argument("--arena-envs", type=int, default=64)
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


def arena(model: Model, opponent: Model | None, args: argparse.Namespace, iteration: int) -> dict:
    """Duels against a random policy and the previous checkpoint; returns win rates and Elo."""
    names = [d.stem for d in args.decks]
    env = rollout.vector_env(model.card_features, args.decks, num_envs=args.arena_envs,
                             seed=args.seed * 1_000_003 + iteration, threads=args.threads)
    me = rollout.ModelPolicy(model, args.device, seed=iteration, bf16=not args.fp32)
    name = f"it{iteration}"
    results = rollout.match(env, me, rollout.RandomPolicy(iteration), args.arena_duels, name, "random", names)
    if opponent is not None:
        previous = rollout.ModelPolicy(opponent, args.device, seed=iteration + 1, bf16=not args.fp32)
        results += rollout.match(env, me, previous, args.arena_duels, name, "previous", names)
    # Duels cut off (loop protection, Lua budget) are not counted, so a matchup can have no games.
    summary: dict = {"vs_random": None, "games_random": 0}
    if opponent is not None:
        summary.update(vs_previous=None, games_previous=0)
    for (a, b), stats in ladder.matchups(results).items():
        if a == name:
            summary[f"vs_{b}"] = round(stats.win_rate, 4)
            summary[f"games_{b}"] = stats.games
    summary["elo"] = round(ladder.elo(results, anchor="random").get(name, 0.0), 1) if results else None
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # rollouts and updates alike, so logprobs stay comparable
    torch.backends.cudnn.allow_tf32 = True

    ppo = PPOConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, micro_batch=args.micro_batch,
                    entropy_coef=args.entropy, kl_coef=0.0, bf16=not args.fp32)
    iteration = 0
    if args.resume and (out / "latest").exists():
        model = load_checkpoint(out / "latest", args.device, dropout=0.0)
        state = torch.load(out / "latest" / STATE_FILE, map_location=args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=ppo.lr)
        optimizer.load_state_dict(state["optimizer"])
        iteration = state["iteration"]
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
    for checkpoint in sorted((out / "checkpoints").glob("*"))[-args.league_size:]:  # when resuming
        league.add(checkpoint.name, rollout.ModelPolicy(load_checkpoint(checkpoint, args.device, dropout=0.0),
                                                        args.device, seed=args.seed + int(checkpoint.name), bf16=ppo.bf16))
    usage = rollout.CardUsage(model.card_features.size)
    collector = rollout.AsyncCollector(env, batch_size=max(1, args.envs // 2), segment=args.segment,
                                       league=league if args.league_share > 0 else None, card_usage=usage)
    names = card_names(model.card_features)
    policy = rollout.ModelPolicy(model, args.device, seed=args.seed, bf16=ppo.bf16)
    opponent = None
    deadline = time.time() + args.hours * 3600 if args.hours > 0 else None
    while iteration < args.iterations and (deadline is None or time.time() < deadline):
        iteration += 1
        start = time.time()
        segments, stats = collector.collect(policy, args.steps)
        collected = time.time()
        steps = build_steps(segments, ppo.gamma, ppo.lam)
        history = ppo_update(model, optimizer, steps, ppo, args.device, seed=args.seed * 1_000_003 + iteration)
        done = time.time()
        line = {"iteration": iteration, "time": round(done, 1), "decisions": len(steps),
                "decisionsPerSecond": round(stats.decisions / (collected - start)),
                "collectSeconds": round(collected - start, 2), "updateSeconds": round(done - collected, 2),
                "duels": stats.duels, "leagueDuels": stats.league_duels, "errors": stats.errors,
                "wins": stats.wins, "league": league.scores(),
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in history[-1].items()}}
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)

        if iteration % args.arena_every == 0:
            checkpoint = out / "checkpoints" / f"{iteration:06d}"
            save_checkpoint(model, checkpoint)
            result = {"iteration": iteration, **arena(model, opponent, args, iteration)}
            with open(out / "arena.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            with open(out / "cards.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"iteration": iteration, "cards": card_report(usage, names)}) + "\n")
            usage.reset()
            opponent = load_checkpoint(checkpoint, args.device, dropout=0.0)
            league.add(checkpoint.name, rollout.ModelPolicy(opponent, args.device, seed=args.seed + iteration, bf16=ppo.bf16))
            save_checkpoint(model, out / "latest")
            torch.save({"optimizer": optimizer.state_dict(), "iteration": iteration}, out / "latest" / STATE_FILE)


if __name__ == "__main__":
    if "--no-jemalloc" not in sys.argv[1:]:
        use_jemalloc(sys.argv[1:])
    main()
