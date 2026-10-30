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
from with --resume. --init starts a new run from the weights of an earlier checkpoint (fresh
optimizer, iteration 0); that checkpoint becomes <out>/checkpoints/000000, the first arena opponent
and a league anchor. Run several --seed values to see how much a result depends on luck.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from . import ladder, rollout
from .cards import UNK, CardFeatures, CardTable
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
    p.add_argument("--overlap", action="store_true",
                   help="collect the next iteration's duels while the GPU trains on the last ones (with the "
                        "weights from one update earlier); needs GPU memory for a second copy of the model")
    p.add_argument("--search-data", type=Path,
                   help="a directory that search workers fill with shards (expert generate --forever): each PPO "
                        "minibatch also imitates a sample of the newest search targets")
    p.add_argument("--search-coef", type=float, default=0.5, help="weight of that imitation loss")
    p.add_argument("--search-batch", type=int, default=128, help="search samples per PPO minibatch")
    p.add_argument("--search-window", type=int, default=50_000, help="newest search samples kept")
    p.add_argument("--reset-heads", action="store_true",
                   help="with --init: new policy (score) and value heads, the card and board encoder kept")
    p.add_argument("--extend-seen", action="store_true",
                   help="with --init: the training decks' cards get learned name embeddings too (starting from "
                        "the unknown card's), not only the cards of the run the checkpoint was first built for")
    p.add_argument("--entropy-start", type=float, help="entropy coefficient at iteration 0, ramped to --entropy")
    p.add_argument("--entropy-ramp", type=int, default=0, help="iterations from --entropy-start to --entropy")
    p.add_argument("--tempo-slope", type=float, default=0.0,
                   help="duel results lose this much weight per turn after --tempo-start (0 = off)")
    p.add_argument("--tempo-start", type=int, default=5, help="last turn (both players' turns) at full weight")
    p.add_argument("--tempo-floor", type=float, default=0.6, help="lowest weight of a result")
    p.add_argument("--tempo-losses", action="store_true", help="shrink losses too (default: wins only)")
    p.add_argument("--skip-forced", action="store_true",
                   help="answer single-candidate decisions without the model and leave them out of training "
                        "(about a third of all decisions; --steps still counts every decision)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", type=Path, help="start from this checkpoint's weights (e.g. <run>/latest)")
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


# Settings a resumed run keeps from run.json unless given again: its origin and its reward and
# exploration schedules (dropping them silently would change what the run optimizes mid-way).
RESUMED_SETTINGS = {"init": None, "tempo_slope": 0.0, "tempo_start": 5, "tempo_floor": 0.6, "tempo_losses": False,
                    "entropy_start": None, "entropy_ramp": 0, "skip_forced": False}


def restore_run_settings(args: argparse.Namespace, run: dict) -> None:
    for name, default in RESUMED_SETTINGS.items():
        if getattr(args, name) == default and run.get(name, default) != default:
            value = run[name]
            setattr(args, name, Path(value) if name == "init" else value)
            print(f"--resume: {name} = {value} from run.json", flush=True)


def extend_seen(model: Model, decks: list[Path]) -> int:
    """Marks every card of `decks` as seen (a learned name embedding instead of "unknown"); newly
    seen cards start from the unknown card's embedding, so the model plays them as before until
    training moves them. Returns how many cards were added."""
    index = model.card_features.index_map()
    cards = sorted({index[code] for deck in decks for code in deck_codes(deck) if code in index})
    new = [i for i in cards if not model.card_features.seen[i]]
    with torch.no_grad():
        model.card_seen[new] = True
        model.name.weight[new] = model.name.weight[UNK].clone()
    model.card_features.seen[new] = True
    return len(new)


def search_auxiliary(model: Model, data, args: argparse.Namespace, micro_batch: int):
    """PPO's auxiliary term: cross-entropy to a sample of the newest search targets, weighted by
    --search-coef and run in micro-batches like the PPO loss."""
    from .expert import _batch, search_policy_loss

    def run():
        samples = data.sample(args.search_batch)
        if not samples:
            return None
        stats = {"searchLoss": 0.0, "searchAgree": 0.0}
        for start in range(0, len(samples), micro_batch):
            part = samples[start:start + micro_batch]
            batch, targets, _ = _batch(part, args.device)
            loss, part_stats = search_policy_loss(model, batch, targets)
            (loss * args.search_coef * len(part) / len(samples)).backward()
            for key in stats:
                stats[key] += part_stats[key] * len(part) / len(samples)
        return stats

    return run


def reset_heads(model: Model) -> None:
    """Fresh policy and value heads: what the encoder knows about cards and boards stays, what the
    model learned to do with it starts over."""
    for head in (model.score, model.value):
        for layer in head.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()


def entropy_coef(args: argparse.Namespace, iteration: int) -> float:
    """--entropy, or a linear ramp from --entropy-start to it over --entropy-ramp iterations."""
    if args.entropy_start is None or args.entropy_ramp <= 0:
        return args.entropy
    share = min(1.0, iteration / args.entropy_ramp)
    return args.entropy_start + (args.entropy - args.entropy_start) * share


class Prefetch:
    """collector.collect(policy, steps) on a daemon thread (the engine and torch release the GIL), so
    the duels of the next iteration are played while the learner trains. Daemon: an interrupt does
    not wait for it."""

    def __init__(self, collector: rollout.AsyncCollector, policy, steps: int):
        self.result, self.error = None, None

        def run() -> None:
            start = time.time()
            try:
                self.result = (*collector.collect(policy, steps), time.time() - start)
            except BaseException as error:  # handed to the main thread by get()
                self.error = error

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def get(self):
        """(segments, stats, collect seconds) once the collection is done."""
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self.result


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


def extra_report(usage: rollout.ExtraDeckUsage, names: dict[int, str]) -> list[dict]:
    """Every Extra Deck monster that reached the learner's field, most played first: duels it was
    played in, and own turns (all, and the duel's first turn) it ended on the field."""
    rows = [{"card": names.get(int(i), str(int(i))), "played": int(usage.played[i]),
             "endBoard": int(usage.end_board[i]), "firstTurn": int(usage.first_board[i])}
            for i in np.flatnonzero(usage.played | usage.end_board)]
    return sorted(rows, key=lambda r: (-r["played"], -r["endBoard"]))


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
    # Duels cut off (Lua budget, decision limit) are not counted, so a matchup can have no games. A loop
    # loss counts as a win for the other side (since the loop blocking; earlier runs dropped those duels).
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
    if not latest.exists() and old.exists():  # an earlier save stopped between the two renames
        old.rename(latest)
    shutil.rmtree(tmp, ignore_errors=True)
    save_checkpoint(model, tmp)
    torch.save({"optimizer": optimizer.state_dict(), "iteration": iteration}, tmp / STATE_FILE)
    shutil.rmtree(old, ignore_errors=True)
    if latest.exists():
        latest.rename(old)
    tmp.rename(latest)
    shutil.rmtree(old, ignore_errors=True)


def save_checkpoint_whole(model: Model, path: Path) -> None:
    """save_checkpoint into <path>.tmp, then renamed: a kill midway leaves no half checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    save_checkpoint(model, tmp)
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def drop_after(path: Path, iteration: int) -> None:
    """Remove log lines of iterations after `iteration` (run again after a resume) and a line cut
    off by a kill."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line)["iteration"] <= iteration:
                kept.append(line)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
    os.replace(tmp, path)


def error_reasons(messages: list[str], top: int = 3) -> dict[str, int]:
    return dict(Counter((m.split(" (seed")[0].splitlines() or [""])[0][:120] for m in messages).most_common(top))


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
    if args.resume and (out / "run.json").exists():
        restore_run_settings(args, json.loads((out / "run.json").read_text(encoding="utf-8")))
    if args.resume and saved is None and (out / "metrics.jsonl").exists():
        raise SystemExit(f"--resume: {out} has a run but no saved state to resume from")
    if args.resume and saved is not None:
        model = load_checkpoint(saved, args.device, dropout=0.0)
        state = torch.load(saved / STATE_FILE, map_location=args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=ppo.lr)
        optimizer.load_state_dict(state["optimizer"])
        iteration = state["iteration"]
        for log in ("metrics.jsonl", "arena.jsonl", "cards.jsonl"):
            drop_after(out / log, iteration)
    elif args.init is not None:
        model = load_checkpoint(args.init, args.device, dropout=0.0)
        save_checkpoint_whole(model, out / "checkpoints" / f"{0:06d}")
        shape = lambda c: (c.d_model, c.heads, c.layers, c.ff)  # noqa: E731
        if shape(SIZES[args.size]) != shape(model.config):
            print(f"--init: the model keeps its own size, not --size {args.size}", flush=True)
        if args.reset_heads:  # checkpoint 0 above keeps the old heads: the arena's "first" is the init model
            reset_heads(model)
        if args.extend_seen:
            print(f"--extend-seen: {extend_seen(model, args.decks)} more cards with a name embedding", flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=ppo.lr)
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

    for checkpoint in sorted((out / "checkpoints").glob("*")) if args.resume or args.init else []:  # league drops old ones
        if checkpoint.name.isdigit() and int(checkpoint.name) <= iteration:
            try:
                admit(checkpoint)
            except Exception as error:  # e.g. cut off by a kill before checkpoints were written whole
                print(f"skipping checkpoint {checkpoint.name}: {error}", flush=True)
    usage = rollout.CardUsage(model.card_features.size)
    extra = rollout.ExtraDeckUsage(model.card_features)
    collector = rollout.AsyncCollector(env, batch_size=max(1, args.envs // 2), segment=args.segment,
                                       league=league if args.league_share > 0 else None, card_usage=usage,
                                       extra_usage=extra,
                                       tempo=rollout.Tempo(args.tempo_start, args.tempo_slope, args.tempo_floor, args.tempo_losses)
                                       if args.tempo_slope > 0 else None, skip_forced=args.skip_forced)
    names = card_names(model.card_features)
    # With --overlap an actor copy plays the next iteration's duels while `model` trains; it gets the
    # learner's weights before each collection, so its data is one update old (PPO's ratio uses the
    # logprobs stored with it, so the update stays consistent).
    actor = copy.deepcopy(model) if args.overlap else model
    policy = rollout.ModelPolicy(actor, args.device, seed=args.seed, bf16=ppo.bf16)
    prefetch = Prefetch(collector, policy, args.steps) if args.overlap else None
    admitting = None  # an arena checkpoint for the league, admitted while no collection runs
    deadline = time.time() + args.hours * 3600 if args.hours > 0 else None
    search_data = None
    if args.search_data is not None:
        from .expert import SearchData

        args.search_data.mkdir(parents=True, exist_ok=True)
        search_data = SearchData(args.search_data, args.search_window, seed=args.seed)
    saved_iteration, failing = iteration, 0
    loops = env.blocked_answers, env.loop_losses  # duels end between iterations too: carry the baseline
    try:
        while iteration < args.iterations and (deadline is None or time.time() < deadline):
            start = time.time()
            if prefetch is not None:
                segments, stats, collect_seconds = prefetch.get()
            else:
                segments, stats = collector.collect(policy, args.steps)
                collect_seconds = time.time() - start
            if (iteration + 1) % args.arena_every == 0:  # the card logs up to here; the next collection may start
                cards_line = {"cards": card_report(usage, names), "extra": extra_report(extra, names), **extra.summary()}
                usage.reset()
                extra.reset()
            if admitting is not None:  # the league only changes while nothing plays against it
                admit(admitting)
                admitting = None
            last = iteration + 1 >= args.iterations or (deadline is not None and time.time() >= deadline)
            if prefetch is not None:
                actor.load_state_dict(model.state_dict())
                prefetch = Prefetch(collector, policy, args.steps) if not last else None
            collected = time.time()
            steps = build_steps(segments, ppo.gamma, ppo.lam)
            config = replace(ppo, entropy_coef=entropy_coef(args, iteration))
            auxiliary = None
            if search_data is not None:
                search_new = search_data.refresh()
                auxiliary = search_auxiliary(model, search_data, args, ppo.micro_batch)
            history = ppo_update(model, optimizer, steps, config, args.device, seed=args.seed * 1_000_003 + iteration + 1,
                                 auxiliary=auxiliary)
            iteration += 1
            done = time.time()
            finished = stats.duels + stats.league_duels + stats.errors
            line = {"iteration": iteration, "time": round(done, 1), "decisions": len(steps),
                    "decisionsPerSecond": round(stats.decisions / max(collect_seconds, 1e-6)),
                    "collectSeconds": round(collect_seconds, 2), "waitSeconds": round(collected - start, 2),
                    "updateSeconds": round(done - collected, 2),
                    "duels": stats.duels, "leagueDuels": stats.league_duels, "errors": stats.errors,
                    "errorReasons": error_reasons(stats.error_messages),
                    # loop protection: answers blocked and duels lost by looping anyway
                    "blockedAnswers": env.blocked_answers - loops[0],
                    "loopLosses": env.loop_losses - loops[1],
                    "forcedDecisions": stats.forced_decisions,
                    "leagueDataShare": round(stats.league_decisions / max(1, stats.learner_decisions), 3),
                    **({"searchSamples": len(search_data.samples), "searchNew": search_new} if search_data else {}),
                    "wins": stats.wins, "league": league.scores(), "entropyCoef": round(config.entropy_coef, 5),
                    "endTurn": round(float(np.mean(stats.end_turns)), 2) if stats.end_turns else None,
                    "endByTurn5": round(float(np.mean(np.array(stats.end_turns) <= 5)), 3) if stats.end_turns else None,
                    **{k: round(v, 5) if isinstance(v, float) else v for k, v in history[-1].items()}}
            loops = loops[0] + line["blockedAnswers"], loops[1] + line["loopLosses"]
            with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
            print(json.dumps(line), flush=True)
            failing = failing + 1 if finished >= 10 and stats.errors > finished / 2 else 0
            if failing >= 3:
                print(f"WARNING: more than half of the finished duels failed for {failing} iterations: "
                      f"{line['errorReasons']}", flush=True)

            if iteration % args.arena_every == 0:
                checkpoint = out / "checkpoints" / f"{iteration:06d}"
                save_checkpoint_whole(model, checkpoint)
                opponents = {"previous": previous} if previous is not None else {}
                if first is not None and first is not previous:
                    opponents["first"] = first
                result = {"iteration": iteration, **arena(model, opponents, args, iteration),
                          **{k: v for k, v in cards_line.items() if k not in ("cards", "extra")}}
                with open(out / "arena.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)
                with open(out / "cards.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iteration": iteration, **cards_line}) + "\n")
                admitting = checkpoint
            if iteration % args.save_every == 0 or iteration % args.arena_every == 0:
                save_latest(model, optimizer, out, iteration)
                saved_iteration = iteration
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        if prefetch is not None:  # an interrupted run: let the collection in flight finish briefly
            prefetch.thread.join(timeout=30)
        if iteration > saved_iteration:  # the deadline, the last iterations or an interrupt
            save_latest(model, optimizer, out, iteration)
            print(f"saved iteration {iteration} to {out / 'latest'}", flush=True)


if __name__ == "__main__":
    if "--no-jemalloc" not in sys.argv[1:]:
        use_jemalloc(sys.argv[1:])
    main()
