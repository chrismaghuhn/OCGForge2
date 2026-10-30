"""Expert iteration, one round: the honest search plays, the network learns its moves.

generate: self-play duels in which both seats plan their own turns with the honest search
(search.HonestSearchPolicy). At every decision a seat plays from its plan it stores the decision and
a target distribution over the answers: the posterior over the searched plans that still agree with
this turn's play, each weighted by exp(mean score / temperature) (a regularized improved policy in
the spirit of Grill et al. 2020, "Monte-Carlo tree search as regularized policy optimization"). The
duel's result for that seat becomes the value target. Several workers write shards in parallel.

distill: the network is trained on those shards: cross-entropy to the target distribution plus the
value error, from a checkpoint (usually the one that searched), and saved as a new checkpoint; a
held-out share reports how well the targets are matched.

    python -m ocgforge.expert generate --checkpoint CKPT --decks ... --duels 200 --workers 3 --out DIR
    python -m ocgforge.expert distill --checkpoint CKPT --data DIR --out NEWCKPT
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import features, rollout


def compact(row: dict) -> dict:
    """A Duel.encode row in the smallest dtypes (collate widens them again)."""
    return {k: v.astype(features._COMPACT[k]) if k in features._COMPACT else v for k, v in row.items()}


def generate_shard(checkpoint: Path, decks: list[Path], duels: int, k: int, seed: int, out: Path,
                   device: str = "cuda", temperature: float = 0.05) -> dict:
    """Plays `duels` duels with both seats searching and pickles the samples to `out`."""
    from .model import load_checkpoint
    from .search import DuelSpec, GreedyPolicy, HonestSearchPolicy

    model = load_checkpoint(checkpoint, device, dropout=0.0)
    card_index = model.card_features.index_map()
    policy = GreedyPolicy(model, device, sample=True, seed=seed)
    rng = random.Random(seed)
    samples: list[dict] = []
    started, results = time.time(), [0, 0, 0]
    for n in range(duels):
        pending: list[dict] = []

        def on_target(duel, keys, target, pending=pending):
            pending.append({**compact(duel.encode(card_index)), "target": np.asarray(target, np.float32),
                            "seat": duel.player})

        seats = [HonestSearchPolicy(policy, card_index, k, k, on_target=on_target, temperature=temperature)
                 for _ in range(2)]
        spec = DuelSpec(rng.choice(decks), rng.choice(decks), rng.getrandbits(63), rng.randrange(2))
        duel, actions = spec.start(), []
        while not duel.finished and duel.decision_count < 3000:
            key = seats[duel.player].choose(spec, actions, duel)
            duel.act(key)
            actions.append(key)
        winner = duel.winner if duel.finished else None
        results[2 if winner is None or winner > 1 else winner] += 1
        for sample in pending:
            seat = sample.pop("seat")
            sample["result"] = 0.0 if winner is None or winner > 1 else (1.0 if winner == seat else -1.0)
        samples.extend(pending)
        print(f"[{seed}] duel {n + 1}/{duels}: {len(samples)} samples, {time.time() - started:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(samples, f)
    return {"shard": str(out), "samples": len(samples), "duels": duels, "results": results,
            "seconds": round(time.time() - started)}


def _shard_job(args: tuple) -> dict:
    return generate_shard(*args)


def generate(checkpoint: Path, decks: list[Path], duels: int, k: int, workers: int, out: Path, seed: int = 0,
             device: str = "cuda", temperature: float = 0.05) -> list[dict]:
    """`duels` duels split over `workers` processes, one shard each."""
    per = [duels // workers + (i < duels % workers) for i in range(workers)]
    jobs = [(checkpoint, decks, per[i], k, seed * 1000 + i, out / f"shard-{seed}-{i}.pkl", device, temperature)
            for i in range(workers) if per[i]]
    with multiprocessing.get_context("spawn").Pool(len(jobs)) as pool:
        return pool.map(_shard_job, jobs)


def load_samples(directory: Path) -> list[dict]:
    samples = []
    for shard in sorted(Path(directory).glob("shard-*.pkl")):
        with open(shard, "rb") as f:
            samples.extend(pickle.load(f))
    return samples


def distill_loss(model, batch: dict, targets: torch.Tensor, results: torch.Tensor,
                 value_coef: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Cross-entropy to the target distribution (padding and zero-target answers ignored) plus the
    value head's squared error against the duel result."""
    scores, value = model(batch)
    logprobs = torch.log_softmax(scores.float(), dim=-1)
    picked = torch.where(targets > 0, logprobs, torch.zeros_like(logprobs))
    policy_loss = -(targets * picked).sum(-1).mean()
    value_loss = ((value.float() - results) ** 2).mean()
    agree = (logprobs.argmax(-1) == targets.argmax(-1)).float().mean()
    return policy_loss + value_coef * value_loss, {"policy": float(policy_loss.detach()), "value": float(value_loss.detach()),
                                                   "agree": float(agree)}


def _batch(samples: list[dict], device: str):
    batch = features.collate(samples, device)
    width = batch["candidate_kind"].shape[1]
    targets = torch.zeros(len(samples), width)
    for i, s in enumerate(samples):
        targets[i, : len(s["target"])] = torch.from_numpy(s["target"])
    results = torch.tensor([s["result"] for s in samples], dtype=torch.float32)
    return batch, targets.to(device), results.to(device)


def distill(checkpoint: Path, data: list[Path], out: Path, epochs: int = 2, lr: float = 1e-5,
            batch_size: int = 256, device: str = "cuda", holdout: float = 0.05, seed: int = 0,
            micro_batch: int = 64) -> dict:
    """Trains `checkpoint` on the samples in `data` and saves the result to `out`; each batch is
    run in `micro_batch` chunks whose gradients add up (GPU memory)."""
    from .model import load_checkpoint, save_checkpoint

    model = load_checkpoint(checkpoint, device, dropout=0.0)
    samples = [s for d in data for s in load_samples(d)]
    random.Random(seed).shuffle(samples)
    cut = max(1, int(len(samples) * holdout))
    held, train = samples[:cut], samples[cut:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        totals, n = {"policy": 0.0, "value": 0.0, "agree": 0.0}, 0
        for start in range(0, len(held), micro_batch):
            part = held[start:start + micro_batch]
            _, stats = distill_loss(model, *_batch(part, device))
            for key in totals:
                totals[key] += stats[key] * len(part)
            n += len(part)
        return {key: round(value / n, 4) for key, value in totals.items()}

    history = [{"epoch": 0, **evaluate()}]
    print(json.dumps(history[-1]), flush=True)
    rng = random.Random(seed + 1)
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train)
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            optimizer.zero_grad()
            for micro in range(0, len(batch), micro_batch):
                part = batch[micro:micro + micro_batch]
                loss, _ = distill_loss(model, *_batch(part, device))
                (loss * len(part) / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        history.append({"epoch": epoch, **evaluate()})
        print(json.dumps(history[-1]), flush=True)
    model.eval()
    save_checkpoint(model, out)
    summary = {"checkpoint": str(checkpoint), "samples": len(samples), "train": len(train), "held_out": len(held),
               "history": history}
    (Path(out) / "distill.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="search-played duels to training samples")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--decks", type=Path, nargs="+", default=sorted((rollout.REPO / "decks").glob("*.ydk")))
    g.add_argument("--duels", type=int, default=100)
    g.add_argument("--k", type=int, default=4, help="plans and worlds per search")
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--temperature", type=float, default=0.05, help="of the posterior over plans")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("distill", help="train a checkpoint on the samples")
    d.add_argument("--checkpoint", type=Path, required=True)
    d.add_argument("--data", type=Path, nargs="+", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--epochs", type=int, default=2)
    d.add_argument("--lr", type=float, default=1e-5)
    d.add_argument("--batch-size", type=int, default=256)
    d.add_argument("--micro-batch", type=int, default=64, help="chunk of a batch run at once (GPU memory)")
    d.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.command == "generate":
        for line in generate(args.checkpoint, args.decks, args.duels, args.k, args.workers, args.out, args.seed,
                             args.device, args.temperature):
            print(json.dumps(line), flush=True)
    else:
        summary = distill(args.checkpoint, args.data, args.out, args.epochs, args.lr, args.batch_size, args.device,
                          micro_batch=args.micro_batch)
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
