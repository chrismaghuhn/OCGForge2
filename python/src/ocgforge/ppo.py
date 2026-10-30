"""PPO core, ported from Argentum P1: GAE, the clipped loss and one update over collected decisions.

Rollout collection is not here; it will drive the C++ duels directly. What this module needs per
decision is a features.py sample plus:

    chosen       index of the candidate that was played
    logprob      its log-probability under the policy that played it
    value        the value head's estimate at the time

grouped into trajectories of one seat: (decisions, result) for a whole duel, with the duel result
from that seat's view (+1 win, -1 loss, 0 draw), or (decisions, result, bootstrap) for a segment
that the duel continues after: then `bootstrap` is the value at that seat's next decision (and
result 0), or None when the duel ended after the segment. Reward only arrives at the end.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .features import collate
from .model import Model


@dataclass(frozen=True)
class PPOConfig:
    epochs: int = 3
    batch_size: int = 512
    micro_batch: int = 128  # forward/backward chunk of a batch (gradients accumulate), bounds GPU memory
    lr: float = 1.5e-4
    clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1  # KL to a fixed reference policy; 0 = off
    target_kl: float = 0.03  # stop the update once the policy moved this far from the rollout policy
    max_grad_norm: float = 0.5
    # bf16 autocast on CUDA, ~3x faster updates; rollouts must use it too (ModelPolicy(bf16=True)),
    # so the stored logprobs differ from the update's only by noise (~0.02 in log space).
    bf16: bool = False


def gae(values: Sequence[float], final_reward: float, gamma: float, lam: float,
        bootstrap: float | None = None) -> tuple[list[float], list[float]]:
    """Generalized advantage estimation for a trajectory whose only reward arrives after the last
    decision. Returns (advantages, returns). The episode ends after the last step, unless
    `bootstrap` gives the value of the state that follows it."""
    advantages = [0.0] * len(values)
    running = 0.0
    for t in reversed(range(len(values))):
        last = t == len(values) - 1
        reward = final_reward if last else 0.0
        next_value = (0.0 if bootstrap is None else bootstrap) if last else values[t + 1]
        delta = reward + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages, [a + v for a, v in zip(advantages, values)]


def build_steps(trajectories: Sequence[tuple], gamma: float, lam: float) -> list[dict]:
    """Trajectories (see the module docstring) -> decisions with `advantage` and `return`,
    advantages normalized over the whole batch."""
    steps: list[dict] = []
    for decisions, result, *rest in trajectories:
        bootstrap = rest[0] if rest else None
        advantages, returns = gae([float(d["value"]) for d in decisions], result, gamma, lam, bootstrap)
        for decision, advantage, ret in zip(decisions, advantages, returns):
            steps.append({**decision, "advantage": advantage, "return": ret})
    if steps:
        advantages = torch.tensor([s["advantage"] for s in steps])
        mean, std = advantages.mean(), advantages.std().clamp(min=1e-6) if len(steps) > 1 else torch.tensor(1.0)
        for s in steps:
            s["advantage"] = float((s["advantage"] - mean) / std)
    return steps


def collate_steps(steps: Sequence[dict], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    batch = collate(steps, device)
    for name in ("logprob", "value", "advantage", "return"):
        batch[name] = torch.tensor([float(s[name]) for s in steps], device=device)
    if "ref_logprobs" in steps[0]:
        width = batch["candidate_kind"].shape[1]
        rows = torch.full((len(steps), width), float("-inf"))
        for i, s in enumerate(steps):
            rows[i, : len(s["ref_logprobs"])] = torch.as_tensor(s["ref_logprobs"])
        batch["ref_logprobs"] = rows.to(device)
    return batch


def ppo_loss(model: Model, batch: dict[str, torch.Tensor], config: PPOConfig) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.bf16 and batch["chosen"].is_cuda):
        scores, value = model(batch)
    logprobs = torch.log_softmax(scores.float(), dim=-1)
    valid = torch.isfinite(logprobs)
    probs = logprobs.exp()
    new_logprob = logprobs.gather(1, batch["chosen"].unsqueeze(1)).squeeze(1)

    advantage = batch["advantage"]
    log_ratio = new_logprob - batch["logprob"]
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - config.clip, 1 + config.clip) * advantage)
    policy_loss = -surrogate.mean()
    value_loss = torch.mean((value.float() - batch["return"]) ** 2)
    safe = torch.where(valid, logprobs, torch.zeros_like(logprobs))
    entropy = -(probs * safe).sum(-1).mean()
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

    kl_ref = torch.zeros((), device=scores.device)
    if config.kl_coef > 0 and "ref_logprobs" in batch:
        reference = torch.where(valid, batch["ref_logprobs"], torch.zeros_like(logprobs))
        kl_ref = (probs * (safe - reference)).sum(-1).mean()
        loss = loss + config.kl_coef * kl_ref

    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > config.clip).float().mean()
    stats = {"loss": loss.item(), "policyLoss": policy_loss.item(), "valueLoss": value_loss.item(),
             "entropy": entropy.item(), "klRef": kl_ref.item(), "approxKl": approx_kl.item(),
             "clipFraction": clip_fraction.item()}
    return loss, stats


@torch.no_grad()
def attach_reference_logprobs(reference: Model, steps: Sequence[dict], device: torch.device | str = "cpu",
                              batch_size: int = 512) -> None:
    reference.eval()
    for start in range(0, len(steps), batch_size):
        chunk = steps[start:start + batch_size]
        scores, _ = reference(collate(chunk, device))
        logprobs = torch.log_softmax(scores.float(), dim=-1).cpu()
        for row, step in zip(logprobs, chunk):
            step["ref_logprobs"] = row[: len(step["candidate_kind"])].clone()


def _size(step: dict) -> tuple[int, int, int]:
    """What collate pads: events, card tokens, candidates."""
    return len(step.get("event_kind", ())), len(step["token_card"]), len(step["candidate_kind"])


def ppo_update(model: Model, optimizer: torch.optim.Optimizer, steps: Sequence[dict], config: PPOConfig,
               device: torch.device | str = "cpu", seed: int = 0, auxiliary=None) -> list[dict[str, float]]:
    """Clipped-PPO epochs over `steps` (from build_steps). Returns per-epoch mean stats; stops early
    once the approximate KL to the rollout policy exceeds 1.5 x target_kl.

    The forward passes run in eval mode: rollouts are played in eval mode, and dropout or name
    dropout here would make the probability ratio and KL measure masking noise, not policy change.
    Gradients flow as usual.

    `auxiliary()`, if given, is called once per minibatch, after its PPO gradient: it backpropagates an
    extra term (e.g. imitating the search) into the same step and returns its stats, or None."""
    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.eval()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        stopped = False
        order = list(range(len(steps)))
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            # Similar sizes per micro-batch: collate pads to the largest sample, and the summed
            # gradient of the batch does not depend on how it is split.
            chosen = sorted(order[start:start + config.batch_size], key=lambda i: _size(steps[i]))
            optimizer.zero_grad(set_to_none=True)
            stats: dict[str, float] = defaultdict(float)
            for micro in range(0, len(chosen), config.micro_batch):
                part = chosen[micro:micro + config.micro_batch]
                share = len(part) / len(chosen)
                loss, part_stats = ppo_loss(model, collate_steps([steps[i] for i in part], device), config)
                (loss * share).backward()
                for key, value in part_stats.items():
                    stats[key] += value * share
            if auxiliary is not None:
                for key, value in (auxiliary() or {}).items():
                    stats[key] += value
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            for key, value in stats.items():
                totals[key] += value
            batches += 1
            if stats["approxKl"] > 1.5 * config.target_kl:
                stopped = True
                break
        history.append({**{k: v / max(batches, 1) for k, v in totals.items()},
                        "epoch": epoch, "batches": batches, "stoppedEarly": stopped})
        if stopped:
            break
    return history
