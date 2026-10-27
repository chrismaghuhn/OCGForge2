"""The model's input for one decision, and batching.

One decision is a dict of numpy arrays, produced by the C++ encoder (src/encode/encoder.cpp, exposed
as ocgforge._core); tests/test_core.py checks that both sides use the same names and order:

    token_card         [T]      card index (cards.CardFeatures), UNK for a hidden card
    token_zone         [T]      index into ZONES
    token_side         [T]      0 = acting player, 1 = opponent
    token_numeric      [T, len(TOKEN_NUMERIC)]
    global_numeric     [len(GLOBAL_NUMERIC)]
    phase              []       index into PHASES
    decision           []       index into DECISIONS
    candidate_kind     [C]      index into CANDIDATE_KINDS
    candidate_source   [C]      token index of the card the candidate acts with, -1 = none
    candidate_card     [C]      card index of that card when it has no token (own cards only, e.g. a
                                deck search) or of the card offering an option; 0 = none
    candidate_zone     [C]      index into ZONES of the candidate's card or place, 0 = none
    candidate_targets  [C, MAX_TARGETS]  token indices, -1 = none
    candidate_numeric  [C, len(CANDIDATE_NUMERIC)]

Training adds `chosen` (candidate index) and PPO fields (see ppo.py). Every index vocabulary keeps
0 for padding.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

MAX_TARGETS = 4

# protocol::DecisionRequestKind (include/ygo/protocol/continuation.hpp), shifted by one for padding.
DECISIONS = ("<pad>", "idle_command", "battle_command", "chain", "option", "card_selection", "tribute", "sum",
             "place", "counter", "ordering", "announcement", "unselect_card", "position", "yes_no")

# observation::SemanticZone (include/ygo/observation/observed_zone.hpp), plus the chain.
ZONES = ("<pad>", "unknown", "main_deck", "hand", "monster_zone", "spell_trap_zone", "graveyard", "banished",
         "extra_deck", "field_zone", "pendulum_zone", "overlay", "chain")

# Index = log2(PHASE_* bit) + 1.
PHASES = ("<pad>", "draw", "standby", "main1", "battle_start", "battle_step", "damage", "damage_calculation",
          "battle", "main2", "end")

# Finer than protocol::ActionKind: idle and battle commands are split by what they do.
CANDIDATE_KINDS = (
    "<pad>",
    "summon", "special_summon", "reposition", "set_monster", "set_spell_trap", "activate",
    "to_battle_phase", "to_end_phase", "shuffle_hand",
    "attack", "to_main_phase_2",
    "chain_activate", "chain_pass",
    "option", "select_card", "unselect_card", "announce", "place", "position", "yes", "no",
    "pick", "finish", "cancel", "assign_amount",
)

TOKEN_NUMERIC = (
    "faceUp",
    "attackPosition",
    "sequence",  # zone column / 7
    "atk",  # current values / 5000
    "def",
    "level",  # current level / rank / 12
    "overlayCount",
    "counters",
    "disabled",
    "selected",  # already picked in an ongoing multi-card selection
    "chainIndex",  # position on the chain / 10, 0 = not chained
)

_PLAYER = ("lp", "hand", "deck", "extra", "graveyard", "banished", "monsters", "spellTraps")
GLOBAL_NUMERIC = (
    *(f"self_{name}" for name in _PLAYER),
    *(f"opp_{name}" for name in _PLAYER),
    "turn",
    "isOwnTurn",
    "chainSize",
    "selectMin",  # multi-card selections
    "selectMax",
    "selectedCount",
)

CANDIDATE_NUMERIC = (
    "hasSource",
    "targetCount",
    "effectIndex",  # which effect of the source card (str1..str16) / 16
    "value",  # option / number / amount, scaled
    "faceUp",  # position candidates
    "attackPosition",
    "zone",  # column / 7 of the candidate's card or place
    "opponentSide",  # that card or place is the opponent's
    "variant",  # n-th repeat of an otherwise identical candidate (e.g. one per summon procedure) / 4
)

_WIDTHS = {"token_numeric": (len(TOKEN_NUMERIC),), "candidate_targets": (MAX_TARGETS,),
           "candidate_numeric": (len(CANDIDATE_NUMERIC),)}
_TOKEN_FIELDS = ("token_card", "token_zone", "token_side", "token_numeric")
_CANDIDATE_FIELDS = ("candidate_kind", "candidate_source", "candidate_card", "candidate_zone", "candidate_targets",
                     "candidate_numeric")
_PAD_VALUES = {"token_side": 2, "candidate_source": -1, "candidate_targets": -1}


def collate(samples: Sequence[dict], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Pad decisions into one batch. Adds `token_pad` and `candidate_pad` (True = padding)."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    for i, s in enumerate(samples):
        check(s, f"decision {i}")
    b = len(samples)
    t = max(1, max(len(s["token_card"]) for s in samples))
    c = max(len(s["candidate_kind"]) for s in samples)
    out: dict[str, torch.Tensor] = {}
    for name in _TOKEN_FIELDS + _CANDIDATE_FIELDS:
        width = (t if name.startswith("token") else c,) + _WIDTHS.get(name, ())
        dtype = np.float32 if name.endswith("numeric") else np.int64
        array = np.full((b, *width), _PAD_VALUES.get(name, 0), dtype=dtype)
        for i, s in enumerate(samples):
            values = np.asarray(s[name], dtype=dtype)
            if len(values):  # an empty list has shape (0,), not (0, width)
                array[i, : len(values)] = values
        out[name] = torch.from_numpy(array)
    token_counts = torch.tensor([len(s["token_card"]) for s in samples])
    candidate_counts = torch.tensor([len(s["candidate_kind"]) for s in samples])
    out["token_pad"] = torch.arange(t).unsqueeze(0) >= token_counts.unsqueeze(1)
    out["token_pad"][:, 0] &= token_counts > 0  # keep one attendable token per row
    out["candidate_pad"] = torch.arange(c).unsqueeze(0) >= candidate_counts.unsqueeze(1)
    out["global_numeric"] = torch.from_numpy(np.stack([np.asarray(s["global_numeric"], np.float32) for s in samples]))
    for name in ("phase", "decision", "chosen"):
        if name in samples[0]:
            out[name] = torch.tensor([int(s[name]) for s in samples])
    return {k: v.to(device, non_blocking=True) for k, v in out.items()}


def check(sample: dict, where: str = "decision") -> None:
    """Reject inputs the model would silently mis-read: no candidates (all scores -inf, NaN loss),
    and candidate source/target indices outside the decision's own tokens (these would read
    another sample's padding once batched)."""
    tokens = len(sample["token_card"])
    if len(sample["candidate_kind"]) == 0:
        raise ValueError(f"{where}: no candidates")
    for name in ("candidate_source", "candidate_targets"):
        values = np.asarray(sample[name])
        if values.size and (values.min() < -1 or values.max() >= tokens):
            raise ValueError(f"{where}: {name} must be -1 or a token index below {tokens}")
