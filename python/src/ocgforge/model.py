"""Candidate-scoring policy with a value head, ported from Argentum P1 (set transformer v2).

Every visible card is a token: what the card *is* (looked up per card index from cards.CardFeatures:
types, attribute, race, archetypes, stats, rules-text terms, optional frozen text embedding) plus
where it is and its current state. A global token carries life points, counts, phase and the kind of
question being asked. A small transformer encodes the set; each legal candidate is scored from its
kind, its source token and its target tokens against the encoded global token, and a softmax over
the candidates is the policy. The value head predicts the duel result (-1..1) for the acting player.

Name dropout replaces the learned per-card embedding by "unknown" during training only, so the model
has to play from what a card does rather than memorize names; that is what lets one base model
handle cards it never trained with.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import cards, features
from .cards import UNK, CardFeatures

ARCHITECTURE = "ocgforge-set-transformer-candidate-scorer@2"  # @2: event memory


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    heads: int = 4
    layers: int = 3
    ff: int = 256
    dropout: float = 0.1
    name_dropout: float = 0.5
    text_embedding_dim: int = 0  # frozen sentence embedding per card (embed_cards.py), 0 = off

    def to_json(self) -> dict:
        return {"architecture": ARCHITECTURE, **asdict(self)}

    @classmethod
    def from_json(cls, payload: dict) -> "ModelConfig":
        if payload.get("architecture") != ARCHITECTURE:
            raise ValueError(f"unexpected model architecture {payload.get('architecture')!r}")
        return cls(**{k: payload[k] for k in asdict(cls()) if k in payload})


class Model(nn.Module):
    def __init__(self, card_features: CardFeatures, config: ModelConfig = ModelConfig(),
                 text_embeddings: np.ndarray | None = None):
        """`text_embeddings`: [card_features.size, config.text_embedding_dim] rows by card index."""
        super().__init__()
        d = config.d_model
        self.config = config
        self.card_features = card_features
        f = card_features
        self.register_buffer("card_types", torch.from_numpy(f.types))
        self.register_buffer("card_attribute", torch.from_numpy(f.attribute))
        self.register_buffer("card_race", torch.from_numpy(f.race))
        self.register_buffer("card_setcodes", torch.from_numpy(f.setcodes))
        self.register_buffer("card_numeric", torch.from_numpy(f.numeric))
        self.register_buffer("card_text", torch.from_numpy(f.text))
        self.register_buffer("card_seen", torch.from_numpy(f.seen))
        if config.text_embedding_dim > 0:
            if text_embeddings is None:
                raise ValueError("text_embedding_dim > 0 needs text_embeddings (embed_cards.load_text_embeddings)")
            rows = torch.from_numpy(np.asarray(text_embeddings, dtype=np.float32))
            if rows.shape != (f.size, config.text_embedding_dim):
                raise ValueError(f"text embeddings must be {(f.size, config.text_embedding_dim)}, got {tuple(rows.shape)}")
            self.register_buffer("card_textemb", rows)
            self.text_embedding = nn.Linear(config.text_embedding_dim, d)

        self.name = nn.Embedding(f.size, d, padding_idx=0)
        self.types = nn.EmbeddingBag(CardFeatures.TYPE_IDS, d, mode="sum", padding_idx=0)
        self.attribute = nn.Embedding(CardFeatures.ATTRIBUTE_IDS, d, padding_idx=0)
        self.race = nn.Embedding(CardFeatures.RACE_IDS, d, padding_idx=0)
        self.setcodes = nn.EmbeddingBag(f.setcode_size(), d, mode="sum", padding_idx=0)
        self.text = nn.EmbeddingBag(f.text_size(), d, mode="mean", padding_idx=0)
        self.static_numeric = nn.Linear(len(cards.CARD_NUMERIC), d)

        self.zone = nn.Embedding(len(features.ZONES), d, padding_idx=0)
        self.side = nn.Embedding(3, d)
        self.token_numeric = nn.Linear(len(features.TOKEN_NUMERIC), d)
        self.global_numeric = nn.Linear(len(features.GLOBAL_NUMERIC), d)
        self.phase = nn.Embedding(len(features.PHASES), d, padding_idx=0)
        self.decision = nn.Embedding(len(features.DECISIONS), d, padding_idx=0)
        self.event_kind = nn.Embedding(len(features.EVENT_KINDS), d, padding_idx=0)
        self.event_from = nn.Embedding(len(features.ZONES), d, padding_idx=0)
        self.event_to = nn.Embedding(len(features.ZONES), d, padding_idx=0)
        self.event_numeric = nn.Linear(len(features.EVENT_NUMERIC), d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=config.heads, dim_feedforward=config.ff,
                                           dropout=config.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)

        self.kind = nn.Embedding(len(features.CANDIDATE_KINDS), d, padding_idx=0)
        self.no_source = nn.Parameter(torch.zeros(d))
        self.candidate_numeric = nn.Linear(len(features.CANDIDATE_NUMERIC), d)
        self.score = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1), nn.Tanh())

    def card_static(self, cards_: torch.Tensor) -> torch.Tensor:
        """[...] card indices -> [..., d]: what the card is, from its database facts and text.
        Computed once per distinct card in the batch (the text bag alone reads up to MAX_TEXT
        terms per card), then spread back to every position."""
        shape = cards_.shape
        distinct, inverse = torch.unique(cards_.reshape(-1), return_inverse=True)
        return self._card_static_rows(distinct)[inverse].reshape(*shape, self.config.d_model)

    def _card_static_rows(self, flat: torch.Tensor) -> torch.Tensor:
        x = (
            self.types(self.card_types[flat])
            + self.attribute(self.card_attribute[flat])
            + self.race(self.card_race[flat])
            + self.setcodes(self.card_setcodes[flat])
            + self.text(self.card_text[flat])
            + self.static_numeric(self.card_numeric[flat])
        )
        if self.config.text_embedding_dim > 0:
            x = x + self.text_embedding(self.card_textemb[flat])
        return x

    def card_embedding(self, cards_: torch.Tensor) -> torch.Tensor:
        """card_static plus a learned per-card embedding, which is "unknown" for cards never seen in
        training and, while training, for a random name_dropout share of cards."""
        shown = torch.where(self.card_seen[cards_], cards_, torch.full_like(cards_, UNK))
        if self.training and self.config.name_dropout > 0:
            hide = (torch.rand(cards_.shape, device=cards_.device) < self.config.name_dropout) & (cards_ > UNK)
            shown = shown.masked_fill(hide, UNK)
        return self.name(shown) + self.card_static(cards_)

    def event_tokens(self, batch: dict[str, torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
        """Recent public events as [B, E, d]: what happened, to which card (if it was public), by
        whom, from where to where, how long ago, plus the input of the field card token the event
        provably happened to (so "set a card" is tied to that face-down card).

        Events describe their card by its facts only (card_static), not the learned per-card
        embedding: otherwise a name hidden by name dropout on a token could come back through an
        event about the same card."""
        card = batch["event_card"]
        x = (
            self.event_kind(batch["event_kind"])
            + torch.where((card > 0).unsqueeze(-1), self.card_static(card), torch.zeros(1, device=card.device))
            + self.side(batch["event_side"])
            + self.event_from(batch["event_from"])
            + self.event_to(batch["event_to"])
            + self.event_numeric(batch["event_numeric"])
        )
        link = batch["event_token"]
        if tokens.shape[1] > 0 and link.shape[1] > 0:
            linked = torch.gather(tokens, 1, link.clamp(min=0).unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
            x = x + linked * (link >= 0).unsqueeze(-1)
        return x

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns candidate scores [B, C] (-inf on padding) and values [B]."""
        b = batch["token_card"].shape[0]
        tokens = (
            self.card_embedding(batch["token_card"])
            + self.zone(batch["token_zone"])
            + self.side(batch["token_side"])
            + self.token_numeric(batch["token_numeric"])
        )
        global_token = (
            self.global_numeric(batch["global_numeric"]) + self.phase(batch["phase"]) + self.decision(batch["decision"])
        ).unsqueeze(1)
        events = self.event_tokens(batch, tokens)
        sequence = torch.cat([global_token, tokens, events], dim=1)
        padding = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=tokens.device), batch["token_pad"],
                             batch["event_pad"]], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        t = tokens.shape[1]
        context, card_tokens = encoded[:, 0], encoded[:, 1:1 + t]

        # Candidates are scored packed (only real ones, all decisions back to back): padding
        # every decision to the batch's largest candidate list would blow up memory when one
        # decision has thousands of candidates.
        row, col = (~batch["candidate_pad"]).nonzero(as_tuple=True)
        width = card_tokens.size(-1)

        def card_token(indices: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
            """card_tokens[rows, indices] with zeros where indices is -1."""
            if card_tokens.size(1) == 0:
                return torch.zeros(*indices.shape, width, device=card_tokens.device, dtype=card_tokens.dtype)
            return card_tokens[rows, indices.clamp(min=0)] * (indices >= 0).unsqueeze(-1)

        source_index = batch["candidate_source"][row, col]
        source = torch.where((source_index >= 0).unsqueeze(-1), card_token(source_index, row), self.no_source)
        target_index = batch["candidate_targets"][row, col]  # [n, MAX_TARGETS]
        targets = card_token(target_index, row.unsqueeze(1).expand_as(target_index)).sum(1)
        targets = targets / (target_index >= 0).sum(-1, keepdim=True).clamp(min=1)
        # Without a source token the candidate's own card (if any) stands in for it.
        own = batch["candidate_card"][row, col]
        own_card = torch.where((own > 0).unsqueeze(-1), self.card_embedding(own), torch.zeros_like(source))
        candidate = (self.kind(batch["candidate_kind"][row, col]) + source + own_card
                     + self.zone(batch["candidate_zone"][row, col]) + targets
                     + self.candidate_numeric(batch["candidate_numeric"][row, col]))
        ctx = context[row]
        # The heads run in fp32 even under autocast: bf16 logits are rounded by up to ~0.1 once the
        # policy is sharp, and rollout and update would then disagree on the logprobs.
        with torch.autocast(context.device.type, enabled=False):
            features_ = torch.cat([candidate, ctx, candidate * ctx], dim=-1).float()
            packed = self.score(features_).squeeze(-1)
            value = self.value(context.float()).squeeze(-1)
        scores = torch.full(batch["candidate_kind"].shape, float("-inf"), device=packed.device, dtype=packed.dtype)
        scores[row, col] = packed
        return scores, value


CONFIG_FILE = "config.json"
CARDS_FILE = "cards.json"
WEIGHTS_FILE = "model.safetensors"


def save_checkpoint(model: Model, directory: Path) -> None:
    """config.json, cards.json (card index and vocabularies) and model.safetensors (weights and card rows)."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(directory / WEIGHTS_FILE))
    (directory / CONFIG_FILE).write_text(json.dumps(model.config.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    (directory / CARDS_FILE).write_text(json.dumps(model.card_features.to_json(), sort_keys=True), encoding="utf-8")


def load_checkpoint(directory: Path, device: torch.device | str = "cpu", dropout: float | None = None) -> Model:
    """Load a checkpoint; `dropout` overrides the stored dropout (PPO uses 0)."""
    from safetensors.torch import load_file

    directory = Path(directory)
    config = ModelConfig.from_json(json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8")))
    if dropout is not None:
        config = replace(config, dropout=dropout)
    card_features = CardFeatures.from_json(json.loads((directory / CARDS_FILE).read_text(encoding="utf-8")))
    placeholder = np.zeros((card_features.size, config.text_embedding_dim), np.float32)  # filled by the weights
    model = Model(card_features, config, text_embeddings=placeholder if config.text_embedding_dim else None)
    model.load_state_dict(load_file(str(directory / WEIGHTS_FILE)))
    # Keep the numpy rows in sync with the loaded buffers, so saving again writes the same table.
    for name, attr in (("card_types", "types"), ("card_attribute", "attribute"), ("card_race", "race"),
                       ("card_setcodes", "setcodes"), ("card_numeric", "numeric"), ("card_text", "text"),
                       ("card_seen", "seen")):
        setattr(card_features, attr, getattr(model, name).numpy().copy())
    return model.to(device)
