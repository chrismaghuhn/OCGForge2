"""Precompute a frozen sentence embedding of every card's text, ported from Argentum P1.

    python -m ocgforge.embed_cards --cdb .cache/babelcdb/cards.cdb --out .cache/card_text_emb.npz

Text per card: a short type line ("Effect Monster, LIGHT, Warrior, Level 4") and the card text with
its own name replaced by "this card", encoded by a small pretrained text encoder (default
BAAI/bge-small-en-v1.5, MIT; CLS pooling, L2 normalized). The encoder is never trained and not needed
at play time: the vectors go into the model as a buffer (see load_text_embeddings).
Needs the `embed` extra (transformers).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cards import TYPE_LINK, TYPE_MONSTER, Card, CardFeatures, CardTable

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"

_TYPE_NAMES = {0x1: "Monster", 0x2: "Spell", 0x4: "Trap", 0x10: "Normal", 0x20: "Effect", 0x40: "Fusion",
               0x80: "Ritual", 0x200: "Spirit", 0x400: "Union", 0x800: "Gemini", 0x1000: "Tuner",
               0x2000: "Synchro", 0x4000: "Token", 0x10000: "Quick-Play", 0x20000: "Continuous",
               0x40000: "Equip", 0x80000: "Field", 0x100000: "Counter", 0x200000: "Flip", 0x400000: "Toon",
               0x800000: "Xyz", 0x1000000: "Pendulum", 0x4000000: "Link"}
_ATTRIBUTES = {0x1: "EARTH", 0x2: "WATER", 0x4: "FIRE", 0x8: "WIND", 0x10: "LIGHT", 0x20: "DARK", 0x40: "DIVINE"}
_RACES = ("Warrior", "Spellcaster", "Fairy", "Fiend", "Zombie", "Machine", "Aqua", "Pyro", "Rock", "Winged Beast",
          "Plant", "Insect", "Thunder", "Dragon", "Beast", "Beast-Warrior", "Dinosaur", "Fish", "Sea Serpent",
          "Reptile", "Psychic", "Divine-Beast", "Creator God", "Wyrm", "Cyberse", "Illusion", "Cyborg",
          "Magical Knight", "High Dragon", "Omega Psychic")


def card_sentence(card: Card) -> str:
    types = " ".join(name for bit, name in _TYPE_NAMES.items() if card.type & bit)
    parts = [types]
    if card.type & TYPE_MONSTER:
        parts.append(_ATTRIBUTES.get(card.attribute, ""))
        parts.append(next((r for i, r in enumerate(_RACES) if card.race >> i & 1), ""))
        level = card.level & 0xFF
        parts.append(f"Link {level}" if card.type & TYPE_LINK else f"Level {level}")
    header = ", ".join(p for p in parts if p)
    text = card.text.replace(f'"{card.name}"', "this card").replace(card.name, "this card").strip()
    return f"{header}. {text}".strip()


def load_text_embeddings(path: Path, card_features: CardFeatures) -> np.ndarray:
    """embed_cards output -> [card_features.size, dim] rows by card index (zero rows for unknown cards)."""
    data = np.load(path)
    index = card_features.index_map()
    rows = np.zeros((card_features.size, data["vectors"].shape[1]), dtype=np.float32)
    for code, vector in zip(data["codes"], data["vectors"]):
        i = index.get(int(code))
        if i is not None:
            rows[i] = vector
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cdb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModel, AutoTokenizer

    table = CardTable.from_cdb(args.cdb)
    sentences = [card_sentence(card) for card in table.cards]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder).to(device).eval()
    vectors = np.zeros((len(sentences), encoder.config.hidden_size), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(sentences), args.batch_size):
            batch = sentences[start:start + args.batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            cls = encoder(**tokens).last_hidden_state[:, 0]
            vectors[start:start + len(batch)] = torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, codes=np.array([c.code for c in table.cards], dtype=np.int64), vectors=vectors,
                        encoder=np.array(args.encoder))
    print(f"embedded {len(sentences)} cards with {args.encoder} ({vectors.shape[1]} dims) -> {args.out}")


if __name__ == "__main__":
    main()
