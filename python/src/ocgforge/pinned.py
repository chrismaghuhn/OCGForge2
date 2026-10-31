"""Pinned Branded Discord replays as seeds: Duelingbook logs ("<replay id>.txt", read by
`seeds.replay_seed` / `db_logs`) plus the posts that came with them (`descriptions.txt`).

descriptions.txt, one block per replay:
    == 817316-56847305 | Katsune 20.01.24 | Branded Fusion + 2 discards
    endboard: Quem, Cartesia, Mirrorjade (used), set Branded in Red
    (more lines: starting hands, the combo in words)

A seed per log: the log's steps and author, the replay id as origin, the cards the log only discards
(their first move is hand -> GY) left to the filler, the post's end board (field cards only: "in
hand" / "in GY" parts dropped), the post and the log's comments (`A "..."` lines, which db_logs skips)
as notes.

Hook for seeds.load:
    seeds += pinned.pinned_seeds(args.pinned, deck_type, resolve) if args.pinned else []
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import aliases
from . import combos as cb
from . import seeds as sd

AUTHORS = {"katsune": "KatsuneNiko", "colossus": "ScarTheColossus"}  # post signature -> Duelingbook user
COMMENT = re.compile(r'^[A-Z]\s+"(?P<text>[^"]+)"\s*$')  # a log comment: actor, quoted text, no verb
BOARD = re.compile(r"^\s*(?:if [^,]*,\s*)?(?:end\s*board|board)(?:\s+is|\s*:)\s*(?P<board>.+)$", re.IGNORECASE)


@dataclass
class Post:
    replay: str
    author: str  # Duelingbook user name
    date: str
    hand: str  # "Branded Fusion + 2 discards"
    variant: str  # "Puppet lock"
    lines: list[str] = field(default_factory=list)

    def end_board_text(self) -> str:
        return next((m["board"] for line in self.lines if (m := BOARD.match(line))), "")


def read_posts(path: Path) -> dict[str, Post]:
    """Replay id -> post, from descriptions.txt."""
    return read_posts_text(Path(path).read_text(encoding="utf-8"))


def read_posts_text(text: str) -> dict[str, Post]:
    posts: dict[str, Post] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("== "):
            parts = [p.strip() for p in line[3:].split("|")]
            who = (parts[1] if len(parts) > 1 else "").split()
            current = Post(parts[0], AUTHORS.get(who[0].lower(), who[0]) if who else "", " ".join(who[1:]),
                           parts[2] if len(parts) > 2 else "", " | ".join(parts[3:]))
            posts[current.replay] = current
        elif current is not None and line.strip():
            current.lines.append(line.strip())
    return posts


def board_cards(text: str, resolve) -> list[int]:
    """A post's end board ("Mirrorjade + Quem (Cartesia and Albaz in GY) + Mercourier in hand",
    "Lost search Merc, double Jade through Ad Lib") -> the cards on the field."""
    cards = []
    for part in re.split(r"[,+]|\s+and\s+", re.sub(r"\([^)]*\)", "", text)):
        if re.search(r"\bin (hand|gy|grave)\b", part, re.IGNORECASE):
            continue
        part = re.split(r"\s+(?:search|through|use)\b", part.strip(), flags=re.IGNORECASE)[0]
        part = re.sub(r"^(?:(?:set|double|a|the)\s+)+", "", part, flags=re.IGNORECASE).strip()
        if part and (code := resolve(part)) is not None and code not in cards:
            cards.append(code)
    return cards


def pinned_seeds(directory: Path, deck_type: str, resolve) -> list:
    """A seed per "<replay id>.txt" log in `directory` (with `descriptions.txt` beside them)."""
    directory = Path(directory)
    resolve = aliases.with_aliases(resolve)
    posts = read_posts(directory / "descriptions.txt") if (directory / "descriptions.txt").exists() else {}
    seeds = []
    for path in sorted(directory.glob("*-*.txt")):
        seed = sd.replay_seed(path, deck_type, resolve)
        if seed is None:
            continue
        text = path.read_text(encoding="utf-8")
        hand = list(dict.fromkeys(seed.hand))  # replay_seed already left out the discard fodder
        post = posts.get(path.stem)
        comments = [m["text"] for line in text.splitlines() if (m := COMMENT.match(line.strip()))]
        notes = [h for h in [" | ".join(p for p in (post.hand, post.variant) if p)] if h] + post.lines if post else []
        notes += [f"log: {c}" for c in comments]
        seed.combo.starting = [cb.Entry(c, "", "") for c in hand]
        seed.combo.notes = "\n".join(notes)
        if post and (board := board_cards(post.end_board_text(), resolve)):
            seed.combo.end_board = board
        seed.combo.author = seed.author = seed.author or (post.author if post else "")
        seed.id = seed.combo.id = f"pinned-{path.stem}"
        seed.hand, seed.origin, seed.source = hand, path.stem, "replay"
        seeds.append(seed)
    return seeds
