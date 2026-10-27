#!/usr/bin/env python3
"""Fetch the pinned rules engine, card scripts and card database, patch ocgcore,
and extract the card data for our decks.

Idempotent: re-running only does work that is missing. Called by CMake at
configure time; can also be run by hand.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPE_PENDULUM = 0x01000000
TYPE_LINK = 0x04000000
# Cards created by effects that appear in no deck list.
TOKEN_CODES = [20001444]  # Swordsoul Token


def git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def checkout(name: str, spec: dict, cache: Path) -> Path:
    path = cache / name
    if not path.exists():
        print(f"fetching {name} @ {spec['commit'][:12]}", file=sys.stderr)
        git("clone", "--filter=blob:none", "--no-checkout", spec["repository"], str(path))
    head = git("rev-parse", "HEAD", cwd=path, check=False).stdout.strip()
    if head != spec["commit"]:
        git("fetch", "origin", spec["commit"], cwd=path, check=False)
        git("checkout", "--force", "--detach", spec["commit"], cwd=path)
    if spec.get("submodules"):
        git("submodule", "update", "--init", "--recursive", cwd=path)
    return path


def apply_patches(path: Path, patches: list[str]) -> None:
    for name in patches:
        patch = ROOT / "third_party" / "patches" / "ocgcore" / name
        already = git("apply", "--reverse", "--check", str(patch), cwd=path, check=False).returncode == 0
        if not already:
            git("apply", str(patch), cwd=path)


def read_deck_codes(paths: list[Path]) -> list[int]:
    codes: set[int] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "!")):
                codes.add(int(line))
    return sorted(codes)


def write_card_data(database: Path, decks: list[Path], output: Path) -> None:
    codes = sorted(set(read_deck_codes(decks)) | set(TOKEN_CODES))
    query = ("select id, alias, setcode, type, level, attribute, race, atk, def from datas "
             f"where id in ({','.join('?' for _ in codes)}) order by id")
    with sqlite3.connect(database) as connection:
        rows = connection.execute(query, codes).fetchall()
    missing = sorted(set(codes) - {row[0] for row in rows})
    if missing:
        raise SystemExit(f"cards missing from database: {missing}")

    lines = ["# code|alias|setcode|type|level|attribute|race|atk|def|lscale|rscale|link_marker"]
    for code, alias, setcode, card_type, level, attribute, race, atk, defense in rows:
        pendulum = card_type & TYPE_PENDULUM
        link = card_type & TYPE_LINK
        lines.append("|".join(str(v) for v in (
            code, alias, setcode, card_type, level & 0xFF, attribute, race, atk,
            0 if link else defense,
            (level >> 16) & 0xFF if pendulum else 0,
            (level >> 24) & 0xFF if pendulum else 0,
            defense if link else 0,
        )))
    content = "\n".join(lines) + "\n"
    if not output.exists() or output.read_text(encoding="utf-8") != content:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache")
    args = parser.parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)

    deps = json.loads((ROOT / "third_party" / "deps.json").read_text(encoding="utf-8"))
    try:
        ocgcore = checkout("ocgcore", deps["ocgcore"], cache)
        apply_patches(ocgcore, deps["ocgcore"]["patches"])
        checkout("cardscripts", deps["cardscripts"], cache)
        babelcdb = checkout("babelcdb", deps["babelcdb"], cache)
    except subprocess.CalledProcessError as error:
        print(f"git {' '.join(error.cmd[1:])} failed:\n{error.stderr}", file=sys.stderr)
        return 1
    write_card_data(babelcdb / "cards.cdb", sorted((ROOT / "decks").glob("*.ydk")), cache / "card_data.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
