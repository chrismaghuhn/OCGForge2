#!/usr/bin/env python3
"""Fetch the pinned rules engine, card scripts and card database, patch ocgcore,
and extract the card data for our decks.

Idempotent: re-running only does work that is missing. Called by CMake at
configure time; can also be run by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPE_PENDULUM = 0x01000000
TYPE_LINK = 0x04000000


def git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def checkout(name: str, spec: dict, cache: Path) -> Path:
    path = cache / name
    fresh = not path.exists()
    if fresh:
        print(f"fetching {name} @ {spec['commit'][:12]}", file=sys.stderr)
        git("clone", "--filter=blob:none", "--no-checkout", spec["repository"], str(path))
    head = git("rev-parse", "HEAD", cwd=path, check=False).stdout.strip()
    # A fresh --no-checkout clone has no files yet, even when HEAD already is the pinned commit.
    if fresh or head != spec["commit"]:
        git("fetch", "origin", spec["commit"], cwd=path, check=False)
        git("checkout", "--force", "--detach", spec["commit"], cwd=path)
    if spec.get("submodules"):
        git("submodule", "update", "--init", "--recursive", cwd=path)
    return path


def apply_patches(path: Path, patches: list[str]) -> None:
    """Apply our patches in order on a clean checkout. A marker with the patches' hashes makes this
    a no-op when they are already in place (patches touching the same lines can't be checked one
    by one)."""
    marker = path / ".ocgforge-patches"
    files = [ROOT / "third_party" / "patches" / "ocgcore" / name for name in patches]
    expected = "\n".join(f"{f.name} {hashlib.sha256(f.read_bytes()).hexdigest()}" for f in files) + "\n"
    if marker.exists() and marker.read_text(encoding="utf-8") == expected:
        return
    git("checkout", "--force", "--", ".", cwd=path)
    for patch in files:
        git("apply", str(patch), cwd=path)
    marker.write_text(expected, encoding="utf-8")


def read_deck_codes(paths: list[Path]) -> list[int]:
    """Main and extra deck passcodes; the side deck is not played and is skipped."""
    codes: set[int] = set()
    for path in paths:
        side = False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line in ("#main", "#extra"):
                side = False
            elif line == "!side":
                side = True
            elif line and not side and not line.startswith(("#", "!")):
                codes.add(int(line))
    return sorted(codes)


def databases(babelcdb: Path, patterns: list[str]) -> list[Path]:
    """Card databases in priority order: for a passcode in several, the first one wins."""
    paths = [path for pattern in patterns for path in sorted(babelcdb.glob(pattern))]
    if not paths:
        raise SystemExit(f"no card database matches {patterns} in {babelcdb}")
    return list(dict.fromkeys(paths))


def write_card_data(databases_: list[Path], decks: list[Path], output: Path) -> None:
    """Every card of the databases (effects create cards that are in no deck list), after checking
    that every deck card exists."""
    query = "select id, alias, setcode, type, level, attribute, race, atk, def from datas"
    found: dict[int, tuple] = {}
    for database in databases_:
        with sqlite3.connect(database) as connection:
            for row in connection.execute(query):
                found.setdefault(row[0], row)
    rows = [found[code] for code in sorted(found)]
    missing = sorted(set(read_deck_codes(decks)) - set(found))
    if missing:
        where = {code: [p.stem for p in decks if code in read_deck_codes([p])] for code in missing}
        raise SystemExit("cards missing from every card database (deck: passcodes): "
                         + "; ".join(f"{code} in {', '.join(where[code])}" for code in missing))

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
    decks = sorted((ROOT / "decks").glob("**/*.ydk"))
    write_card_data(databases(babelcdb, deps["babelcdb"]["databases"]), decks, cache / "card_data.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
