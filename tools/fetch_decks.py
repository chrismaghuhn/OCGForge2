"""Candidate decks from yugiohmeta.com tournament results: the best recent list per archetype.

    python3 tools/fetch_decks.py --weeks 6

Downloads the top decks uploaded in the last --weeks (raw JSON stays in .cache/yugiohmeta/, never
committed), drops lists marked incomplete, OCG-only or Genesys, and per archetype keeps the
best-placed list, preferring lists legal under the current banlist and then the newest. Card names
are mapped to passcodes through the pinned BabelCDB (official passcodes before pre-release ones).
Each candidate becomes .cache/yugiohmeta/decks/<archetype>.ydk; decks with cards that are not in
the card pool are reported and not written. Copy the ones you want into decks/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
OUT = CACHE / "yugiohmeta"
API = "https://www.yugiohmeta.com/api/v1/top-decks"


def fetch(since: dt.date) -> list[dict]:
    query = urllib.parse.urlencode({"uploaded[$gte]": since.isoformat(), "limit": 0, "fields[_id]": 0, "fields[__v]": 0})
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "OCGForge deck import"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def normalize(name: str) -> str:
    name = name.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    name = name.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", name).strip().casefold()


def loose(name: str) -> str:
    """Letters and digits only: 'Ars Magna - "Citrinitas"' and 'Ars Magna "Citrinitas"' match."""
    return "~" + re.sub(r"[^0-9a-z]", "", normalize(name))


def card_codes() -> dict[str, int]:
    """Normalized card name -> passcode, from cards.cdb and release-*.cdb. Alternate artworks and
    later pre-release duplicates lose against the official passcode."""
    databases = [CACHE / "babelcdb" / "cards.cdb", *sorted((CACHE / "babelcdb").glob("release-*.cdb"))]
    codes: dict[str, int] = {}
    for path in databases:
        with sqlite3.connect(path) as connection:
            rows = connection.execute("select d.id, d.alias, t.name from datas d join texts t on t.id = d.id")
            for code, alias, name in rows:
                if alias and abs(code - alias) < 20:  # alternate artwork
                    continue
                official = code < 100_000_000
                for key in (normalize(name), loose(name)):
                    current = codes.get(key)
                    if current is None or (official and current >= 100_000_000):
                        codes[key] = code
    return codes


def placement(deck: dict) -> float:
    value = deck.get("tournamentPlacement")
    return float(value) if isinstance(value, (int, float)) else 999.0


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def to_ydk(deck: dict, codes: dict[str, int]) -> tuple[str, list[str]]:
    """The deck as .ydk text, and the card names that have no passcode."""
    missing: list[str] = []
    sections = []
    for section in ("main", "extra", "side"):
        lines = []
        for entry in deck.get(section) or []:
            name = entry["card"]["name"]
            code = codes.get(normalize(name), codes.get(loose(name)))
            if code is None:
                missing.append(name)
                continue
            lines += [str(code)] * int(entry.get("amount", 1))
        sections.append(lines)
    event = (deck.get("event") or {}).get("name", "?") if isinstance(deck.get("event"), dict) else "?"
    header = [f"#created by OCGForge tools/fetch_decks.py from yugiohmeta.com",
              f"#{deck['deckType']['name']}, {event}, placement {deck.get('tournamentPlacement')}, "
              f"uploaded {deck.get('uploaded', '')[:10]}"]
    text = "\n".join([*header, "#main", *sections[0], "#extra", *sections[1], "!side", *sections[2]]) + "\n"
    return text, missing


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weeks", type=int, default=6)
    parser.add_argument("--min-lists", type=int, default=1, help="skip archetypes with fewer top lists")
    args = parser.parse_args(argv)

    since = dt.date.today() - dt.timedelta(weeks=args.weeks)
    decks = fetch(since)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"top-decks-{dt.date.today().isoformat()}.json").write_text(json.dumps(decks), encoding="utf-8")

    usable = [d for d in decks if d.get("deckType") and not d.get("incomplete") and not d.get("ocg")
              and not d.get("genesys")]
    by_type: dict[str, list[dict]] = {}
    for deck in usable:
        by_type.setdefault(deck["deckType"]["name"], []).append(deck)

    codes = card_codes()
    (OUT / "decks").mkdir(exist_ok=True)
    rows = []
    for name, lists in by_type.items():
        if len(lists) < args.min_lists:
            continue
        best = min(lists, key=lambda d: (bool(d.get("illegal")), placement(d), -dt.datetime.fromisoformat(
            d["uploaded"].replace("Z", "+00:00")).timestamp()))
        text, missing = to_ydk(best, codes)
        path = OUT / "decks" / f"{slug(name)}.ydk"
        if missing:
            path.unlink(missing_ok=True)
        else:
            path.write_text(text, encoding="utf-8")
        rows.append((len(lists), name, best.get("tournamentPlacement"), bool(best.get("illegal")),
                     path.name if not missing else "-", missing))

    print(f"{len(decks)} top decks since {since}, {len(usable)} usable, {len(rows)} archetypes\n")
    print(f"{'lists':>5}  {'archetype':32} {'best':>5}  {'legal':5}  file")
    for count, name, place, illegal, file, missing in sorted(rows, key=lambda r: -r[0]):
        note = f"  missing: {', '.join(sorted(set(missing)))}" if missing else ""
        print(f"{count:>5}  {name:32} {str(place):>5}  {'no' if illegal else 'yes':5}  {file}{note}")


if __name__ == "__main__":
    main()
