"""Community combos, counters and combo spreadsheets from masterduelmeta.com, for the combo teacher.

    python3 tools/fetch_combos.py

Downloads /api/v1/combos, /counters and /spreadsheets (all entries) to
.cache/masterduelmeta/<endpoint>-<date>.json. The raw data stays there: never committed, not passed
on, only read locally (python/src/ocgforge/combos.py). See docs/combo-teacher-design.md.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / ".cache" / "masterduelmeta"
API = "https://www.masterduelmeta.com/api/v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for endpoint in ("combos", "counters", "spreadsheets"):
        request = urllib.request.Request(f"{API}/{endpoint}?limit=0", headers={"User-Agent": "OCGForge combo import"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
        path = OUT / f"{endpoint}-{dt.date.today().isoformat()}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        print(f"{endpoint}: {len(data)} entries -> {path}")


if __name__ == "__main__":
    main()
