# Datenquellen und Credits

Woher OCGForge2 sein Spielerwissen hat: Combos, Endboards, Replays, Decklisten, Guides. Die Datei ist
die Grundlage, um später Credits zu geben. Neue Quellen kommen hier dazu, sobald sie genutzt werden.

**Regel:** Rohdaten (gescrapte JSONs, Tabellen, Replay-Logs, Guide-Texte) bleiben lokal in `.cache/` und
werden nie committet oder hochgeladen. Ins Repo kommen nur Code, `.ydk`-Decklisten und diese Liste.

Stand: 2026-10-31

## Combo- und Meta-Daten

| Quelle | Wer | Was wir nutzen | Abgerufen | Lokal |
|---|---|---|---|---|
| [masterduelmeta.com](https://www.masterduelmeta.com/combos-and-counters/): Combos, Counters, Spreadsheets | Community-Autoren pro Combo (Branded: 55 Autoren, u. a. kanna ogata (namden), のQuỷ Đá, snLCarry, Gash; Elfnote: 16, u. a. GalacticJoey, Mediator, yuannnzzz; Sky Striker: 8, u. a. cnydo, combatant, cc12_) | 1457 Combos (Starthand, Schritte, Endboard, vsHandTraps), 588 Counter-Einträge, 51 Combo-Spreadsheets | 2026-10-28 | `.cache/masterduelmeta/` |
| [yugiohmeta.com](https://www.yugiohmeta.com/) Turnier-Decklisten | Spieler der jeweiligen Turniere | Top-Decks → `decks/*.ydk` (via `tools/fetch_decks.py`); `snake_eye.ydk` als Agent-Deck im Board-Breaking | 2026-10-28/29 | `.cache/yugiohmeta/` |
| [Road of the King](https://roadoftheking.com/tag/ocg-metagame-weekly/): OCG Metagame Reports 2026.04–2026.10 (12 Berichte) | Road of the King ([Patreon](https://www.patreon.com/roadoftheking)) | Meta-Verteilung pro Woche, ausgeschriebene Combo-Linien (Elfnote ×5, Toon, Kewl Tune, Light and Darkness Ritual, Power Patron), Tech-Hinweise | 2026-10-31 | `.cache/roadoftheking/` |

## Elfnote

| Quelle | Wer | Was wir nutzen | Abgerufen | Lokal |
|---|---|---|---|---|
| [Elfnote Guide](https://www.masterduelmeta.com/articles/guides/elfnote-galacticjoey) (masterduelmeta, 2026-05-08) | **GalacticJoey** | Guide-Text, Kartenrollen | 2026-10-31 | `.cache/guides/`, `.cache/docs/` |
| [Elfnote Spreadsheet Masterduel Edition](https://docs.google.com/spreadsheets/d/e/2PACX-1vSFvPBndRhGdzs291ir5umCYBk2VM0ljemIV2caN2ZNSfD0-HPk5wWc3K2uZULu8mIZKibh6-To_v7Y/pubhtml) | **GalacticJoey** | 29 Combos (Pure, Power Patron, Branded, KT, vs Handtraps, Fiendsmith) mit Duelingbook-Replays (Nutzer 284601); alle 29 Replays als Log | 2026-10-31 | `.cache/elfnote-joey/` |
| [Elfnote-Tabelle (Fiendsmith-Elfnote)](https://docs.google.com/spreadsheets/d/1zE00DzDJiLeNK_Zr6dJWR9pm6JSBo5vM91Rq5NmG45U) | Autor unbekannt (vom Nutzer gefunden) | 11 verifizierte Combos: Starthand → Endboard, Notizen | 2026-10-31 | `.cache/elfnote-sheet2/` |
| Elfnote-Discord, Kanal #replays | gepostet von **baguette** (Spieler „Happy Land City“) | 2 Duelingbook-Replays (1060853-82698852, 1060853-82828405) | 2026-10-31 | `.cache/elfnote-replays/` |

## Branded

| Quelle | Wer | Was wir nutzen | Abgerufen | Lokal |
|---|---|---|---|---|
| [Super Branded Combos](https://docs.google.com/spreadsheets/d/1PUdBhS5FWUqzYu367y4EDduR1q1PBf7BDmvyZZ3JkIo) (Branded-Discord) | **April** (AprilShowerz; [Twitch](https://www.twitch.tv/aprilshowerzs), YouTube, Ko-fi), laut Tabellenkopf „Created by April“; Replays von Nutzer 1027670 | ~220 Combos: Starthand, Endboard, Notizen, Replay-Links | 2026-10-31 | `.cache/branded-discord/` |
| Branded-Discord, angepinnte Nachrichten | **Katsune** (KatsuneNiko), **Colossus** (ScarTheColossus) | 5 Duelingbook-Replays mit Beschreibung (817316-56847590, -56847789, -56847305; 215034-54886036, -54885505) | 2026-10-31 | `.cache/branded-discord/pinned/` |

## Sky Striker

| Quelle | Wer | Was wir nutzen | Abgerufen | Lokal |
|---|---|---|---|---|
| [Striker Guide](https://docs.google.com/document/d/1if-E-ceCn1_8kQkEnNlAIBL8sOECVkWz7aCCMFyE3wo) (Sky-Striker-Discord) | **Gracie** | Kartenanzahlen, Schwachstellen gegen Handtraps | 2026-10-31 | `.cache/sky-striker-guide/` |
| [Sky Striker: Master Duel Guide](https://docs.google.com/document/d/185LPYSYHMml057jxYk91_UbbdaEn9Ius53AnN1Wyvwc) (2022) | **Kevin Kawaii** | Abschnitt „Playing Against Striker“ (Handtrap-Ziele) | 2026-10-31 | `.cache/docs/` |
| Reddit: „Basic Sky Striker combo for beginners“ (4 Bilder) | unbekannt | angesehen, nicht genutzt (veraltet, Zug 2+) | 2026-10-31 | – |

## Plattformen

- [Duelingbook](https://www.duelingbook.com/): alle Replays. Gelesen nur einzeln über den Browser des Nutzers,
  ohne Umgehung der Cloudflare-Prüfung, mit Pausen zwischen den Abrufen.

## Software und Kartendaten

| Projekt | Lizenz | Wofür |
|---|---|---|
| [edo9300/ygopro-core](https://github.com/edo9300/ygopro-core) und [YGO-Gym-Fork von rifa-456](https://github.com/rifa-456/ygopro-core) | AGPL-3.0-or-later | Spiel-Engine (ocgcore), Fork mit Duel-Klonen |
| [rifa-456/ygo-lua](https://github.com/rifa-456/ygo-lua) | – | Lua-Fork der Engine |
| [ProjectIgnis CardScripts](https://github.com/ProjectIgnis/CardScripts) | AGPL-3.0-or-later | Kartenskripte |
| ProjectIgnis BabelCDB | – | Kartendatenbank (`cards.cdb`, `release-*.cdb`) |
| [ProjectIgnis/Multirole](https://github.com/ProjectIgnis/Multirole) | AGPL-3.0 | EDOPro-Server-Anbindung |
| [wjakob/nanobind](https://github.com/wjakob/nanobind) | BSD-3-Clause | Python-Bindung |

## Offen

- Pro Combo die Autoren automatisch aus den Rohdaten ziehen (masterduelmeta liefert `author.username`) und
  hier als vollständige Liste erzeugen.
- Lizenzen von ygo-lua und BabelCDB nachtragen.
- Vor einer Veröffentlichung bei GalacticJoey, den Discords (Branded, Elfnote, Sky Striker) und Road of
  the King nachfragen, ob und wie sie genannt werden möchten.
