# Plan: EDOPro-Brücke in OCGForge2

Ziel: Ein OCGForge2-Checkpoint tritt in EDOPro als normaler Spieler in einem lokalen/privaten Raum
an (Mensch gegen Bot, später Bot gegen WindBot). Ersetzt OCGForge-Ignis; dessen Lobby-/Protokoll-
Wissen (I1/I2) wird als Vorlage genutzt, der Rest nicht.

## Warum nicht Ignis weiterbauen

- Ignis dekodiert eine feste Liste von ~45 Nachrichtentypen und bricht bei jedem anderen Typ ab
  (`GameplayMessageDecoderV1.cs`, `UnknownMessageId`); Auswahlfragen nur Idle und Kette.
- Jeder neue Typ braucht eine eigene Vertrags-/Beweisrunde (byte-exakter Cross-Oracle zum alten
  OCGForge), daher 72k Zeilen C# ohne vollständiges Duell.
- Es zielt auf das Modellformat des archivierten OCGForge, nicht auf OCGForge2.

OCGForge2 hat bereits: einen Decoder für 76 Nachrichtentypen inkl. aller Auswahlfragen
(`src/protocol/message_decoder.cpp`, kennt auch `MSG_UPDATE_DATA`, `MSG_UPDATE_CARD`,
`MSG_RELOAD_FIELD`), die Spieler-Sicht aus dem Nachrichtenstrom (`ObservationSession`), den
Encoder fürs Modell und zu jeder Antwortmöglichkeit die fertigen Antwort-Bytes
(`ActionCandidate::exact_response_bytes`, Mehrschritt-Auswahlen über `apply_continuation_action`).

## Grundregel gegen das Ignis-Problem

Unbekannte oder nicht verstandene **nicht-interaktive** Nachrichten werden geloggt und übersprungen,
nie fatal. Nur eine Auswahlfrage, die wir nicht beantworten können, ist ein Problem; dann eine
sichere Ersatzantwort (Abbrechen/erste legale Option, falls bekannt) und Log, oder Aufgeben mit
klarer Meldung. Regeln und Legalität entscheidet weiter EDOPros Engine (`MSG_RETRY` = wir lagen
falsch → loggen, Ersatzantwort).

## Architektur

```
EDOPro-Raum (Host: Spieler im EDOPro-Client)
   │ TCP, CTOS/STOC
   ▼
python/src/ocgforge/edopro/client.py      Lobby, Deck, Stein-Schere-Papier, STOC_GAME_MSG ↔ CTOS_RESPONSE
   │ rohe GAME_MSG-Bytes
   ▼
C++ ClientDuel (src/client/, nanobind)    wie duel::Duel, aber ohne Engine:
   ├─ ObservationSession.ingest            sichtbare Ereignisse (wie bisher)
   ├─ ClientView.apply                     Kartenwerte aus MSG_UPDATE_DATA/UPDATE_CARD/RELOAD_FIELD,
   │                                       LP/Kette/Phase aus den Ereignissen
   ├─ decode_messages → DecisionRequest    Auswahlfrage + Kandidaten
   └─ build_player_observation(ClientView) gleiche Beobachtung wie im Training
   ▼
encode → Model (ModelPolicy, greedy) → semantic_key → act() → Antwort-Bytes → CTOS_RESPONSE
```

### C++: Engine-Abfragen abstrahieren

`build_player_observation` fragt heute direkt `CoreHost` ab: `query_field()`, `query_location()`
(Hand, Monster-/Zauberzonen, Friedhof, Verbannt, Extra Deck je Spieler), `query()` für
Xyz-Material und `static_card_data()`.

1. Interface `QuerySource` mit diesen vier Methoden; `CoreHost` implementiert es, die Signatur von
   `build_player_observation` nimmt `const QuerySource&`. Reiner Umbau, Training unverändert
   (bestehender Test „model-input observations encode exactly like full ones“ sichert ab).
2. `ClientView : QuerySource`:
   - `query_location`: letzte Puffer aus `MSG_UPDATE_DATA` (Spieler, Ort, Query-Puffer) je
     (Spieler, Ort), per `MSG_UPDATE_CARD` einzeln aktualisiert; Abfrage-Puffer sind
     selbstbeschreibend (API 11), unser `decode_location_query` liest sie.
   - `query_field`: kein Server-Paket dafür; aus `MSG_START`/`MSG_RELOAD_FIELD` plus LP-Nachrichten
     (Damage, Recover, LpUpdate, PayLpCost) und Ketten-Nachrichten nachführen. Dafür eine Variante
     des Builders, die die dekodierte Feldstruktur statt Bytes nimmt.
   - `static_card_data`: aus unserer Kartendatenbank (wie bisher).
3. `ClientDuel`: `feed(game_msg_bytes)` → falls Auswahlfrage: `request()` + `observation()`;
   `act(semantic_key)` → Antwort-Bytes (Mehrschritt-Auswahlen intern wie in `Duel::act`).
   Kein Schleifenschutz nötig, das macht der Server.

### Python: Netzwerk-Client

`ocgforge/edopro/protocol.py` (Paketformat: u16-Länge LE + u8-Typ + Nutzdaten) und `client.py`
(asyncio): `CTOS_PLAYER_INFO`, `CTOS_JOIN_GAME` (PRO_VERSION, Raum-Passwort), `CTOS_UPDATE_DECK`,
`CTOS_HS_READY`, Stein-Schere-Papier, Wahl des Startspielers, dann `STOC_GAME_MSG` → `ClientDuel`
und `CTOS_RESPONSE`; `STOC_DUEL_END`/Chat/Fehlerpakete. Layouts aus Ignis I1/I2 übernehmen (eigener,
dort schon gegen einen echten Server geprüfter Code). Geschätzt 300–500 Zeilen.

Start: `python -m ocgforge.play --host 127.0.0.1 --port 7911 --password … --deck decks/x.ydk
--checkpoint runs/focus-large-s0/latest`.

## Meilensteine

**M0 – Spike, Risiken klären (½–1 Tag)**
- Minimaler Client, der nur beitritt und alle `STOC_GAME_MSG` eines echten Duells in eine Datei
  schreibt (Gegner: Mensch oder WindBot, Bot-Seite antwortet zunächst per Konsole/Zufall).
- Diese Aufzeichnung offline durch `decode_messages` und `ObservationSession` jagen:
  Liste unbekannter Typen und Abweichungen.
- Prüfen, mit welchen Query-Flags der Server `MSG_UPDATE_DATA` schickt und ob alle Felder unserer
  Modelleingabe (`kModelInputQueryFlags`: Code, Position, Typ, Level/Rang/Link, ATK/DEF,
  Xyz-Material, Zählmarken, Besitzer, Status, öffentlich) enthalten sind. Fehlende Felder → aus
  Ereignissen nachführen oder im Modell als unbekannt kodieren.
- Prüfen, wie verdeckte Gegnerkarten im gefilterten Strom aussehen (Code 0) und dass unsere
  Sitzung damit klarkommt.

**M1 – ClientDuel offline (2–3 Tage)**
- `QuerySource`-Umbau, `ClientView`, `ClientDuel`, Bindings.
- Tests: Wiedergabe der M0-Aufzeichnungen; jede Auswahlfrage liefert eine Beobachtung, die sich
  kodieren lässt, und Antwort-Bytes. Gleichheitstest gegen die Engine: eigene Duelle mit `Duel`
  spielen, daraus den gefilterten Strom samt Updates erzeugen, Beobachtungen beider Wege vergleichen
  (statistisch über viele Duelle, nicht byte-exakt für jeden Sonderfall).

**M2 – Spielen über das Netz (1–2 Tage)**
- Netzwerk-Client und `ocgforge.play`; erstes komplettes Duell Mensch gegen Bot.
- Danach Bot gegen WindBot in Serie, Fehlerzähler: Ersatzantworten, `MSG_RETRY`, Abbrüche.

**M3 – Robustheit (nach Bedarf)**
- Zeitlimits, Neuverbindung, Match/Siding, `MSG_ANNOUNCE_CARD` (fehlt auch im Training; blockiert
  8 Decks), Protokollierung für Analysen (welche Karten, welche Fehler).

## Versionen

- EDOPro-Pin aus Ignis: Commit 30935e8, ocgcore-Gitlink 46779fb, PRO_VERSION 0x1354 (41.0.2).
  Der aktuelle EDOPro-Master zeigt auf denselben ocgcore-Gitlink.
- Unser ocgcore-Pin baa3881 liegt 14 Commits danach (Build-Dateien, Lua-Enums, ein Simple-AI-Fix,
  `EFFECT_OPPO_CHOOSES_SPSUMMON_ZONE`, Summon-Prüfungen); kein Hinweis auf geänderte
  Nachrichtenformate. Im Spiel rechnet ohnehin EDOPros eigene Engine, wir lesen nur ihre Nachrichten.
- PRO_VERSION muss zur installierten EDOPro-Version passen; im Spike prüfen.

## Ergebnis M0 (2026-10-28)

Gebaut: `ClientDuel` (C++, `src/client/`, Binding `ocgforge._core.ClientDuel`) und der Spike-Client
`python -m ocgforge.edopro.client` (Lobby, Deck, Stein-Schere-Papier, Antworten, `--record`).

- Offline: `ClientDuel` spielt 20 Duelle aus dem rohen Nachrichtenstrom unserer Engine mit
  Zufallsantworten zu Ende; die Engine nimmt jede Antwort an (`tests/client/client_duel_test.cpp`).
- Echtes Duell gegen Chris in EDOPro 41.0.2 (Linux, LAN-Raum ohne Deckprüfung), Aufzeichnung
  `runs/edopro/spike-3.bin`: Lobby und Protokoll funktionieren auf Anhieb. 7792 Spielnachrichten in
  49 Typen, 512 Fragen beantwortet, **0 abgelehnte Antworten (`MSG_RETRY`), 0 unlesbare Nachrichten**,
  auch `ObservationSession` liest den gefilterten Strom fehlerfrei. Duell regulär zu Ende gespielt.
- Mit Deckprüfung lehnt der Raum Decks mit neuen oder limitierten Karten ab (Elfnote Tinia,
  Called by the Grave): fürs Testen „Don't check deck“ setzen.
- Aus EDOPros Server-Code (`generic_duel.h`): Kartendaten-Updates nutzen feste Query-Flags (Monster
  0x3981fff, Zauber/Fallen 0x3f81fff, Hand 0x3781fff, Friedhof/Extra 0x381fff). Darin fehlen
  Xyz-Material, Zählmarken und Besitzer, die unsere Modelleingabe nutzt; `ClientView` muss sie aus
  den Ereignissen nachführen (Moves mit LOCATION_OVERLAY, AddCounter/RemoveCounter, Herkunft).

Damit ist das größte Risiko weg: Unser Decoder versteht EDOPro vollständig. Nächster Schritt ist M1
(`QuerySource`/`ClientView`, damit das Modell statt des Zufalls entscheidet).

## Ergebnis M1 (2026-10-28)

- Statt eines `QuerySource` auf Byte-Ebene liest der Beobachtungsaufbau jetzt über
  `detail::DecodedQueries` (dekodierte Abfragen); die Engine antwortet über einen Adapter um
  `CoreHost`, der Client über `ClientView` (`src/client/client_view.cpp`).
- `ClientView` führt das Feld aus den Servernachrichten: Updates, Moves, Draws, Swaps,
  Positionswechsel, Zählmarken, LP, Kette. Ergänzt Besitzer, Xyz-Material und Zählmarken aus den
  Ereignissen, Link-Wert/-Marker im Friedhof/Extra Deck aus der Kartendatenbank, verfolgt die Karten
  offener Kettenglieder und vergisst „negiert“ außerhalb des Feldes.
- `tests/client/client_view_test.cpp`: ein nachgebauter EDOPro-Server (Updates zum selben Zeitpunkt
  wie `generic_duel.cpp`, für den Gegner nur die öffentlichen Daten wie `Query::GenerateBuffer`).
  Bei 2379 Fragen in 12 Duellen ist die Modelleingabe exakt gleich der des Trainings.
- Die echten Aufzeichnungen (642 Fragen) baut `ClientDuel` fehlerfrei zu Modelleingaben.
- `ocgforge.edopro.client --checkpoint …` lässt das Modell spielen (Standard: wahrscheinlichste
  Antwort); die Regeln kommen aus dem Raum, Abweichung von Master Rule 2020 wird gemeldet.
