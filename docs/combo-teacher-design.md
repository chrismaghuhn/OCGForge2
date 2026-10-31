# Design: Combo-Lehrer aus Spielerwissen (masterduelmeta)

Stand: 2026-10-28. Status: Entwurf, noch nicht umgesetzt; Review-Punkte (Seed-Orakel, Value außerhalb der
Verteilung, DAgger, variabler Unterbrecher, unterbrechender Sitz, Messdefinition) eingearbeitet.

## Ziel

Das Modell soll die Combos seiner Decks lernen und lernen, sie gegen gezielte Unterbrechungen zu
Ende zu bringen. Die Lehrer-Daten kommen nicht aus Selbstspiel, sondern aus von Spielern
geschriebenen Combos, Endboards und Choke Points (masterduelmeta.com), geprüft und ergänzt durch
unsere Engine. Echte Matches sind dafür nicht nötig: Alles läuft als Puzzle-Szenario im eigenen Zug.

## Warum (Befunde vom 2026-10-28)

- **Suche als Lehrer im vollen Spiel trägt noch nicht.** Orakel-Suche (kennt verdeckte Karten)
  schlägt das reine Modell 27:13, die ehrliche Suche mit ausgelosten Welten nur 60:60 über 120
  Duelle, nach Phasen 20:20. Der Vorteil kam aus dem Wissen über verdeckte Karten.
- **Im eigenen Zug ohne Gegner ist das Spiel fast vollständig informiert:** eigene Hand bekannt,
  Combos suchen Karten aus dem Deck statt zu ziehen. Dort kann Suche ein brauchbarer Lehrer sein
  (Bedingung aus DuoForge-Entscheidung 0024: Suche muss das Modell klar schlagen). „Fast“: Ziehen,
  Aufdecken und Ausgraben hängen von der Deck-Reihenfolge ab, die die Suche nicht kennen darf (K1).
- **Value-Kopf ist brauchbar:** AUC gegen den Duellausgang 0,80 (erstes Viertel) bis 0,98 (letztes
  Viertel), Korrelation 0,72 (`runs/diagnose/`). Er taugt als Bewerter für Endboards.
- **Verhalten:** Bei Kettenfragen aktiviert das Modell in 56 % der Fälle; Infinite Impermanence in
  74 % ihrer Gelegenheiten. Combos mit Bedacht und Unterbrechungen am richtigen Punkt fehlen.

## Datenquellen

Alle über die öffentliche API von masterduelmeta.com, Rohdaten nur in `.cache/masterduelmeta/`
(nie committen, nicht weitergeben; nur lokale Auswertung).

| Endpunkt | Inhalt | Umfang |
|---|---|---|
| `/api/v1/combos` | Combo: Deck-Typ, Startkarten, Schritte (je Karte + Aktionstext wie „Normal Summon“, „Synchro 10 into Baronne“, oder freier Text), Notizen, Up-/Downvotes | 1457 Combos, 210 Deck-Typen |
| `/api/v1/counters` | Pro Deck-Typ: Choke-Point-Karten, Counter-Karten mit Wirkung (High/…), Erklärungen („Veiler für Ophion aufheben“) | 588 Einträge, 116 Deck-Typen |
| `/api/v1/spreadsheets` | Kuratierte Sammlungen pro Deck: Builds und Abschnitte, Combos eingebettet mit `endBoard` (455), `vsHandTraps` (216), `requiredCards`, `garnetCards` (Bricks), `branches`/`condition`, generische Startkarten („Any Non-FIRE Monster“) | 51 Tabellen, verweisen auf 379 der Combos |

Der Deck-Filter der Webseite liefert dieselben Daten (Branded: 184 Combos im Filter und im Download).

### Zuordnung zu unseren Decks

Über Kartenüberschneidung (Karten unserer .ydk gegen die Karten in den Combos eines Deck-Typs,
ohne allgemeine Staples), nicht über Namen. Stand der ersten Auswertung: Elfnote → Elfnote (59 %,
50 Combos, 16 Counter), Branded → Branded (70 %), Light and Darkness Ritual → Chaos Ritual (55 %),
Invoked Fairy Tail → Magistus Fairy Tail (64 %), DoomZ → DoomZ (57 %), Artmage → Artmage (51 %).
Noch falsch: Synchrons (landet bei Elfnote wegen geteilter Extra-Deck-Karten) und Blitzclique (kein
Deck-Typ). Verbesserung: nur Main-Deck-Karten, seltene Karten höher gewichten; bei Unklarheit fest
eintragen (`docs`-Tabelle oder Datei in `tools/`).

## Ablauf

### K0 – Import und Namen

- `tools/fetch_combos.py`: lädt die drei Endpunkte nach `.cache/masterduelmeta/<endpunkt>-<datum>.json`
  (wie `tools/fetch_decks.py`).
- Kartennamen → Passcodes mit der Namenszuordnung aus `fetch_decks.py` (normalisiert, „loose“,
  offizielle Codes bevorzugt). Karten, die es in unserem Pool nicht gibt (Master-Duel-only) oder die
  anders funktionieren, markieren; Combos damit fallen weg.
- Ausgabe pro Deck: Combos mit Passcodes, Startkarten, Schritten, Endboard (falls vorhanden),
  vs-Handtrap-Linien, Choke Points.

### K1 – Combos in der Engine prüfen (Solo-Puzzle)

- **Szenario:** Puzzle-Welt (`_core.Duel.world`, `ocgforge.worlds`): wir beginnen Zug 1, Gegner
  passt immer (`worlds.decline`). Hand = Startkarten der Combo, aufgefüllt aus der Deckliste
  (generische Startkarten wie „Any Non-FIRE Monster“ passend gewählt), Deck = Rest gemischt. Braucht
  die Combo Karten, die nicht in unserer Liste sind (meist Extra Deck), kommen sie ins Szenario-Deck
  und werden markiert.
- **Geführte Suche:** Die Schritte sind Leitplanken. Bei jeder Entscheidung werden Antworten bevorzugt,
  die die Karte des nächsten Schritts betreffen (über `worlds.card_ref`), passend zum Verb
  (Summon/Special Summon/Synchro/Xyz/Link → Beschwörung, „Add to hand“ → Auswahl, „Send to GY“ …).
  Beam-Search über Nachspielen (`Duel` ist deterministisch); wo die Zuordnung fehlt, füllt die Suche
  die Lücke frei, bewertet mit dem Value-Kopf.
- **Keine Orakel-Lücke über den Seed:** Ein deterministisches `Duel` kennt die Deck-Reihenfolge. Jede
  Kandidatenlinie wird deshalb in mehreren Welten mit neu gemischten Decks bewertet (gleiche Hand,
  andere Seeds; gemeinsame Zufallszahlen zwischen den Linien). Ins Buch kommen nur Linien, die über
  diese Welten robust sind; an Zieh-, Aufdeck- und Ausgrab-Punkten wird verzweigt statt einer festen
  Fortsetzung.
- **Bewertung der Endboards:** Gute Combo-Endboards liegen außerhalb der Selbstspiel-Verteilung, auf
  der die Value-AUC gemessen wurde; die Suche fände sonst Boards, die der Value-Kopf überschätzt.
  Deshalb: nach dem Zug einen gegnerischen Zug ausspielen (Gegner-Policy, mehrere ausgeloste
  Gegnerhände) und den Value danach nehmen, gemittelt über die Hände. Plausibilitätscheck: Value des
  Buch-Endboards gegen das der Suche; „schlägt“ die Suche das Buch regelmäßig, ist das eher ein
  Bewerterfehler als eine bessere Combo.
- **Erfolg („Endboard erreicht“):** Schlüsselkarten aus den Tabellen (`endBoard`), sonst aus dem
  Endboard der Engine-Prüfung; gemessen als Anteil der Schlüsselkarten auf dem Feld (bzw. am Ort, den
  die Tabelle nennt). Eine Combo gilt als geprüft, wenn alle Schrittkarten in Reihenfolge benutzt und
  mindestens 80 % der Schlüsselkarten erreicht sind.
- **Abdeckung pro Verb:** Für jeden Aktionstext wird geloggt, ob er sich einer Engine-Aktion zuordnen
  ließ. So ist bei schlechter Prüfquote sichtbar, welche Verben fehlen.
- **Ausgabe:** Combo-Buch `runs/combo-book/<deck>.jsonl` (abgeleitete Daten): Szenario-Skript, geprüfte
  Zugfolge (semantische Schlüssel), Endboard, Bewertung, Quelle (Combo-ID).

### K2 – Gezielte Unterbrechungen

- **Szenario wie K1**, aber der Gegner hält Counter-Karten aus `/counters` bzw. die Handtrap aus
  `vsHandTraps`. Wann er sie einsetzt, wird gemischt, sonst lernt das Modell nur diesen einen
  berechenbaren Gegner: am Choke Point (Mehrheit), früher auf eine weniger wichtige Aktivierung
  („fällt auf den Köder herein“), später, oder gar nicht. Nur so wird Ködern belohnt, also eine
  unwichtige Aktivierung vorzuschieben, um den Counter zu ziehen.
- **Lehrerlinie:** vorhandene „vs Handtrap“-Linien aus den Tabellen prüfen wie in K1; sonst sucht die
  Engine die beste Fortsetzung nach der Unterbrechung (deterministisch, also starke Suche), bewertet
  mit dem Value-Kopf am Zugende.
- **Ausgabe:** zusätzliche Einträge im Combo-Buch mit Unterbrechung.
- **Der unterbrechende Sitz:** Dieselben Szenarien liefern Lehrer-Daten für die Gegenseite, also genau
  das gemessene Problem (Imperm in 74 % der Gelegenheiten). Für jedes Fenster, in dem der Counter
  möglich ist, wird bewertet, wie viel er dem Combo-Spieler laut Value kostet; Ziel ist das Fenster mit
  dem größten Schaden. Der Unterbrecher kennt die Hand des Combo-Spielers nicht, also wird diese wie
  in der ehrlichen Suche ausgelost (`ocgforge.worlds`) und über die Welten gemittelt.

### K3 – Training

- **Nicht nur imitieren, sondern DAgger-artig:** Combos sind lange Ketten; weicht das Modell einmal
  vom Buch ab, steht es in Zuständen, die im Buch nie vorkamen. Deshalb spielt das Modell die
  Szenarien selbst, und an jeder Entscheidung, besonders nach Abweichungen, liefert die Suche das Ziel
  (in diesen Szenarien ist sie stark). Die Buch-Linien sind der Startpunkt, nicht die einzigen Daten.
- **Imitationsverlust:** Diese Ziele (und die Buch-Linien) gehen als zusätzliche Kreuzentropie neben
  dem PPO-Verlust ins Update, mit kleinem Gewicht (Startwert 0,1–0,3), damit das Selbstspiel führend
  bleibt. Dasselbe für den unterbrechenden Sitz aus K2.
- **Szenario-Umgebungen:** Ein Teil der Envs (z. B. 10 %) spielt Szenarien statt Duelle; Belohnung
  ist der Ausgang bzw. der Value am Zugende. Optional erst nach der reinen Imitation.
- **Lehrplan:** zuerst ohne Unterbrechung (K1-Szenarien), dann mit (K2).

### K4 – Alle Decks

Nach dem Pilot mit Elfnote alle Trainingsdecks mit ausreichend Daten (siehe Zuordnung).

## Messung

- **Zurückhalten:** Ein fester Teil der Combos und Starthände (z. B. 20 % pro Deck) kommt nie ins
  Training und dient nur zum Messen; sonst misst der Combo-Abschluss Auswendiglernen.
- **Combo-Abschluss:** Auf den zurückgehaltenen Szenarien ohne Suche, nur das Modell
  (wahrscheinlichste Antworten): mittlerer Anteil der erreichten Schlüsselkarten (Definition in K1)
  und Value nach dem ausgespielten Gegnerzug, im Vergleich zum Buch. Vor und nach dem Training.
- **Gegen Unterbrechung:** Value nach der Unterbrechung (alle Zeitpunkte aus K2, auch Köder und
  keine), Modell gegen Buch bzw. Suche.
- **Unterbrecher:** Wählt das Modell auf dem unterbrechenden Sitz das Fenster mit dem größten Schaden?
  Anteil Treffer auf dem Choke Point vs. Köder.
- **Gesamtstärke:** Arena gegen frühere Stände, WindBot-Benchmark (`ocgforge.edopro.windbot`, Stand
  Checkpoint ~150: 104/201), Suche gegen reines Modell (`ocgforge.search`).
- **Verhalten:** Aktivierungsquoten bei Kettenfragen (`runs/diagnose/diagnose.py`), Ziel: weniger
  Aktivierung ohne lohnendes Ziel (Imperm 74 %), Value-AUC unverändert gut.

## Abbruchkriterien

- K1 Pilot (Elfnote): weniger als 30 % der Combos sind geprüft (Definition in K1: alle Schrittkarten
  in Reihenfolge, mindestens 80 % der Schlüsselkarten) → anhand der Verb-Abdeckung die Zuordnung der
  Aktionstexte überarbeiten, bevor K2/K3 starten.
- K1: Die Suche „schlägt“ die Buch-Endboards regelmäßig im Value → Bewertung prüfen (Gegnerzug
  ausspielen, mehr Hände), bevor ihre Linien ins Buch kommen.
- K3: Combo-Abschluss steigt nicht, oder WindBot-Benchmark bzw. Arena fällt klar → Imitationsgewicht
  senken oder Szenarien aus dem Training nehmen.

## K1-Befund: Lucina 1cc von Hand (2026-10-28)

Ein lokales Skript (`runs/diagnose/drive.py`, nicht im Repo; `runs/` ist ignoriert) spielt die Combo
mit expliziten Antworten komplett durch: 85 Antworten, Ende mit Baronne, Welcome Home,
Rhapsodia auf dem Feld, Ecclesia and the Dark Dragon bis zur End Phase verbannt, Omega hat eine
Gegnerkarte verbannt (Gegnerhand 5 → 4). Die 29 Planschritte reichen also; was fehlt, sind Regeln
für die Fragen dazwischen. Aufteilung der 85 Antworten:

| Art | Anzahl | Regel |
|---|---|---|
| Ziehen-/Standby-Fenster, eigene Ketten ohne Plan | 10 auto + 6 pass | Passen. Nie eigene Handtraps/Baronne/Regina auf die eigene Kette legen. |
| Position | 11 | Offen angriff (deterministisch). |
| Idle (Beschwören, Aktivieren, Synchro) | 11 | Nächster Planschritt (Verb + Karte). |
| Kette mit Plankarte (Quick Effects, Fallen) | 5 | Aktivieren, wenn die Karte im nächsten Planschritt „activate“ hat. |
| Karte wählen (Suche, Ziel, Beschwörung) | 13 | Die Karte des nächsten Planschritts; sonst die, die später im Plan vorkommt. |
| Synchro-Material | 8 | Planbenannte Karten; nie Karten des Endboards (Baronne). Stufensumme muss passen. |
| Zone | 15 | Siehe unten – hier scheitert die Suche bisher am meisten. |
| Ja/Nein | 6 | Ja bei Auslösern, die zur nächsten Plankarte führen; sonst beide Zweige prüfen. |

Regeln, die `combo_check.check()` dafür braucht:

1. **Deterministisch statt Beam:** Passen, Position, Ziehen/Standby: keine Verzweigung (fast 40 % der
   Antworten). Spart Beam-Breite für die echten Entscheidungen.
2. **Mittlere Zone:** Elfnote-Hauptdeck-Monster mit „center“-Effekten (Lucina, Tinia, Regina) und
   Strelitzia (3000 ATK nur in der Mitte) in `place.0.4.2`. Vorher die Mitte freihalten: Hilfsmonster
   (Power Patron, Fortuna, Fallen, Incredible Ecclesia) in Seitenzonen.
3. **Extra Monster Zone** für Extra-Deck-Monster, die nicht in die Mitte gehören (Baronne →
   `place.0.4.5`); sonst blockiert sie später eine Hauptzone.
4. **Zone frei lassen** für Karten, die zurückkehren (Ecclesia kommt in der End Phase zurück).
5. **Optionale Ja/Nein-Fragen, die keine Plankarte betreffen, mit Vorausschau entscheiden:**
   Strelitzias Stufensenkung muss „nein“ sein (Power Patron 1 + Strelitzia 7 = 8 für Ecclesia);
   die Synchro-Frage nach Power Patrons Effekt muss „nein“ sein, weil der Plan Baronne aus dem
   Idle will. Regel: beide Zweige ein paar Schritte weit spielen, den nehmen, in dem der nächste
   Planschritt als Antwort existiert.
6. **Einmal pro Zug beachten:** Power Patrons Suche als Synchro-Material greift beim zweiten Mal
   nicht; der Plan darf so einen Schritt nicht als Pflicht verlangen (Schritt überspringen, wenn
   die Engine ihn nicht anbietet).
7. **Endboard-Karten schützen:** keine Karte als Material/Kosten verwenden, die im Endboard steht.

Umgesetzt in `combo_check.check()`: feste Position und Zone (`forced`), Ketten nur mit Karten der
nächsten Schritte (sonst passen), Ja/Nein ohne Modell (beide Zweige bleiben im Beam), Endboard-Karten
nicht als Material, Bonus nach Nähe zum nächsten Schritt, Rang nach erledigten Schritten (ein
übersprungener Schritt zählt nicht; „geprüft“ erlaubt höchstens 2), doppelte Zustände raus. Mit den
festen Zonen läuft die Hand-Linie von Lucina 1cc weiter komplett.

K1 danach (5 Elfnote-Combos, 1 Mischung): 0/5 geprüft. Schritte vorher → nachher: Lucina 1cc 6 → 25/29,
Lucina 1.5 5 → 28/29, Tinia 5 → 12/27, Regina 7 → 7/20, Welcome Home 2 → 10/24 – aber mit 2–11
übersprungenen Schritten. Die Suche weicht noch an Stellen ab, an denen das Modell eine andere,
ebenso legale Karte bevorzugt (Welcome Home ruft Fortuna statt Regina). Nächster Hebel: an solchen
Stellen nur die Karte des nächsten Schritts zulassen, wenn sie angeboten wird.

## Risiken und offene Fragen

- **Master Duel ≠ TCG:** andere Banlist, einige Karten fehlen oder sind anders; die Engine-Prüfung
  filtert, kostet aber Combos.
- **Freitext:** Aktionstexte („Overlay both and detach“) sind nicht normiert; die Zuordnung zu
  Engine-Aktionen gelingt nicht immer, die Suche muss Lücken füllen.
- **Puzzle-Grenzen:** Zug 1 ohne Vorgeschichte passt zu Combos gut; Szenarien mitten im Spiel (Zug 3+)
  brauchen Felder aus echten Duellen (`worlds.sample_world`).
- **Qualität der Community-Daten:** schwankend; Up-/Downvotes als Filter, Engine-Prüfung als Pflicht.
- **Überanpassung:** Das Modell könnte nur die Buch-Linien spielen; kleines Imitationsgewicht und
  Selbstspiel als Hauptsignal halten dagegen.
- **Nutzung der Daten:** nur lokal zum Training, Rohdaten nicht im Repo, keine Weitergabe.

## Bausteine, die schon da sind

`_core.Duel` (Nachspielen, `Duel.world` für Puzzle-Szenarien, `Duel.snapshot`), `ocgforge.worlds`
(Welten, `decline`, `card_ref`, `match`), `ocgforge.search` (Best-of-K, ehrliche Suche,
Phasen), `ocgforge.edopro.windbot` (Benchmark), `runs/diagnose/diagnose.py` (Value-AUC,
Aktivierungsquoten), `tools/fetch_decks.py` (Namenszuordnung).

## K1 v2 (2026-10-30, Branch `combo-boards`)

- **Tempo:** Beam-Linien klonen das Duell ihres Elternteils (`Duel.clone`) statt die Linie jedes Mal
  neu nachzuspielen; gleiche Linien wie vorher, 24 s → 5,2 s pro Combo und Mischung (ein Thread).
  Combos und Mischungen laufen in Threads (`--workers`, `--torch-threads`). Jede Buchlinie wird am
  Ende aus dem Szenario allein nachgespielt (`replays`, bisher immer wahr).
- **Regeln:** Aktionstext nach erstem Verb und Objekt („Tribute Tinia“ bei Welcome Home = nur
  benutzt, „Synchro 10 into Baronne“ beim Material = Friedhof); Einträge eines Community-Schritts in
  beliebiger Reihenfolge; Gegner-Einträge („Opp chains Fuwalos“) zählen nicht; Bonus für die Karte
  des nächsten Schritts als „Absicht“, die bis zum Schritt hält (halber Schritt im Rang);
  Ja/Nein-Fragen mit Vorausschau auf einem Klon; Beam 24.
- **Ergebnis (3 Mischungen):** Elfnote Schritte 34 % → 51 %, geprüft 0/48 → 0/48 (3 Combos in
  einzelnen Mischungen); DoomZ 1/19, Artmage 1/18, Branded (Top 40) 1/40, Chaos Ritual 0/9,
  Magistus Fairy Tail 0/17. Geprüfte Linien enden mit kleinen Boards (1–3 Karten); für
  `board_break` taugen eher Linien mit großem Feld (`--min-field 5`).
- **Was am meisten scheitert:** „use“-Schritte ohne Ort (die Karte wird benannt, aber für etwas
  anderes als im Text gemeint), Pendel-Schritte (Vidolium, Junoldo), Schritte, die eine bestimmte
  Starthand voraussetzen („Any Elfnote“), und die Forderung „alle 3 Mischungen“, die auch die
  aufgefüllte Hand ändert.
