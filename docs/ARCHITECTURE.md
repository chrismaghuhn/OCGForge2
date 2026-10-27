# Architecture

```
Duel  ──process()──▶ CoreHost ──▶ ocgcore (+ Lua card scripts)
  │                     ▲
  │  raw messages       │ response bytes
  ▼                     │
protocol::decode_messages ──▶ DecisionRequest { candidates[] }
  │
  ├─▶ ObservationSession (per player, collects visible events)
  └─▶ build_player_observation ──▶ PlayerObservation (no hidden info)
```

## Decisions

The engine asks questions (`MSG_SELECT_IDLECMD`, `MSG_SELECT_CARD`, ...). The
decoder turns each into a `DecisionRequest` with **every** legal answer as an
`ActionCandidate`. Each candidate has a stable `semantic_key` (e.g.
`idle.4.0.14821890.0.2.0`) and the exact response bytes for the engine.

Questions where the engine expects several cards at once (select 2 of 5,
ordering, counters, ...) are split into one-card-at-a-time steps on our side
(`protocol/continuation`). The engine only sees the final combined answer.

`MSG_SELECT_UNSELECT_CARD` sends two lists: first the cards that can still be
**selected**, then the cards already selected that can be **unselected**.
(They were swapped once, which caused the bot to loop forever.)

## Observations

`build_player_observation` queries the engine and keeps only what the given
player is allowed to know: own hand, face-up cards, public events, and so on.
`Duel` additionally checks that every card a candidate refers to is visible to
the deciding player.

## Model input and the batched environment

`encode::encode` turns one decision (the deciding player's observation plus the
request) into the arrays described in `python/src/ocgforge/features.py`: one token
per visible card, a global feature vector, and one row per candidate pointing at
its source card's token. Card identities become model card indices through the
checkpoint's `cards.json`; hidden cards are "unknown".

`encode::VectorEnv` keeps N duels that always have a pending decision. `step`
plays one candidate per duel on a thread pool and restarts finished or failed
duels with a new seed and deck pair. `ocgforge._core` exposes it to Python with
the GIL released during `step`.

## Determinism

Same decks + same seed + same sequence of choices ⇒ same duel. The tests check
this by replaying random duels.

## Loop protection

`DuelConfig::max_decisions` (default 5000) caps a duel's length.
`DuelConfig::max_repeats` (default 16) stops a duel when the same engine
question in the same field state comes up again and again. Both throw
`DuelError` with the last 20 decisions.

## Rules

Fixed inputs live in `third_party/deps.json` (exact commits) and
`third_party/patches/ocgcore/`:

- `0001-fix-overlay-seq-parent-query.patch`: `OCG_DuelQuery` looked up the
  parent of an Xyz material with the wrong location, so materials could not be
  queried.
- `0002-add-starting-player-control.patch`: lets us choose who goes first.

Duel mode is `DUEL_MODE_MR5` (Master Rule 2020).
