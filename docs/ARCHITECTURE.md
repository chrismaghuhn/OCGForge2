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

Memory: the 64 most recent remembered public events from the deciding player's
own event stream become event tokens (kind, card when it was public, side,
zones, turns ago, life point change). Phase changes, shuffles, chain
bookkeeping and the player's own draws are left out. An event links to the
field card token it happened to only while that is provably the same card: no
card has moved into or out of that place since (hand, graveyard and banished
positions renumber and are never linked). Tests check the events against the
stream and every link against the true card.

`encode::VectorEnv` keeps N duels that always have a pending decision. `step`
plays one candidate per duel on a thread pool and restarts finished or failed
duels with a new seed and deck pair. `ocgforge._core` exposes it to Python with
the GIL released during `step`.

## Determinism

Same decks + same seed + same sequence of choices ⇒ same duel. The tests check
this by replaying random duels.

## Loop protection

`DuelConfig::max_decisions` (default 5000) caps a duel's length and throws
`DuelError` with the last 20 decisions.

A policy can loop without progress, typically by picking a card, cancelling
the follow-up selection and picking the same card again. The duel hashes each
engine question together with the field state; an answer given
`DuelConfig::block_after` times (default 8) to the same question in the same
state is removed from its candidates there. When no candidate is left, or the
question comes back more than `max_repeats` times (default 64), the asking
player loses (`DuelResult::reason == kLoopLoss`), so the duel still has a
result to learn from. Questions with a single candidate are not counted (the
player asked has no choice). Multi-step selections built on our side
(continuations) name the engine step in their keys, so their answers are not
blocked; they only count towards `max_repeats`. `VectorEnv` sums
`blocked_answers` and `loop_losses`; `train.py` logs them per iteration. With
`block_after = 0` visit `max_repeats + 1` throws a "stuck" `DuelError`
instead (the old behaviour, with `max_repeats = 16`). With the overnight
model this turned about 40% stuck duels into finished ones (16% of them lost by
looping).

Some card scripts search combinations of materials (Synchro, Ritual, ...)
and, in unusual board states, never finish. Patch 0004 gives each
`OCG_DuelProcess` call a budget of 15M Lua function calls (about 3 seconds;
the slowest legitimate calls seen take about 2); a call that exceeds it makes
`CoreHost::process` throw, so that duel fails instead of blocking. In random
self-play across all deck pairings this cuts off roughly 3% of duels. The
budget is counted in Lua's call path (`lua/src/ldo.c`) rather than with a
count hook, which would make the VM check every instruction (~10% slower).

Patch 0007 adds a Lua memory budget the same way: `CoreHostConfig::lua_memory_limit`
(default 64 MiB, `DuelConfig::lua_memory_limit`, 0 = none) is set with
`OCG_DuelSetLuaMemoryLimit` right after `OCG_CreateDuel`, never while the Lua state is
being built. It wraps the fork's region allocator in Lua's allocation path (`lua/src/lmem.c`):
a request that would take the state past the limit fails like an out-of-memory, Lua runs an
emergency collection and then raises a memory error, and `CoreHost` throws
"Lua memory budget exceeded" (after creation, loading cards and every `process`). Only
allocations that can raise an error are refused: one made outside every protected call (ocgcore
creating a group from C++) would abort the process instead, so it may go past the limit. The
counter is Lua's own byte count, part of the Lua state, so a clone copies it.
`OCG_DuelLuaMemory` / `OCG_DuelLuaMemoryPeak` / `OCG_DuelLuaHeapExtent` (Python:
`Duel.lua_memory`) report it. Random duels over every deck pairing (decks/ and holdout, 118
duels): 1.7 MB after creation, peak median 3.6 MB, max 5.2 MB; per deck the highest peak is
3.9-5.2 MB, so the default leaves >10x headroom below the fork's 256 MiB heap reserve.

Lua runs its generational collector (`CoreHostConfig::lua_generational_gc`, default on, patch
0008's `OCG_DuelSetLuaGenerationalGC` right after creation; `OCGFORGE_LUA_GC=inc|gen` forces one
mode for the whole process, over the config, for A/B runs; any other value makes duel creation
fail). Card scripts make mostly short-lived garbage, and the fork's heap region never
shrinks, so a smaller heap stays smaller: with random play the region grows ~0.9 MB per 1000
decisions instead of ~2.2 MB (max 5.4 vs 8.9 MB over 6 duels), VectorEnv runs 6-20% more
decisions/s on one thread and 256 envs used ~7% less RSS after 60 s. Play is the same in both
modes (tested), and clones copy the generational lists (clone tests, incl. `[slow]`, pass in
both modes).

Card data (`CardDataStore`) and compiled card scripts (`ScriptCache`) are read once per process
and kept until it exits; a process only sees an updated `.cache` after a restart.

`VectorEnv::info(env)` shows the seed, decks and actions of an env's current
duel (also while `step` runs), which is enough to replay a stuck duel with
`Duel` alone. Duels that fail to start are counted in `start_failures`.

## Rules

Fixed inputs live in `third_party/deps.json` (exact commits) and
`third_party/patches/ocgcore/`:

- `0001-fix-overlay-seq-parent-query.patch`: `OCG_DuelQuery` looked up the
  parent of an Xyz material with the wrong location, so materials could not be
  queried.
- `0002-add-starting-player-control.patch`: lets us choose who goes first.
- `0003-memoize-sum-greater-check.patch`: the "at least N levels" material
  check (Ritual Summons) tried every subset when there was no solution; it now
  remembers partial results (same answers, polynomial time).
- `0004-lua-call-budget.patch`: the per-call Lua call budget above, with
  `OCG_DuelLuaBudgetExceeded` and `OCG_DuelLuaCalls`; it also patches the
  bundled Lua (`lua/src`), so `setup_deps.py` resets that submodule too.
- `0005-own-applied-chain-counters.patch`: frees per-chain-link counter lists
  of a duel destroyed mid-chain (see `docs/engine-clone.md`).
- `0006-get-card-effect-by-code.patch`: `Card.GetCardEffect(code)` walked every
  effect of the card, its equips, Xyz materials and the field's auras and compared
  codes; the containers are keyed by code, so it now starts at that key (same
  effects, same order). Scripts call it tens of millions of times in a run of random
  duels. It relies on no script changing an effect's code after registering it,
  as `filter_effect` already does.
- `0007-lua-memory-budget.patch`: the Lua memory budget above, with
  `OCG_DuelSetLuaMemoryLimit`, `OCG_DuelLuaMemory`, `OCG_DuelLuaMemoryPeak`,
  `OCG_DuelLuaMemoryExceeded` and `OCG_DuelLuaHeapExtent`.
- `0008-lua-gc-control.patch`: `OCG_DuelSetLuaGenerationalGC` (above) and
  `OCG_DuelCollectLuaGarbage` (a full collection; see "Clone cost" in `docs/engine-clone.md`).
- `0009-reset-phase-cascading-removal.patch`: the end-of-phase reset walked the
  phase-reset effects with a saved iterator, but removing a grant also removes the
  effects it granted. When those reset at the same phase (`Pendulum.GrantAdditionalPendulumSummon`:
  Solfachord Happiness, D/D/D Sky King Zeus Ragnarok, ...) the saved node could be freed,
  and the engine crashed in `field::reset_phase`. It now walks a copy, skips effects already
  removed, and resets them in creation order (as `reset_chain` does), not in hash order.

Card script patches live in `third_party/patches/cardscripts/`:

- `0001-synchro-search-memo.patch`: `Synchro.Condition` (is a Synchro Summon possible?)
  searches materials by picking them one at a time and so reaches every material set
  once per order of its cards; with "1+ non-Tuners" or several Tuners that is factorial
  in the number of candidates. While the condition searches, the part of the search
  after a pick is now remembered per state (picked Tuners, picked non-Tuners, candidates
  left). The checks of each pick itself (material restrictions, hand Synchro) still run
  for every order, so the answers stay the same. Synchro checks run for every Extra
  Deck card at every Main Phase decision and were about a third of engine time.

`tools/setup_deps.py` applies the patches in order to a clean checkout and
records their hashes in `.ocgforge-patches` in that checkout.

Duel mode is `DUEL_MODE_MR5` (Master Rule 2020).
