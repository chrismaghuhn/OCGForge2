# Engine cloning

`Duel.clone()` copies a duel at a pending decision: engine state, Lua VM and everything our
wrapper keeps. Given the same answers, the copy plays on exactly as the original would. Search
(best-of-K, MCTS) can then branch from a position instead of replaying the whole duel into a
new `Duel`.

```python
copy = duel.clone()          # same future as duel, given the same answers
other = duel.clone(seed=7)   # same position, engine random state reseeded (draws, coin flips, ...)
```

C++: `ygo::duel::Duel::clone(std::optional<std::uint64_t> seed)` and
`ygo::core::CoreHost::clone(seed)`.

## Where it comes from

Upstream ocgcore cannot copy a duel. We pin YGO-Gym's fork, which adds
`OCG_CloneDuel`, `OCG_CloneDuelWithSeed`, checkpoints (`OCG_CreateCheckpoint` /
`OCG_RestoreCheckpoint`) and `OCG_ReidentifyCard`. Only cloning is wired up so far.

| | repository | commit |
|---|---|---|
| ocgcore fork | https://github.com/rifa-456/ygopro-core | `230651623a5e57bc7a7096aab04c3ae6d52da205` |
| based on upstream | https://github.com/edo9300/ygopro-core | `baa3881d0bae48e9994f915f5c931be53c1ba8ef` (our previous pin) |
| Lua fork (submodule `lua/src`) | https://github.com/rifa-456/ygo-lua | `b449bd6dd97b0b7265efb20d47e7a6a26e3db520` (Lua 5.4.8) |

The Lua fork adds about 70 lines: every collectable object gets a sequential id (`clone_hash`),
tables hash object keys by that id instead of by address (so a cloned table iterates in the
same order), `tostring`/`%p` use it, and `luaL_newstatewithalloc` takes the fork's allocator.
The fork's `clone/` sources use Boost.Interprocess containers, header-only. CMake takes the
system Boost when it is exactly 1.83, otherwise the fork's `clone/dependencies.cmake` downloads
the 1.83 tarball (hash-checked), which needs network at configure time; on a machine without
network install the headers first (Ubuntu 24.04: `libboost1.83-dev`). The fork reports API version 11.1; `CoreHost` checks for that.

Both forks are AGPL-3.0-or-later like upstream; their license headers are kept.

## What changed here

- `third_party/deps.json`: the ocgcore pin points at the fork and records the Lua submodule
  commit. `tools/setup_deps.py` follows a repository change of a pin (existing `.cache/ocgcore`
  checkouts are moved over), checks the Lua commit against `deps.json`, and reapplies our
  patches after the pin moves.
- `third_party/patches/ocgcore/0001`–`0004`: regenerated against the fork. Only context moved,
  except that 0002 (starting player) now also copies `starting_player_` / `started_` in
  `duel_clone::copy_metadata` and the duel archive, so a clone keeps them.
  0004 (Lua call budget) adds fields to Lua's `global_State`; the clone copies them with the rest
  of the state and every `OCG_DuelProcess` resets the budget anyway.
- New patch 0005: the per-chain-link counter lists (`applied_chain_counters`) that ocgcore
  allocates are registered with the duel (the fork's `cloned_chain_counters` set), so a duel
  destroyed in the middle of a chain frees them. Upstream leaks them then; with clones,
  discarding a duel mid-chain is routine (found by ASan in the clone tests).
- Patch 0006 (Lua memory budget, `docs/ARCHITECTURE.md`): its counters live in Lua's
  `global_State`, inside the heap region a clone copies, so a clone keeps them (tested).
- `CMakeLists.txt`: compiles the fork's `clone/` and `identity/` sources and links the Boost
  headers (`ocg_clone_boost`).
- `CoreHost::clone`: a new host whose card data is shared (read-only) and whose script store,
  callback error and log are its own; the clone callbacks point at the new host, so a clone
  never writes into its source.
- `Duel::clone`: copies observation sessions, pending request and observation, loop-guard maps
  (`seen_states_`, `answers_`, `state_`), counters, recent decisions and config.
- Python: `_core.Duel.clone(seed=None)` (releases the GIL).

## Guarantees and limits

- Only while a decision is pending. Cloning a finished duel raises (`std::logic_error` /
  Python exception). Mid multi-card selection (a continuation) is fine: the engine is still
  waiting for its one combined answer.
- The clone is independent of its source: either may be destroyed first, clones can be cloned.
- Threads: cloning only reads its source, so one duel may be cloned from several threads at
  once (tested: 32 parallel clones, each checked against a clone made alone, also under asan).
  Nothing may act on a duel while it is being cloned, and one duel is never acted on from two
  threads at once.
- Same answers, same play: legal action keys, encoded model inputs, privileged snapshots and
  results match. The order of engine messages within one step is not promised to match
  (pointer-keyed containers), and nothing here relies on it.
- `clone(seed)` reseeds the engine's random state and Lua's `math.random`; the deck order is
  already fixed, so only future shuffles, draws after shuffles, coin flips and dice differ.
  `config()` stays the duel's origin (its seed made the deck order and opening hands); the new
  seed is appended to `reseeds()` (Python: `reseeds`) and shown in engine error messages.
- **Hidden information:** a clone, reseeded or not, holds the real hidden cards (both decks'
  order, the opponent's hand, face-down cards). It is not a sampled world. An honest search must
  still build its worlds from what the player knows (`worlds.py`) and may clone only within such a
  world (e.g. one clone per plan instead of replaying the plan's prefix); branching search
  directly from the real duel is an oracle search.
- The fork can refuse a state it cannot copy (`OCG_CLONE_UNSUPPORTED`, e.g. a Lua userdata type
  it does not know); `clone()` then throws with the engine's log. Our tests never hit this.
- Only the safe Lua libraries may be loaded (`enableUnsafeLibraries = 0`, which we use).

## Tests and numbers

- C++ `tests/duel/clone_test.cpp` (`[clone]`): lockstep of original and clone over several deck
  pairs, clone vs a replay of the same actions after the original is destroyed, clones of
  clones, reseeded clones, finished duels. The hidden `[slow]` case runs 45 deck pairings,
  cloning every 3rd decision and playing 60 decisions in lockstep.
- Python `python/tests/test_clone.py`: the same through the binding, comparing `Duel.encode`
  rows, keys and snapshots, plus a never-cloned duel to show cloning does not disturb its
  source.
- `tools/bench_clone.py`: clone vs new `Duel` + replay at many decision points, memory per
  clone.

Release build, random play, 6 duels over 6 deck pairs, every 5th decision (1587 points, median
depth 660 decisions; machine busy with other jobs, so absolute times are rough):

| | median | p90 |
|---|---|---|
| `Duel.clone()` | 5.5 ms | 8.7 ms |
| new `Duel` + replay of the actions | 136 ms | 364 ms |
| new `Duel` alone (to its first decision) | 23 ms | |

Clone is ~25x faster than replay at the median point and ~7.6x faster in the first 100
decisions; it costs a constant ~5 ms (mostly copying the Lua heap), replay grows with depth.
Each clone holds ~4.7 MB resident (200 clones of one decision, measured by RSS).

The slow lockstep case covered 1077 clone points and 63,133 decisions in lockstep without a
divergence. In one duel a card script hit the Lua call budget (patch 0004); it did so in the
original and the clone alike, with the same error.

## Updating the fork

1. Pick the new fork commit; read its changes to `ocgapi.h`, `duel.h` and `clone/`.
2. Set `ocgcore.commit` in `third_party/deps.json`. Run `python3 tools/setup_deps.py`: it stops
   if the Lua submodule moved; put the new submodule commit into `ocgcore.lua.commit` after
   checking its diff against Lua 5.4.
3. If a patch no longer applies, refresh it: check out the new commit in `.cache/ocgcore`, apply
   the series by hand (`patch -p1`), fix the rejects and regenerate each file as the diff
   between consecutive trees, e.g. `diff -ruN` of snapshots before and after each patch with
   `a/` / `b/` prefixes. New duel members that our patches add must also be copied in
   `clone/graph/duel.cpp` (`copy_metadata`) and archived in `clone/persistence/graph.cpp`.
4. Check whether the fork's `clone/sources.cmake` / `dependencies.cmake` changed (new files,
   other Boost version) and whether `OCG_VERSION_MINOR` moved (update `CoreHost`).
5. Build, run `ctest --preset debug`, `ocgforge_tests "[clone]"` (and `"[slow]"` once in release)
   and `python -m pytest python`.
