# OCGForge2

A deterministic Yu-Gi-Oh! duel environment for game-AI experiments, built as a
thin C++ layer around [ygopro-core](https://github.com/edo9300/ygopro-core).

Current scope: a pool of current meta decks (`decks/`, with two hold-out decks
in `decks/holdout/` that are never used for training), full duels, every legal
choice exposed as a candidate, and a per-player view that never leaks hidden
information.

Decks are `.ydk` files (`#main`, `#extra`; a `!side` section is ignored). The
card pool is the pinned BabelCDB (`cards.cdb` plus `release-*.cdb`); a card
missing from it makes `tools/setup_deps.py` fail and name the deck.

## Build and test (Linux)

Requirements: CMake ≥ 3.24, Ninja, a C++20 compiler, Python 3, git.

```bash
cmake --preset debug          # also fetches and patches ocgcore, CardScripts, BabelCDB into .cache/
cmake --build --preset debug
ctest --preset debug
```

Presets: `debug`, `release`, `asan` (AddressSanitizer + UBSan).

`OCGFORGE_RANDOM_DUELS=<n>` sets how many random duels the smoke test plays
(default 25; CI uses 5 for the slow sanitizer build).

The slow soak tests (500 random duels, and one random duel for every pairing of
the decks in `decks/`) are hidden by default:

```bash
./build/release/tests/ocgforge_tests "[slow]"
```

## Playing a duel

```cpp
#include "ygo/duel/duel.hpp"

ygo::duel::DuelConfig config;
config.card_scripts_root = ".cache/cardscripts";
config.card_data_tsv = ".cache/card_data.tsv";
config.deck0 = ygo::core::load_deck("decks/swordsoul_tenyi.ydk");
config.deck1 = ygo::core::load_deck("decks/salamangreat.ydk");
config.seed = 42;

ygo::duel::Duel duel(config);
while (!duel.finished()) {
    const auto& request = duel.request();      // whose turn, what kind of choice
    const auto* view = duel.observation();     // what that player can see
    duel.act(request.candidates.front().semantic_key);
}
auto result = *duel.result();                  // winner, reason
```

A duel that needs more than `max_decisions` choices, or that keeps asking the
same question in the same state, throws `DuelError` with the last decisions in
the message instead of hanging.

## Python: model, training and the batched environment

`python/` is the `ocgforge` package: the policy/value model (a set transformer that scores every
legal candidate, ported from Argentum P1), the PPO core, an Elo ladder, card-text embeddings, and
`ocgforge._core`, the C++ `VectorEnv` (many duels on a thread pool, encoded for the model).

```bash
uv venv .venv && uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e "python[test]"
cmake --preset python && cmake --build --preset python   # builds python/src/ocgforge/_core*.so
.venv/bin/python -m pytest python
```

Self-play and one PPO update:

```python
from ocgforge.model import Model
from ocgforge.ppo import PPOConfig, build_steps, ppo_update
from ocgforge.rollout import ModelPolicy, collect, vector_env

env = vector_env(model.card_features, ["decks/swordsoul_tenyi.ydk", "decks/salamangreat.ydk"], num_envs=256)
trajectories, stats = collect(env, ModelPolicy(model, device), duels=512)
ppo_update(model, optimizer, build_steps(trajectories, gamma=1.0, lam=0.95), PPOConfig(), device)
```

`rollout.AsyncCollector(env, batch_size, segment=128).collect(policy, steps)` drives the same env
with `recv_batch`/`send` instead, so the policy only waits for the first `batch_size` ready duels
(faster with a GPU model). It hands out segments of at most `segment` decisions per seat, cut-off
ones with a value bootstrap; duels keep running across calls (only each seat's latest decision
carries over), so memory does not depend on how long duels get.

Install jemalloc (`sudo apt install libjemalloc2`): with glibc malloc the engine threads fragment
memory and run ~25% slower. `python -m ocgforge.train` preloads it by itself; elsewhere use
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`.

Memory: each env holds a duel engine (~3.5–5.5 MB, all engines share the card data) plus up to two
open segments of ~6 KB per decision, so ~7 MB per env with the default segment; the collected
segments (`steps` decisions) come on top.

`rollout.match(env, policy_a, policy_b, duels)` plays two policies against each other on both seats
and returns results for `ladder.elo`.

Card text embeddings (needs the `embed` extra; downloads BAAI/bge-small-en-v1.5 once):

```bash
.venv/bin/python -m ocgforge.embed_cards --out .cache/card_text_emb.npz
```

The model's input format is described at the top of `python/src/ocgforge/features.py`; the C++
encoder (`src/encode/encoder.cpp`) produces it, and `python/tests/test_core.py` checks that both
sides agree.

## Layout

| Path | What |
| --- | --- |
| `include/ygo/core`, `src/core` | ocgcore host (RAII), decks, card data and script loading |
| `include/ygo/protocol`, `src/protocol` | engine messages → decision requests with complete candidate lists; responses |
| `include/ygo/observation`, `src/observation` | per-player view of the duel without hidden information |
| `include/ygo/duel`, `src/duel` | `Duel`: runs the engine between decisions, loop protection |
| `include/ygo/encode`, `src/encode` | model input encoding, `VectorEnv` (batched duels on a thread pool) |
| `src/python` | nanobind module `ocgforge._core` |
| `tests/` | Catch2 tests, one executable |
| `third_party/` | pinned dependency commits and our ocgcore patches |
| `tools/setup_deps.py` | fetches/patches dependencies, extracts card data (run by CMake) |
| `python/` | `ocgforge` Python package: model, PPO, ladder, card embeddings |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit together, and
[docs/engine-clone.md](docs/engine-clone.md) for the ocgcore fork and `Duel.clone()`.

## License

ocgcore and CardScripts are AGPL-3.0-or-later and ocgcore is linked into the
binaries, so builds of this project are covered by the AGPL.
