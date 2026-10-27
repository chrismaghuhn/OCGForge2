# OCGForge2

A deterministic Yu-Gi-Oh! duel environment for game-AI experiments, built as a
thin C++ layer around [ygopro-core](https://github.com/edo9300/ygopro-core).

Current scope: two fixed decks (Swordsoul Tenyi vs. Salamangreat), full duels,
every legal choice exposed as a candidate, and a per-player view that never
leaks hidden information.

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

The slow soak test (500 random duels) is hidden by default:

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

## Python: model and training

`python/` is the `ocgforge` package: the policy/value model (a set transformer that scores every
legal candidate, ported from Argentum P1), the PPO core, an Elo ladder and the card-text
embeddings.

```bash
uv venv .venv && uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e "python[test]"
.venv/bin/python -m pytest python
```

Card text embeddings (needs the `embed` extra; downloads BAAI/bge-small-en-v1.5 once):

```bash
.venv/bin/python -m ocgforge.embed_cards --cdb .cache/babelcdb/cards.cdb --out .cache/card_text_emb.npz
```

The model's input format is described at the top of `python/src/ocgforge/features.py`. The C++
side that produces it from a `Duel` does not exist yet.

## Layout

| Path | What |
| --- | --- |
| `include/ygo/core`, `src/core` | ocgcore host (RAII), decks, card data and script loading |
| `include/ygo/protocol`, `src/protocol` | engine messages → decision requests with complete candidate lists; responses |
| `include/ygo/observation`, `src/observation` | per-player view of the duel without hidden information |
| `include/ygo/duel`, `src/duel` | `Duel`: runs the engine between decisions, loop protection |
| `tests/` | Catch2 tests, one executable |
| `third_party/` | pinned dependency commits and our ocgcore patches |
| `tools/setup_deps.py` | fetches/patches dependencies, extracts card data (run by CMake) |
| `python/` | `ocgforge` Python package: model, PPO, ladder, card embeddings |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit together.

## License

ocgcore and CardScripts are AGPL-3.0-or-later and ocgcore is linked into the
binaries, so builds of this project are covered by the AGPL.
