#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <thread>
#include <string>
#include <utility>
#include <vector>

#include "support.hpp"
#include "ygo/encode/encoder.hpp"

using ygo::duel::Duel;

namespace {

ygo::duel::DuelConfig config_for(const std::string& deck0, const std::string& deck1, std::uint64_t seed) {
    auto config = test::duel_config(seed);
    config.deck0 = ygo::core::load_deck(std::string(OCGFORGE_SOURCE_DIR "/decks/") + deck0 + ".ydk");
    config.deck1 = ygo::core::load_deck(std::string(OCGFORGE_SOURCE_DIR "/decks/") + deck1 + ".ydk");
    return config;
}

ygo::encode::CardIndex card_index(const ygo::duel::DuelConfig& config) {
    ygo::encode::CardIndex index;
    for (const auto* deck : {&config.deck0, &config.deck1}) {
        for (const auto* list : {&deck->main_deck, &deck->extra_deck}) {
            for (const auto code : *list) {
                index.emplace(code, static_cast<std::int64_t>(index.size()) + 2);
            }
        }
    }
    return index;
}

std::vector<std::string> keys(const Duel& duel) {
    std::vector<std::string> out;
    for (const auto& candidate : duel.request().candidates) {
        out.push_back(candidate.semantic_key);
    }
    return out;
}

// Everything a player of the duel can tell apart: the legal answers, the model input, the
// counters and the privileged card list. Message order is not compared on purpose (pointer-keyed
// containers in the engine may emit simultaneous messages in another order).
void require_same(const Duel& a, const Duel& b, const ygo::encode::CardIndex& index) {
    REQUIRE(a.finished() == b.finished());
    REQUIRE(a.decision_count() == b.decision_count());
    REQUIRE(a.blocked_answers() == b.blocked_answers());
    if (a.finished()) {
        REQUIRE(a.result()->winner == b.result()->winner);
        REQUIRE(a.result()->reason == b.result()->reason);
        return;
    }
    REQUIRE(a.request().player == b.request().player);
    REQUIRE(keys(a) == keys(b));
    REQUIRE(ygo::encode::encode(*a.observation(), a.request(), index) ==
            ygo::encode::encode(*b.observation(), b.request(), index));
    const auto sa = a.snapshot();
    const auto sb = b.snapshot();
    REQUIRE(sa.life_points[0] == sb.life_points[0]);
    REQUIRE(sa.life_points[1] == sb.life_points[1]);
    REQUIRE(sa.cards.size() == sb.cards.size());
    for (std::size_t i = 0; i < sa.cards.size(); ++i) {
        const auto& x = sa.cards[i];
        const auto& y = sb.cards[i];
        REQUIRE((x.controller == y.controller && x.location == y.location && x.sequence == y.sequence &&
                 x.code == y.code && x.position == y.position && x.is_public == y.is_public &&
                 x.materials == y.materials && x.counters == y.counters));
    }
}

// The engine error (if any) of one answer, e.g. a card script exceeding the Lua call budget.
std::optional<std::string> act(Duel& duel, const std::string& key) {
    try {
        duel.act(key);
        return std::nullopt;
    } catch (const ygo::duel::DuelError& error) {
        return std::string(error.what());
    }
}

// Plays both duels on with the same random answers. An engine error must happen in both (with the
// same message); it ends the duel for the caller (returns false).
bool lockstep(Duel& a, Duel& b, const ygo::encode::CardIndex& index, std::mt19937_64& rng, std::size_t steps,
              std::size_t& done) {
    for (std::size_t step = 0; step < steps && !a.finished(); ++step) {
        const auto candidates = keys(a);
        const auto key = candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng)];
        const auto error_a = act(a, key);
        const auto error_b = act(b, key);
        REQUIRE(error_a == error_b);
        ++done;
        if (error_a) {
            return false;
        }
        require_same(a, b, index);
    }
    return true;
}

struct Totals {
    std::size_t clone_points = 0;
    std::size_t steps = 0;
    std::size_t engine_errors = 0;  // duels ended by an engine error in original and clone alike
};

// Random play; every `every` decisions the duel is cloned and original and clone are played on
// in lockstep for `steps` decisions (the original then goes on from there).
Totals lockstep_duel(const ygo::duel::DuelConfig& config, std::size_t every, std::size_t steps) {
    const auto index = card_index(config);
    Duel duel(config);
    std::mt19937_64 rng(config.seed);
    Totals totals;
    while (!duel.finished()) {
        if (duel.decision_count() % every == 0) {
            auto copy = duel.clone();
            require_same(duel, *copy, index);
            ++totals.clone_points;
            if (!lockstep(duel, *copy, index, rng, steps, totals.steps)) {
                ++totals.engine_errors;
                break;
            }
            continue;
        }
        const auto candidates = keys(duel);
        if (act(duel, candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng)])) {
            ++totals.engine_errors;
            break;
        }
    }
    return totals;
}

}  // namespace

TEST_CASE("a cloned duel plays on exactly like the original", "[clone]") {
    const std::pair<const char*, const char*> pairs[] = {
        {"swordsoul_tenyi", "salamangreat"}, {"branded", "maliss"}, {"doomz", "elfnote"}};
    Totals totals;
    for (std::size_t i = 0; i < std::size(pairs); ++i) {
        const auto config = config_for(pairs[i].first, pairs[i].second, 100 + i);
        CAPTURE(pairs[i].first, pairs[i].second, config.seed);
        const auto t = lockstep_duel(config, 9, 30);
        totals.clone_points += t.clone_points;
        totals.steps += t.steps;
        totals.engine_errors += t.engine_errors;
    }
    CHECK(totals.engine_errors == 0);
    CHECK(totals.clone_points >= 6);
    CHECK(totals.steps >= 100);
}

TEST_CASE("cloned duels over all deck pairs", "[.][slow][clone]") {
    const char* decks[] = {"artmage", "blitzclique", "branded", "doomz", "elfnote", "light_and_darkness_ritual",
                           "maliss", "salamangreat", "swordsoul_tenyi"};
    Totals totals;
    std::uint64_t seed = 500;
    for (std::size_t i = 0; i < std::size(decks); ++i) {
        for (std::size_t j = 0; j < std::size(decks); j += 2) {
            const auto config = config_for(decks[i], decks[j], seed++);
            CAPTURE(decks[i], decks[j], config.seed);
            const auto t = lockstep_duel(config, 3, 60);
            totals.clone_points += t.clone_points;
            totals.steps += t.steps;
            totals.engine_errors += t.engine_errors;
        }
    }
    WARN("clone points " << totals.clone_points << ", lockstep decisions " << totals.steps
                         << ", duels ended by the same engine error in both " << totals.engine_errors);
}

TEST_CASE("a clone outlives its original and matches a replay of the same actions", "[clone]") {
    const auto config = config_for("branded", "swordsoul_tenyi", 7);
    const auto index = card_index(config);
    std::mt19937_64 rng(7);
    std::vector<std::string> history;
    auto original = std::make_unique<Duel>(config);
    for (int i = 0; i < 25 && !original->finished(); ++i) {
        const auto candidates = keys(*original);
        history.push_back(candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng)]);
        original->act(history.back());
    }
    REQUIRE_FALSE(original->finished());
    auto copy = original->clone();
    original.reset();  // the clone must not depend on the source

    Duel replay(config);
    for (const auto& key : history) {
        replay.act(key);
    }
    require_same(replay, *copy, index);
    std::size_t steps = 0;
    while (!replay.finished()) {
        REQUIRE(lockstep(replay, *copy, index, rng, 1000, steps));
    }
    CHECK(copy->finished());
}

TEST_CASE("clones of clones and reseeded clones play on; finished duels cannot be cloned", "[clone]") {
    const auto config = config_for("maliss", "doomz", 11);
    const auto index = card_index(config);
    std::mt19937_64 rng(11);
    Duel duel(config);
    for (int i = 0; i < 15 && !duel.finished(); ++i) {
        const auto candidates = keys(duel);
        duel.act(candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng)]);
    }
    REQUIRE_FALSE(duel.finished());

    auto first = duel.clone();
    auto second = first->clone();
    first.reset();
    require_same(duel, *second, index);
    std::mt19937_64 copy_rng = rng;
    std::size_t steps = 0;
    REQUIRE(lockstep(duel, *second, index, copy_rng, 40, steps));

    // A reseeded clone is a valid duel at the same decision (its future draws may differ).
    auto reseeded = duel.clone(12345);
    CHECK(reseeded->config().seed == duel.config().seed);
    CHECK(reseeded->reseeds() == std::vector<std::uint64_t>{12345});
    CHECK(reseeded->clone(7)->reseeds() == std::vector<std::uint64_t>{12345, 7});
    CHECK(keys(*reseeded) == keys(duel));
    while (!reseeded->finished()) {
        const auto candidates = keys(*reseeded);
        reseeded->act(candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng)]);
    }
    CHECK(reseeded->result()->winner <= 2);

    while (!duel.finished()) {
        const auto candidates = keys(duel);
        duel.act(candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng)]);
    }
    CHECK_THROWS_AS(duel.clone(), std::logic_error);
}

TEST_CASE("one duel can be cloned from several threads at once", "[clone]") {
    // How search uses it: many branches from one root, cloned in parallel. Each clone must match
    // one made alone, and the root must be unchanged (run under asan for memory errors).
    const auto config = config_for("branded", "maliss", 21);
    const auto index = card_index(config);
    std::mt19937_64 rng(21);
    Duel duel(config);
    for (int i = 0; i < 40 && !duel.finished(); ++i) {
        const auto candidates = keys(duel);
        duel.act(candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng)]);
    }
    REQUIRE_FALSE(duel.finished());
    auto untouched = duel.clone();

    constexpr int kThreads = 8;
    constexpr int kPerThread = 4;
    std::vector<std::unique_ptr<Duel>> clones(kThreads * kPerThread);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kPerThread; ++i) {
                clones[t * kPerThread + i] = duel.clone();
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    require_same(duel, *untouched, index);
    for (std::size_t i = 0; i < clones.size(); i += 5) {
        auto reference = duel.clone();
        std::mt19937_64 play(i);
        std::size_t steps = 0;
        lockstep(*reference, *clones[i], index, play, 30, steps);  // false only on an engine error in both
    }
}
