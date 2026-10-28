#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "support.hpp"
#include "ygo/protocol/message_decoder.hpp"

using ygo::duel::Duel;
using ygo::duel::DuelError;
using ygo::protocol::DecisionRequest;

namespace {

struct Game {
    ygo::duel::DuelResult result;
    std::vector<std::string> actions;
};

// Plays a full duel, choosing uniformly at random among the legal candidates.
Game play_random(std::uint64_t seed) {
    Duel duel(test::duel_config(seed));
    std::mt19937_64 rng(seed);
    Game game;
    while (!duel.finished()) {
        const auto& candidates = duel.request().candidates;
        std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
        game.actions.push_back(candidates[pick(rng)].semantic_key);
        duel.act(game.actions.back());
    }
    game.result = *duel.result();
    return game;
}

// Sanitizer builds are ~40x slower; CI lowers this with OCGFORGE_RANDOM_DUELS.
std::uint64_t random_duel_count() {
    const char* value = std::getenv("OCGFORGE_RANDOM_DUELS");
    return value != nullptr ? std::strtoull(value, nullptr, 10) : 25;
}

}  // namespace

TEST_CASE("random duels always finish") {
    for (std::uint64_t seed = 1; seed <= random_duel_count(); ++seed) {
        CAPTURE(seed);
        const auto game = play_random(seed);
        CHECK(game.result.winner <= 2);  // 2 = draw
    }
}

TEST_CASE("many random duels always finish", "[.][slow]") {
    for (std::uint64_t seed = 1000; seed < 1500; ++seed) {
        CAPTURE(seed);
        CHECK_NOTHROW(play_random(seed));
    }
}

TEST_CASE("same seed and same choices give the same duel") {
    const auto first = play_random(42);
    const auto second = play_random(42);
    CHECK(first.actions == second.actions);
    CHECK(first.result.winner == second.result.winner);
    CHECK(first.result.reason == second.result.reason);
}

TEST_CASE("decision limit stops a duel that runs too long") {
    auto config = test::duel_config(7);
    config.max_decisions = 10;
    Duel duel(config);
    REQUIRE_THROWS_WITH(
        [&] {
            while (!duel.finished()) {
                duel.act(duel.request().candidates.front().semantic_key);
            }
        }(),
        Catch::Matchers::ContainsSubstring("decision limit") &&
            Catch::Matchers::ContainsSubstring("last decisions:"));
}

// Reproduces the old Hiita loop on purpose: a policy that always unselects
// when it can makes no progress, and the duel must report itself as stuck
// instead of running forever.
TEST_CASE("select/unselect ping-pong is detected as stuck") {
    bool detected = false;
    for (std::uint64_t seed = 1; seed <= 50 && !detected; ++seed) {
        Duel duel(test::duel_config(seed));
        std::mt19937_64 rng(seed);
        try {
            while (!duel.finished()) {
                const DecisionRequest& request = duel.request();
                const ygo::protocol::ActionCandidate* chosen = nullptr;
                if (request.kind == ygo::protocol::DecisionRequestKind::UnselectCard) {
                    for (const auto& candidate : request.candidates) {
                        if (candidate.card_selection_operation == ygo::protocol::CardSelectionOperation::Unselect) {
                            chosen = &candidate;
                        }
                    }
                    if (chosen == nullptr) {
                        for (const auto& candidate : request.candidates) {
                            if (candidate.card_selection_operation == ygo::protocol::CardSelectionOperation::Select) {
                                chosen = &candidate;
                            }
                        }
                    }
                }
                if (chosen == nullptr) {
                    std::uniform_int_distribution<std::size_t> pick(0, request.candidates.size() - 1);
                    chosen = &request.candidates[pick(rng)];
                }
                duel.act(chosen->semantic_key);
            }
        } catch (const DuelError& error) {
            CAPTURE(seed, error.what());
            REQUIRE_THAT(error.what(), Catch::Matchers::ContainsSubstring("stuck"));
            detected = true;
        }
    }
    CHECK(detected);
}

// Every deck in decks/ (training and hold-out) against every other, random play. Slow; run with
// "[slow]". Duels cut off by the Lua call budget count as finished but are reported.
TEST_CASE("random duels across all deck pairings finish", "[.][slow]") {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(OCGFORGE_SOURCE_DIR "/decks")) {
        if (entry.path().extension() == ".ydk") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    std::size_t budget_failures = 0;
    std::uint64_t seed = 1;
    for (const auto& a : paths) {
        for (const auto& b : paths) {
            auto config = test::duel_config(seed);
            config.deck0 = ygo::core::load_deck(a);
            config.deck1 = ygo::core::load_deck(b);
            std::mt19937_64 rng(seed++);
            CAPTURE(a.stem().string(), b.stem().string(), config.seed);
            try {
                Duel duel(config);
                while (!duel.finished()) {
                    const auto& candidates = duel.request().candidates;
                    duel.act(candidates[rng() % candidates.size()].semantic_key);
                }
            } catch (const DuelError& error) {
                REQUIRE_THAT(error.what(), Catch::Matchers::ContainsSubstring("Lua call budget exceeded"));
                ++budget_failures;
            }
        }
    }
    WARN(budget_failures << " of " << paths.size() * paths.size() << " duels hit the Lua call budget");
}
