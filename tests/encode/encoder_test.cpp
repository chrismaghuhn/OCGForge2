#include <catch2/catch_test_macros.hpp>

#include <random>
#include <algorithm>
#include <set>

#include "support.hpp"
#include "ygo/encode/encoder.hpp"
#include "ygo/encode/vector_env.hpp"

using namespace ygo::encode;

namespace {

void check_shape(const EncodedDecision& d) {
    const auto tokens = d.token_card.size();
    const auto candidates = d.candidate_kind.size();
    REQUIRE(candidates > 0);
    REQUIRE(d.token_zone.size() == tokens);
    REQUIRE(d.token_side.size() == tokens);
    REQUIRE(d.token_numeric.size() == tokens * token_numeric_names().size());
    REQUIRE(d.global_numeric.size() == global_numeric_names().size());
    REQUIRE(d.candidate_source.size() == candidates);
    REQUIRE(d.candidate_card.size() == candidates);
    REQUIRE(d.candidate_zone.size() == candidates);
    REQUIRE(d.candidate_targets.size() == candidates * kMaxTargets);
    REQUIRE(d.candidate_numeric.size() == candidates * candidate_numeric_names().size());
    for (const auto index : d.candidate_source) {
        REQUIRE(index >= -1);
        REQUIRE(index < static_cast<std::int64_t>(tokens));
    }
    for (const auto kind : d.candidate_kind) {
        REQUIRE(kind > 0);
        REQUIRE(kind < static_cast<std::int64_t>(candidate_kind_names().size()));
    }
    REQUIRE(d.decision > 0);
    REQUIRE(d.phase > 0);
}

}  // namespace

TEST_CASE("encoded decisions of a random duel are well formed and safe") {
    ygo::duel::Duel duel(test::duel_config(3));
    // Every card of both decks known to the model, so hidden cards can only be UNK by design.
    CardIndex index;
    std::int64_t next = 2;
    for (const auto code : ygo::core::card_codes(test::swordsoul(), test::salamangreat())) {
        index[code] = next++;
    }
    std::mt19937_64 rng(3);
    std::set<std::int64_t> kinds;
    std::size_t hidden_checked = 0;
    while (!duel.finished()) {
        const auto& observation = *duel.observation();
        const auto& request = duel.request();
        const auto encoded = encode(observation, request, index);
        check_shape(encoded);
        kinds.insert(encoded.candidate_kind.begin(), encoded.candidate_kind.end());

        // Ground truth from the observation, not from the encoder's own features: a card whose
        // identity the deciding player does not know is never encoded by identity.
        REQUIRE(encoded.token_card.size() == observation.entities.size());
        for (std::size_t t = 0; t < observation.entities.size(); ++t) {
            if (!observation.entities[t].identity_known) {
                REQUIRE(encoded.token_card[t] == 1);
                ++hidden_checked;
            }
        }
        // Candidates that encode identically must be the same card (equivalent copies), and an
        // identity outside the observation is only given for the player's own cards.
        const auto columns = candidate_numeric_names().size();
        for (std::size_t a = 0; a < request.candidates.size(); ++a) {
            if (encoded.candidate_card[a] != 0 && request.candidates[a].action_kind != ygo::protocol::ActionKind::Option) {
                REQUIRE(request.candidates[a].source_controller == request.player);
            }
            for (std::size_t b = a + 1; b < request.candidates.size(); ++b) {
                const bool same_row =
                    encoded.candidate_kind[a] == encoded.candidate_kind[b] &&
                    encoded.candidate_source[a] == encoded.candidate_source[b] &&
                    encoded.candidate_card[a] == encoded.candidate_card[b] &&
                    encoded.candidate_zone[a] == encoded.candidate_zone[b] &&
                    std::equal(encoded.candidate_targets.begin() + a * kMaxTargets,
                               encoded.candidate_targets.begin() + (a + 1) * kMaxTargets,
                               encoded.candidate_targets.begin() + b * kMaxTargets) &&
                    std::equal(encoded.candidate_numeric.begin() + a * columns,
                               encoded.candidate_numeric.begin() + (a + 1) * columns,
                               encoded.candidate_numeric.begin() + b * columns);
                if (same_row) {
                    CAPTURE(request.candidates[a].semantic_key, request.candidates[b].semantic_key);
                    REQUIRE(request.candidates[a].source_card == request.candidates[b].source_card);
                    REQUIRE(request.candidates[a].source_card != 0);
                }
            }
        }
        const auto& candidates = request.candidates;
        duel.act(candidates[rng() % candidates.size()].semantic_key);
    }
    CHECK(kinds.size() >= 8);  // a full duel exercises many kinds of choices
    CHECK(hidden_checked > 0);
}

TEST_CASE("vector env restarts finished duels and is deterministic") {
    VectorEnvConfig config;
    config.card_scripts_root = OCGFORGE_CACHE_DIR "/cardscripts";
    config.card_data_tsv = OCGFORGE_CACHE_DIR "/card_data.tsv";
    config.decks = {test::swordsoul(), test::salamangreat()};
    config.num_envs = 4;
    config.threads = 2;
    config.seed = 11;

    const auto play = [&](std::size_t threads) {
        config.threads = threads;
        VectorEnv env(config, {});
        std::mt19937_64 rng(1);
        std::vector<std::int64_t> trace;
        while (env.duels_finished() < 4) {
            std::vector<std::int64_t> actions;
            for (std::size_t i = 0; i < env.size(); ++i) {
                check_shape(env.decision(i));
                actions.push_back(static_cast<std::int64_t>(rng() % env.decision(i).candidate_kind.size()));
            }
            trace.insert(trace.end(), actions.begin(), actions.end());
            const auto result = env.step(actions);
            for (std::size_t i = 0; i < env.size(); ++i) {
                if (result.done[i]) {
                    CHECK(result.errors[i].empty());
                    CHECK(result.winner[i] <= 2);
                }
            }
        }
        CHECK(env.duel_errors() == 0);
        return trace;
    };
    CHECK(play(1) == play(3));  // same duels whatever the thread count
}

// The `selected` flag of a multi-card selection, checked against the picks actually
// made (tracked by observation locator), not against the encoder's own marking.
TEST_CASE("cards picked in an ongoing selection are marked selected and nothing else") {
    const auto selected_column = static_cast<std::size_t>(
        std::find(token_numeric_names().begin(), token_numeric_names().end(), "selected") -
        token_numeric_names().begin());
    const auto columns = token_numeric_names().size();
    std::size_t checked_with_picks = 0;
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        ygo::duel::Duel duel(test::duel_config(seed));
        std::mt19937_64 rng(seed);
        std::string selection;              // engine message of the ongoing selection
        std::set<std::string> picked;       // locators of cards picked in it
        while (!duel.finished()) {
            const auto& observation = *duel.observation();
            const auto& request = duel.request();
            const auto encoded = encode(observation, request, {});
            // One selection = one engine message; the continuation id changes with every pick.
            const auto id = request.continuation
                                ? request.continuation->raw_message_hash + "@" + std::to_string(request.engine_step_index)
                                : std::string();
            if (id != selection) {
                picked.clear();
                selection = id;
            }
            if (request.continuation) {
                for (std::size_t t = 0; t < observation.entities.size(); ++t) {
                    const bool marked = encoded.token_numeric[t * columns + selected_column] == 1.0f;
                    CAPTURE(seed, observation.entities[t].locator.value);
                    REQUIRE(marked == (picked.count(observation.entities[t].locator.value) == 1));
                }
                checked_with_picks += picked.empty() ? 0 : 1;
            }
            const auto choice = rng() % request.candidates.size();
            const auto& candidate = request.candidates[choice];
            const auto source = encoded.candidate_source[choice];
            if (request.continuation && candidate.action_kind == ygo::protocol::ActionKind::Pick && source >= 0) {
                picked.insert(observation.entities[static_cast<std::size_t>(source)].locator.value);
            }
            duel.act(candidate.semantic_key);
        }
    }
    CHECK(checked_with_picks > 0);  // the situation actually came up
}
