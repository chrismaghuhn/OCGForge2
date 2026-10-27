#include <catch2/catch_test_macros.hpp>

#include <map>
#include <random>
#include <algorithm>
#include <set>

#include "support.hpp"
#include "ygo/encode/encoder.hpp"
#include "ygo/encode/vector_env.hpp"
#include "ygo/observation/decision_integration.hpp"

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
                const auto& c = request.candidates[a];
                REQUIRE((c.source_controller == request.player ||
                         ygo::observation::revealed_this_turn(observation, c.source_card, c.source_controller,
                                                              c.source_location)));
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
        CHECK(env.start_failures() == 0);
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

namespace {

CardIndex all_cards_index() {
    CardIndex index;
    std::int64_t next = 2;
    for (const auto code : ygo::core::card_codes(test::swordsoul(), test::salamangreat())) {
        index[code] = next++;
    }
    return index;
}

}  // namespace

// Every encoded event is a copy of an event in the deciding player's own stream, in order:
// kind, side, zones, card and numeric fields agree with that event, and nothing is invented.
TEST_CASE("encoded events copy the deciding player's public event stream") {
    using K = ygo::observation::VisibleEventKind;
    const auto index = all_cards_index();
    const auto& kinds = event_kind_names();
    const auto kind_name = [](K kind) {
        switch (kind) {
        case K::TurnStarted: return "turn_started"; case K::CardMoved: return "card_moved";
        case K::CardRevealed: return "card_revealed"; case K::Summoned: return "summoned"; case K::Set: return "set";
        case K::LifePointsChanged: return "life_points_changed"; case K::ChainActivated: return "chain_activated";
        case K::CardDestroyed: return "card_destroyed"; case K::CardBanished: return "card_banished";
        case K::CardReturned: return "card_returned"; case K::PositionChanged: return "position_changed";
        case K::CounterChanged: return "counter_changed"; case K::Equipped: return "equipped";
        case K::Targeted: return "targeted"; default: return "";
        }
    };
    ygo::duel::Duel duel(test::duel_config(5));
    std::mt19937_64 rng(5);
    std::size_t named = 0, decisions = 0;
    while (!duel.finished()) {
        const auto& observation = *duel.observation();
        const auto& events = observation.visible_events;
        const auto encoded = encode(observation, duel.request(), index);
        REQUIRE(encoded.event_kind.size() <= kMaxEvents);
        // Match encoded events to stream events, newest to oldest, in order.
        std::size_t cursor = events.size();
        for (std::size_t i = encoded.event_kind.size(); i-- > 0;) {
            const std::string kind = kinds[static_cast<std::size_t>(encoded.event_kind[i])];
            bool found = false;
            while (cursor-- > 0) {
                const auto& e = events[cursor];
                const auto passcode = e.public_passcode.value_or(0);
                const std::int64_t card = passcode == 0 ? 0 : (index.count(passcode) ? index.at(passcode) : 1);
                const std::int64_t from = e.from_zone ? static_cast<std::int64_t>(*e.from_zone) + 1 : 0;
                const std::int64_t to = e.to_zone ? static_cast<std::int64_t>(*e.to_zone) + 1 : 0;
                if (kind == kind_name(e.kind) && card == encoded.event_card[i] && from == encoded.event_from[i] &&
                    to == encoded.event_to[i]) {
                    found = true;
                    break;
                }
            }
            REQUIRE(found);
            const auto& e = events[cursor];
            named += encoded.event_card[i] != 0;
            if (encoded.event_side[i] != 2) {
                REQUIRE(e.player.has_value());
                REQUIRE(encoded.event_side[i] == (*e.player == observation.perspective_player ? 0 : 1));
            }
            const auto* row = &encoded.event_numeric[i * event_numeric_names().size()];
            const bool life_points = e.kind == K::LifePointsChanged;
            REQUIRE(row[2] == (life_points && e.amount ? std::clamp(*e.amount / 8000.0f, -1.0f, 1.0f) : 0.0f));
            REQUIRE(row[1] == static_cast<float>(encoded.event_kind.size() - 1 - i) / kMaxEvents);
            // turnsAgo: turns started after this event.
            const auto later_turns = std::count_if(events.begin() + static_cast<std::ptrdiff_t>(cursor) + 1, events.end(),
                                                   [](const auto& x) { return x.kind == K::TurnStarted; });
            REQUIRE(row[0] == std::min(static_cast<float>(later_turns) / 8.0f, 1.0f));
        }
        ++decisions;
        const auto& candidates = duel.request().candidates;
        duel.act(candidates[rng() % candidates.size()].semantic_key);
    }
    CHECK(named > 0);
    CHECK(decisions > 100);
}

// event_token must point at the very card the event happened to. Ground truth: the token's
// own identity when the deciding player knows it, else the owner's view of that place
// (locators such as "p1:SPELL_TRAP_ZONE:2" are the same for both players). A link to a
// different card would hand the model a false memory.
TEST_CASE("event links point at the card the event happened to") {
    const auto index = all_cards_index();
    std::size_t checked = 0, hidden_checked = 0, wrong = 0;
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        ygo::duel::Duel duel(test::duel_config(seed));
        std::mt19937_64 rng(seed);
        std::map<std::string, std::uint32_t> owner_view[2];
        while (!duel.finished()) {
            const auto& observation = *duel.observation();
            const auto self = observation.perspective_player;
            owner_view[self].clear();
            for (const auto& card : observation.entities) {
                if (card.identity_known && card.passcode && card.controller == self) {
                    owner_view[self][card.locator.value] = *card.passcode;
                }
            }
            const auto encoded = encode(observation, duel.request(), index);
            for (std::size_t i = 0; i < encoded.event_kind.size(); ++i) {
                const auto token = encoded.event_token[i];
                if (token < 0 || encoded.event_card[i] <= 1) {
                    continue;  // no link, or the event does not name a card to compare with
                }
                const auto& card = observation.entities[static_cast<std::size_t>(token)];
                std::int64_t truth = 0;
                if (card.identity_known && card.passcode) {
                    truth = index.count(*card.passcode) ? index.at(*card.passcode) : 1;
                } else if (card.controller) {
                    const auto& theirs = owner_view[*card.controller];
                    const auto it = theirs.find(card.locator.value);
                    if (it == theirs.end() || !index.count(it->second)) {
                        continue;
                    }
                    truth = index.at(it->second);
                    ++hidden_checked;
                }
                if (truth == 0) {
                    continue;
                }
                ++checked;
                if (truth != encoded.event_card[i]) {
                    ++wrong;
                    // A different card with the same name in the same place is indistinguishable here.
                }
            }
            const auto& candidates = duel.request().candidates;
            duel.act(candidates[rng() % candidates.size()].semantic_key);
        }
    }
    INFO(checked << " links checked (" << hidden_checked << " to face-down cards), " << wrong << " wrong");
    CHECK(checked > 100);
    CHECK(wrong == 0);
}

// Cards the opponent moves face-down (set from hand, face-down to the field) stay unnamed.
TEST_CASE("events never name a card the opponent set face-down") {
    using K = ygo::observation::VisibleEventKind;
    const auto index = all_cards_index();
    std::size_t opponent_sets = 0;
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        ygo::duel::Duel duel(test::duel_config(seed));
        std::mt19937_64 rng(seed);
        while (!duel.finished()) {
            const auto& observation = *duel.observation();
            const auto encoded = encode(observation, duel.request(), index);
            const auto set_kind = std::find(event_kind_names().begin(), event_kind_names().end(), "set") -
                                  event_kind_names().begin();
            for (std::size_t i = 0; i < encoded.event_kind.size(); ++i) {
                if (encoded.event_kind[i] == set_kind && encoded.event_side[i] == 1) {
                    ++opponent_sets;
                    REQUIRE(encoded.event_card[i] == 0);
                }
            }
            for (const auto& e : observation.visible_events) {
                if (e.kind == K::Set && e.player && *e.player != observation.perspective_player) {
                    REQUIRE_FALSE(e.public_passcode.has_value());
                }
            }
            const auto& candidates = duel.request().candidates;
            duel.act(candidates[rng() % candidates.size()].semantic_key);
        }
    }
    CHECK(opponent_sets > 0);
}

// Model-input observations query only some card properties; the encoded input must be
// exactly what a full observation gives.
TEST_CASE("model-input observations encode exactly like full ones") {
    const auto index = all_cards_index();
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        auto lean_config = test::duel_config(seed);
        auto full_config = lean_config;
        full_config.full_observations = true;
        ygo::duel::Duel lean(lean_config), full(full_config);
        std::mt19937_64 rng(seed);
        std::size_t decisions = 0;
        while (!lean.finished()) {
            REQUIRE_FALSE(full.finished());
            const auto a = encode(*lean.observation(), lean.request(), index);
            const auto b = encode(*full.observation(), full.request(), index);
            CAPTURE(seed, decisions);
            REQUIRE(a == b);
            const auto& candidates = lean.request().candidates;
            const auto key = candidates[rng() % candidates.size()].semantic_key;
            lean.act(key);
            full.act(key);
            ++decisions;
        }
        REQUIRE(full.finished());
        CHECK(decisions > 100);
    }
}
