#include "ygo/duel/duel.hpp"

#include <sstream>
#include <utility>
#include <vector>

#include "ocgapi_constants.h"
#include "ocgapi_types.h"
#include "ygo/core/core_host.hpp"
#include "ygo/observation/decision_integration.hpp"
#include "ygo/observation/observation_builder.hpp"
#include "ygo/observation/observation_session.hpp"
#include "ygo/protocol/continuation.hpp"
#include "ygo/protocol/message_decoder.hpp"
#include "ygo/util/sha256.hpp"
#include "query_decoder.hpp"

namespace ygo::duel {
namespace {

constexpr std::size_t kRecentDecisions = 20;
// Public events copied into each observation (at least; always back to the turn start).
constexpr std::size_t kEventTail = 512;
// Engine messages between two decisions (draws, phase changes, chain
// resolution, ...). Far above anything a real duel needs.
constexpr std::uint64_t kMaxStepsWithoutDecision = 100000;

std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// ocgcore draws the opening hands from the decks in the order the host adds the cards and does not
// shuffle them first (EDOPro's host shuffles before adding), so the host shuffles the main deck:
// Fisher-Yates from the duel seed, the same on every platform. Same seed, same deck order.
core::Deck shuffled(core::Deck deck, std::uint64_t seed, std::uint8_t player) {
    std::uint64_t state = seed ^ (0xd1b54a32d192ed03ULL * (player + 1));
    auto& cards = deck.main_deck;
    for (std::size_t i = cards.size(); i > 1; --i) {
        const auto j = static_cast<std::size_t>(splitmix64(state) % i);
        std::swap(cards[i - 1], cards[j]);
    }
    return deck;
}

}  // namespace

Duel::Duel(DuelConfig config) : config_(std::move(config)) {
    core::CoreHostConfig host_config;
    host_config.card_scripts_root = config_.card_scripts_root;
    host_config.card_data_tsv = config_.card_data_tsv;
    host_config.seed = core::derive_seed_bundle(config_.seed);
    host_config.duel_flags = DUEL_MODE_MR5;
    host_config.starting_player = config_.starting_player;
    host_config.required_script_codes = core::card_codes(config_.deck0, config_.deck1);

    host_ = std::make_unique<core::CoreHost>(std::move(host_config));
    if (config_.fixture_script.empty()) {
        host_->load_deck(0, shuffled(config_.deck0, config_.seed, 0));
        host_->load_deck(1, shuffled(config_.deck1, config_.seed, 1));
    } else {
        host_->load_script_text("world.lua", config_.fixture_script);
    }
    host_->start_duel();
    for (std::uint8_t player = 0; player < 2; ++player) {
        sessions_[player] = std::make_unique<observation::ObservationSession>(player, DUEL_MODE_MR5);
    }
    advance();
}

Duel::Duel(const Duel& source, std::unique_ptr<core::CoreHost> host)
    : config_(source.config_),
      host_(std::move(host)),
      engine_step_(source.engine_step_),
      decision_count_(source.decision_count_),
      request_(source.request_),
      observation_(source.observation_),
      result_(source.result_),
      seen_states_(source.seen_states_),
      answers_(source.answers_),
      state_(source.state_),
      blocked_answers_(source.blocked_answers_),
      recent_(source.recent_),
      reseeds_(source.reseeds_) {
    for (std::uint8_t player = 0; player < 2; ++player) {
        sessions_[player] = std::make_unique<observation::ObservationSession>(*source.sessions_[player]);
    }
}

std::unique_ptr<Duel> Duel::clone(std::optional<std::uint64_t> seed) const {
    if (!request_) {
        throw std::logic_error("Duel::clone() needs a pending decision (the duel is over)");
    }
    std::unique_ptr<core::CoreHost> host;
    try {
        host = host_->clone(seed);
    } catch (const std::exception& error) {
        fail(std::string("cannot clone the duel: ") + error.what());
    }
    // Not make_unique: the constructor is private.
    std::unique_ptr<Duel> copy(new Duel(*this, std::move(host)));
    if (seed) {
        copy->reseeds_.push_back(*seed);
    }
    return copy;
}

Duel::~Duel() = default;

const protocol::DecisionRequest& Duel::request() const {
    if (!request_) {
        throw std::logic_error("Duel::request() called on a finished duel");
    }
    return *request_;
}

const observation::PlayerObservation* Duel::observation() const noexcept {
    return observation_ ? &*observation_ : nullptr;
}

void Duel::act(const std::string& semantic_key) {
    if (!request_) {
        throw std::logic_error("Duel::act() called on a finished duel");
    }
    const auto& candidate = protocol::select_candidate(*request_, semantic_key);
    if (!state_.empty() && ++answers_[state_][candidate.semantic_key] == config_.block_after) {
        ++blocked_answers_;
    }
    state_.clear();

    std::ostringstream line;
    line << '#' << decision_count_ << " p" << static_cast<int>(request_->player) << ' '
         << protocol::decision_kind_name(request_->kind) << " -> " << candidate.semantic_key;
    recent_.push_back(line.str());
    if (recent_.size() > kRecentDecisions) {
        recent_.pop_front();
    }
    if (++decision_count_ > config_.max_decisions) {
        fail("decision limit of " + std::to_string(config_.max_decisions) + " reached");
    }

    // Multi-card selections are built one card at a time on our side; the
    // engine only sees the final combined response.
    if (request_->continuation) {
        auto transition = protocol::apply_continuation_action(*request_, candidate.semantic_key);
        if (!transition.terminal) {
            publish(std::move(transition.request));
            return;
        }
        submit(std::move(transition.engine_response));
        return;
    }
    submit(candidate.exact_response_bytes);
}

// Takes the response by value: it usually points into request_, which is
// cleared here before the engine continues.
void Duel::submit(std::vector<std::uint8_t> response) {
    request_.reset();
    observation_.reset();
    try {
        host_->submit_response(response);
    } catch (const std::exception& error) {
        fail(std::string("engine rejected response: ") + error.what());
    }
    advance();
}

void Duel::advance() {
    for (std::uint64_t steps = 0;; ++steps) {
        if (steps > kMaxStepsWithoutDecision) {
            fail("engine ran " + std::to_string(kMaxStepsWithoutDecision) +
                 " steps without asking for a decision");
        }
        std::vector<std::uint8_t> message;
        protocol::DecodedMessage decoded;
        try {
            message = host_->process().message;
            const auto step = engine_step_++;
            sessions_[0]->ingest(message, step);
            sessions_[1]->ingest(message, step);
            decoded = protocol::decode_messages(message, step);
        } catch (const std::exception& error) {
            fail(error.what());
        }
        if (decoded.retry) {
            fail("engine sent MSG_RETRY: our last response was invalid");
        }
        if (decoded.terminal) {
            result_ = DuelResult{decoded.winner, decoded.win_reason};
            return;
        }
        if (!decoded.interactive) {
            continue;
        }
        if (decoded.decisions.size() != 1) {
            fail("expected exactly one decision per engine message, got " +
                 std::to_string(decoded.decisions.size()));
        }

        // Same question in the same field state again means no progress so far: answers given
        // block_after times there are dropped, and a player who loops anyway loses. A question with
        // a single candidate is no choice, so it never counts against the player asked.
        auto request = std::move(decoded.decisions.front());
        if (request.candidates.size() > 1) {
            auto state = host_->query_field();
            state.insert(state.end(), message.begin(), message.end());
            auto hash = util::sha256_bytes(state);
            if (++seen_states_[hash] > config_.max_repeats) {
                if (config_.block_after == 0) {
                    fail("duel is stuck: the same decision in the same state was asked " +
                         std::to_string(config_.max_repeats + 1) + " times");
                }
                lose_by_loop(request.player);
                return;
            }
            // Continuation keys name the engine step, so they differ on every visit; those
            // questions are only covered by max_repeats.
            if (config_.block_after != 0 && !request.continuation) {
                if (auto answers = answers_.find(hash); answers != answers_.end()) {
                    std::erase_if(request.candidates, [&](const protocol::ActionCandidate& candidate) {
                        auto given = answers->second.find(candidate.semantic_key);
                        return given != answers->second.end() && given->second >= config_.block_after;
                    });
                    if (request.candidates.empty()) {
                        lose_by_loop(request.player);
                        return;
                    }
                }
                state_ = std::move(hash);
            }
        }
        publish(std::move(request));
        return;
    }
}

void Duel::lose_by_loop(std::uint8_t player) {
    if (player > 1) {
        fail("loop by player " + std::to_string(player) + " in a two-player duel");
    }
    request_.reset();
    observation_.reset();
    state_.clear();
    result_ = DuelResult{static_cast<std::uint8_t>(1 - player), kLoopLoss};
}

void Duel::publish(protocol::DecisionRequest request) {
    try {
        protocol::validate_candidate_set(request);
    } catch (const std::exception& error) {
        fail(error.what());
    }
    request_ = std::move(request);
    if (!config_.build_observations) {
        return;
    }

    observation::ObservationBuildConfig build;
    build.decision_index = decision_count_;
    build.engine_step_index = request_->engine_step_index;
    // Recent events only, unless full observations are asked for: the encoder remembers 64 and
    // "revealed this turn" looks back to the turn start, so keep the whole current turn and at
    // least kEventTail events. The turn/phase state of the whole stream comes from the session.
    const auto& session = *sessions_[request_->player];
    const auto& events = session.visible_events();
    auto first = config_.full_observations || events.size() <= kEventTail ? 0 : events.size() - kEventTail;
    first = std::min(first, session.turn_start());
    build.visible_events.assign(events.begin() + static_cast<std::ptrdiff_t>(first), events.end());
    build.event_summary = session.summary();
    build.detail = config_.full_observations ? observation::ObservationDetail::Full
                                             : observation::ObservationDetail::ModelInput;
    build.knowledge.own_decklist_known = true;
    build.own_deck.known = true;
    build.own_deck.main_deck = request_->player == 0 ? config_.deck0.main_deck : config_.deck1.main_deck;
    observation_ = observation::build_player_observation(*host_, request_->player, std::move(build));
    observation::attach_decision_context(*observation_, *request_);

    // Every card a candidate refers to must be visible to the deciding player.
    for (const auto& candidate : request_->candidates) {
        const bool has_card = (candidate.source_card != 0 && candidate.source_location != 0 &&
                               candidate.source_location != LOCATION_DECK &&
                               candidate.source_location != LOCATION_EXTRA) ||
                              (candidate.target_card != 0 && candidate.target_location != 0 &&
                               candidate.target_location != LOCATION_DECK &&
                               candidate.target_location != LOCATION_EXTRA);
        if (has_card && !observation::candidate_observation_consistent(*observation_, candidate) &&
            !observation::revealed_this_turn(*observation_, candidate.source_card, candidate.source_controller,
                                             candidate.source_location)) {
            fail("candidate refers to a card the player cannot see: " + candidate.semantic_key);
        }
    }
}

DuelSnapshot Duel::snapshot() const {
    DuelSnapshot snapshot;
    const auto field = observation::detail::decode_field_query(host_->query_field());
    for (std::uint8_t player = 0; player < 2; ++player) {
        snapshot.life_points[player] = field.players[player].life_points;
        for (const auto location : {LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE,
                                    LOCATION_REMOVED, LOCATION_EXTRA}) {
            OCG_QueryInfo info{};
            info.flags = QUERY_CODE | QUERY_POSITION | QUERY_OWNER | QUERY_IS_PUBLIC | QUERY_OVERLAY_CARD |
                         QUERY_COUNTERS;
            info.con = player;
            info.loc = location;
            const auto cards = observation::detail::decode_location_query(host_->query_location(info));
            for (std::uint32_t sequence = 0; sequence < cards.entries.size(); ++sequence) {
                const auto& query = cards.entries[sequence];
                if (!query) {
                    continue;
                }
                CardState card;
                card.controller = player;
                card.owner = query->owner.value_or(player);
                card.location = location;
                card.sequence = sequence;
                card.code = query->code.value_or(0);
                card.position = query->position.value_or(0);
                card.is_public = query->is_public.value_or(0) != 0;
                card.materials = query->overlay_codes;
                for (const auto& counter : query->counters) {
                    card.counters.emplace_back(counter.type, counter.count);
                }
                snapshot.cards.push_back(std::move(card));
            }
        }
    }
    return snapshot;
}

void Duel::fail(const std::string& message) const {
    std::ostringstream text;
    text << message << " (seed " << config_.seed;
    for (const auto reseed : reseeds_) {
        text << ", reseeded " << reseed;
    }
    text << ", decision " << decision_count_ << ")";
    if (!recent_.empty()) {
        text << "\nlast decisions:";
        for (const auto& line : recent_) {
            text << "\n  " << line;
        }
    }
    throw DuelError(text.str());
}

}  // namespace ygo::duel
