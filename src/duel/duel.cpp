#include "ygo/duel/duel.hpp"

#include <sstream>
#include <utility>

#include "ocgapi_constants.h"
#include "ygo/core/core_host.hpp"
#include "ygo/observation/decision_integration.hpp"
#include "ygo/observation/observation_builder.hpp"
#include "ygo/observation/observation_session.hpp"
#include "ygo/protocol/continuation.hpp"
#include "ygo/protocol/message_decoder.hpp"
#include "ygo/util/sha256.hpp"

namespace ygo::duel {
namespace {

constexpr std::size_t kRecentDecisions = 20;
// Engine messages between two decisions (draws, phase changes, chain
// resolution, ...). Far above anything a real duel needs.
constexpr std::uint64_t kMaxStepsWithoutDecision = 100000;

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
    host_->load_deck(0, config_.deck0);
    host_->load_deck(1, config_.deck1);
    host_->start_duel();
    for (std::uint8_t player = 0; player < 2; ++player) {
        sessions_[player] = std::make_unique<observation::ObservationSession>(player, DUEL_MODE_MR5);
    }
    advance();
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

        // Same question in the same field state too often means no progress.
        auto state = host_->query_field();
        state.insert(state.end(), message.begin(), message.end());
        if (++seen_states_[util::sha256_bytes(state)] > config_.max_repeats) {
            fail("duel is stuck: the same decision in the same state was asked " +
                 std::to_string(config_.max_repeats) + " times");
        }
        publish(std::move(decoded.decisions.front()));
        return;
    }
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
    build.visible_events = sessions_[request_->player]->visible_events();
    build.knowledge.own_decklist_known = true;
    build.own_deck.known = true;
    build.own_deck.main_deck = request_->player == 0 ? config_.deck0.main_deck : config_.deck1.main_deck;
    observation_ = observation::build_player_observation(*host_, request_->player, build);
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

void Duel::fail(const std::string& message) const {
    std::ostringstream text;
    text << message << " (seed " << config_.seed << ", decision " << decision_count_ << ")";
    if (!recent_.empty()) {
        text << "\nlast decisions:";
        for (const auto& line : recent_) {
            text << "\n  " << line;
        }
    }
    throw DuelError(text.str());
}

}  // namespace ygo::duel
