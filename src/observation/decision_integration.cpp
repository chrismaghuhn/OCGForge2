#include "ygo/observation/decision_integration.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "common.h"
#include "ygo/observation/serialization.hpp"
#include "ygo/protocol/message_decoder.hpp"
#include "zone_projection.hpp"

namespace ygo::observation {
namespace {

std::optional<ObservationLocator> resolve_card(const PlayerObservation& observation,
                                               std::uint32_t code,
                                               std::uint8_t controller,
                                               std::uint32_t location,
                                               std::uint32_t sequence,
                                               std::optional<std::uint32_t> overlay_sequence = std::nullopt) {
    const auto zone = detail::project_zone(location, controller, sequence,
                                           static_cast<std::uint32_t>(observation.match_context.duel_flags)).zone;
    for (const auto& entity : observation.entities) {
        if (entity.controller.value_or(2) != controller || entity.zone != zone) {
            continue;
        }
        if (entity.sequence.has_value() && *entity.sequence != sequence) {
            continue;
        }
        if (overlay_sequence.has_value() && entity.overlay_sequence.has_value() &&
            *entity.overlay_sequence != *overlay_sequence) {
            continue;
        }
        if (code != 0) {
            if (entity.identity_known && entity.passcode.value_or(0) != code) {
                continue;
            }
            if (!entity.identity_known) {
                // The engine has disclosed this code in the legal candidate
                // domain, but the privacy-safe field snapshot must remain
                // redacted. Resolve the existing public locator without
                // copying candidate identity into ObservedCard or its hash.
                return entity.locator;
            }
        }
        return entity.locator;
    }
    return std::nullopt;
}

bool card_reference_consistent(const PlayerObservation& observation,
                               std::uint32_t code,
                               std::uint8_t controller,
                               std::uint32_t location,
                               std::uint32_t sequence,
                               std::optional<std::uint32_t> overlay_sequence = std::nullopt) {
    return resolve_card(observation, code, controller, location, sequence, overlay_sequence).has_value();
}

void append_reference(PlayerObservation& observation,
                      std::uint32_t code,
                      std::uint8_t controller,
                      std::uint32_t location,
                      std::uint32_t sequence
                      ) {
    if (const auto locator = resolve_card(observation, code, controller, location, sequence
    );
        locator.has_value()) {
        if (std::find(observation.decision_context.referenced_entities.begin(),
                      observation.decision_context.referenced_entities.end(), *locator) ==
            observation.decision_context.referenced_entities.end()) {
            observation.decision_context.referenced_entities.push_back(*locator);
        }
    }
}

}  // namespace

void attach_decision_context_impl(PlayerObservation& observation,
                                  const ygo::protocol::DecisionRequest& request
                                  ) {
    DecisionContext context;
    context.decision_id = request.decision_id;
    context.kind = ygo::protocol::decision_kind_name(request.kind);
    context.engine_step_index = request.engine_step_index;
    context.player = request.player;
    context.engine_message_type = request.engine_message_type;
    context.engine_message_name = request.engine_message_name;
    if (request.continuation.has_value()) {
        context.continuation_id = request.continuation->continuation_id;
    }
    observation.decision_context = std::move(context);
    observation.globals.player_to_act = request.player;
    for (const auto& candidate : request.candidates) {
        if (candidate.source_card != 0 && candidate.source_location != 0) {
            append_reference(observation, candidate.source_card, candidate.source_controller,
                             candidate.source_location, candidate.source_sequence
            );
        }
        if (candidate.target_card != 0 && candidate.target_location != 0) {
            append_reference(observation, candidate.target_card, candidate.target_controller,
                             candidate.target_location, candidate.target_sequence
            );
        }
    }
    std::sort(observation.decision_context.referenced_entities.begin(),
              observation.decision_context.referenced_entities.end());
}

void attach_decision_context(PlayerObservation& observation,
                             const ygo::protocol::DecisionRequest& request) {
    attach_decision_context_impl(observation, request
    );
}


bool candidate_observation_consistent_impl(const PlayerObservation& observation,
                                           const ygo::protocol::ActionCandidate& candidate
                                           ) {
    // For Xyz materials ocgcore puts the material index into the position field.
    std::optional<std::uint32_t> source_overlay;
    if ((candidate.source_location & LOCATION_OVERLAY) != 0) {
        source_overlay = candidate.source_position;
    }
    if (candidate.source_card != 0 && candidate.source_location != 0 &&
        !card_reference_consistent(observation, candidate.source_card, candidate.source_controller,
                                   candidate.source_location, candidate.source_sequence, source_overlay)) {
        return false;
    }
    if (candidate.target_card != 0 && candidate.target_location != 0 &&
        !card_reference_consistent(observation, candidate.target_card, candidate.target_controller,
                                   candidate.target_location, candidate.target_sequence
    )) {
        return false;
    }
    return true;
}

bool candidate_observation_consistent(const PlayerObservation& observation,
                                      const ygo::protocol::ActionCandidate& candidate) {
    return candidate_observation_consistent_impl(observation, candidate
    );
}


}  // namespace ygo::observation
