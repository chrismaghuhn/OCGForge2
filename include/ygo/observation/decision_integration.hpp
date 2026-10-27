#pragma once

#include <optional>
#include <string>

#include "ygo/observation/player_observation.hpp"
#include "ygo/protocol/continuation.hpp"


namespace ygo::observation {

// Attach only the identity and continuation reference for a decision. The
// complete legal candidate list remains owned by DecisionRequest and is not
// copied into PlayerObservation.
void attach_decision_context(PlayerObservation& observation,
                             const ygo::protocol::DecisionRequest& request);


// Verify that every card-bearing source/target reference in a request can be
// resolved against the current perspective-safe observation. A false result
// is fail-closed: callers must not manufacture an identity to make a request
// appear consistent.
// The entity a card reference from the engine (code, controller, location,
// sequence; for Xyz materials also the material index) refers to, if the
// observation shows it. code 0 matches any card at that place.
std::optional<ObservationLocator> find_card(const PlayerObservation& observation, std::uint32_t code,
                                            std::uint8_t controller, std::uint32_t location,
                                            std::uint32_t sequence,
                                            std::optional<std::uint32_t> overlay_sequence = std::nullopt);

// Whether, since the start of the current turn, the engine showed this card
// (by passcode) to the observing player in that controller's zone, e.g. an
// effect that looks at the opponent's hand. Such a card may be chosen even
// though the observation keeps it hidden.
bool revealed_this_turn(const PlayerObservation& observation, std::uint32_t code, std::uint8_t controller,
                        std::uint32_t location);

bool candidate_observation_consistent(const PlayerObservation& observation,
                                      const ygo::protocol::ActionCandidate& candidate);


}  // namespace ygo::observation
