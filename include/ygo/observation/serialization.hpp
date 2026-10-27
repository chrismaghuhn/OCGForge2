#pragma once

#include <string>

#include "ygo/observation/player_observation.hpp"


namespace ygo::observation {

std::string canonical_serialize_without_hash(const PlayerObservation& observation);
std::string canonical_serialize(const PlayerObservation& observation);
std::string observation_hash(const PlayerObservation& observation);


}  // namespace ygo::observation
