#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ygo/observation/observed_zone.hpp"

namespace ygo::observation {

struct ChainLink {
    std::uint32_t index = 0;
    std::optional<std::uint8_t> activating_player;
    std::optional<ObservationLocator> source;
    std::optional<SemanticZone> activation_zone;
    std::optional<std::uint64_t> effect_description;
    std::vector<ObservationLocator> targets;
};

struct ChainState {
    std::uint32_t length = 0;
    std::vector<ChainLink> links;
};

}  // namespace ygo::observation
