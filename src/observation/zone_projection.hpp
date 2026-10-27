#pragma once

#include <cstdint>

#include "ygo/observation/observed_zone.hpp"

namespace ygo::observation::detail {

struct ZoneProjection {
    SemanticZone zone = SemanticZone::Unknown;
    bool is_field_slot = false;
};

ZoneProjection project_zone(std::uint32_t engine_location, std::uint8_t controller,
                            std::uint32_t sequence, std::uint64_t duel_flags);

}  // namespace ygo::observation::detail
