#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace ygo::observation {

struct ObservedPlayerGlobals {
    std::uint64_t duel_flags = 0;
    std::vector<std::uint32_t> life_points;
    std::optional<std::uint8_t> player_to_act;
    std::optional<std::uint8_t> turn_player;
    std::optional<std::uint32_t> turn_count;
    std::optional<std::uint32_t> phase;
    std::uint32_t chain_length = 0;
    std::optional<std::uint8_t> winner;
    std::optional<std::uint8_t> win_reason;
    bool terminal = false;
};

}  // namespace ygo::observation
