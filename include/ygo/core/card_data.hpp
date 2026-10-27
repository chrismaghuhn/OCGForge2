#pragma once

#include <cstdint>
#include <vector>

namespace ygo::core {

struct StaticCardData {
    std::uint32_t code = 0;
    std::uint32_t alias = 0;
    std::uint32_t type = 0;
    std::uint32_t level = 0;
    std::uint32_t attribute = 0;
    std::uint64_t race = 0;
    std::int32_t attack = 0;
    std::int32_t defense = 0;
    std::uint32_t left_scale = 0;
    std::uint32_t right_scale = 0;
    std::uint32_t link_marker = 0;
    std::vector<std::uint16_t> setcodes;
};

}  // namespace ygo::core
