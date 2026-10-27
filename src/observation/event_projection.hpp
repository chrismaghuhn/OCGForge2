#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ygo/observation/visible_event.hpp"

namespace ygo::observation::detail {

class EventDecodeError final : public std::runtime_error {
public:
    explicit EventDecodeError(const char* message) : std::runtime_error(message) {}
    explicit EventDecodeError(const std::string& message) : std::runtime_error(message) {}
};

std::vector<VisibleGameEvent> project_visible_events(const std::vector<std::uint8_t>& framed_messages,
                                                     std::uint8_t perspective_player,
                                                     std::uint64_t engine_step_index,
                                                     std::uint32_t duel_flags = 0,
                                                     std::uint64_t first_event_index = 0);

}  // namespace ygo::observation::detail
