#pragma once

#include <cstdint>
#include <vector>

#include "ygo/observation/visible_event.hpp"

namespace ygo::observation {

class ObservationSession final {
public:
    explicit ObservationSession(std::uint8_t perspective_player, std::uint32_t duel_flags = 0);

    void ingest(const std::vector<std::uint8_t>& framed_messages, std::uint64_t engine_step_index);
    void clear();

    std::uint8_t perspective_player() const noexcept { return perspective_player_; }
    const std::vector<VisibleGameEvent>& visible_events() const noexcept { return events_; }

private:
    std::uint8_t perspective_player_ = 0;
    std::uint32_t duel_flags_ = 0;
    std::uint64_t next_event_index_ = 0;
    std::vector<VisibleGameEvent> events_;
};

}  // namespace ygo::observation
