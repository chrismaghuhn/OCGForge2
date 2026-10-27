#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ygo/observation/visible_event.hpp"

namespace ygo::observation {

// Duel state that follows from the whole event stream, kept up to date as events
// arrive so observations do not have to rescan the stream.
struct EventSummary {
    std::uint32_t turns = 0;
    std::optional<std::uint8_t> turn_player;
    std::optional<std::uint32_t> phase;
    bool terminal = false;
    std::optional<std::uint8_t> winner;
    std::optional<std::uint8_t> win_reason;

    void add(const VisibleGameEvent& event);
};

class ObservationSession final {
public:
    explicit ObservationSession(std::uint8_t perspective_player, std::uint32_t duel_flags = 0);

    void ingest(const std::vector<std::uint8_t>& framed_messages, std::uint64_t engine_step_index);
    void clear();

    std::uint8_t perspective_player() const noexcept { return perspective_player_; }
    const std::vector<VisibleGameEvent>& visible_events() const noexcept { return events_; }
    const EventSummary& summary() const noexcept { return summary_; }
    // Index into visible_events() of the latest TurnStarted (0 before the first turn).
    std::size_t turn_start() const noexcept { return turn_start_; }

private:
    std::uint8_t perspective_player_ = 0;
    std::uint32_t duel_flags_ = 0;
    std::uint64_t next_event_index_ = 0;
    std::vector<VisibleGameEvent> events_;
    EventSummary summary_;
    std::size_t turn_start_ = 0;
};

}  // namespace ygo::observation
