#include "ygo/observation/observation_session.hpp"

#include <iterator>
#include <stdexcept>

#include "event_projection.hpp"

namespace ygo::observation {

ObservationSession::ObservationSession(std::uint8_t perspective_player, std::uint32_t duel_flags)
    : perspective_player_(perspective_player), duel_flags_(duel_flags) {
    if (perspective_player > 1) {
        throw std::invalid_argument("observation session perspective must be player 0 or 1");
    }
}

void ObservationSession::ingest(const std::vector<std::uint8_t>& framed_messages,
                                std::uint64_t engine_step_index) {
    auto projected = detail::project_visible_events(framed_messages, perspective_player_,
                                                     engine_step_index, duel_flags_, next_event_index_);
    if (!projected.empty()) {
        next_event_index_ = projected.back().event_index + 1;
        for (std::size_t i = 0; i < projected.size(); ++i) {
            summary_.add(projected[i]);
            if (projected[i].kind == VisibleEventKind::TurnStarted) {
                turn_start_ = events_.size() + i;
            }
        }
        events_.insert(events_.end(), std::make_move_iterator(projected.begin()),
                       std::make_move_iterator(projected.end()));
    }
}

void ObservationSession::clear() {
    events_.clear();
    summary_ = {};
    turn_start_ = 0;
    next_event_index_ = 0;
}

void EventSummary::add(const VisibleGameEvent& event) {
    switch (event.kind) {
    case VisibleEventKind::TurnStarted:
        ++turns;
        if (event.player) {
            turn_player = event.player;
        }
        break;
    case VisibleEventKind::PhaseChanged:
        if (event.phase) {
            phase = event.phase;
        }
        break;
    case VisibleEventKind::Win:
        terminal = true;
        winner = event.winner;
        win_reason = event.win_reason;
        break;
    default:
        break;
    }
}

}  // namespace ygo::observation
