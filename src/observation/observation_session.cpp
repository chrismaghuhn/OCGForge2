#include "ygo/observation/observation_session.hpp"

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
        events_.insert(events_.end(), projected.begin(), projected.end());
    }
}

void ObservationSession::clear() {
    events_.clear();
    next_event_index_ = 0;
}

}  // namespace ygo::observation
