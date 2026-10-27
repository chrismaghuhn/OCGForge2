#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ygo/observation/observation_session.hpp"
#include "ygo/observation/player_observation.hpp"


namespace ygo::core {
class CoreHost;
}

namespace ygo::observation {

enum class ObservationDetail {
    Full,        // every card property the engine reports
    ModelInput,  // only what the model input uses (much cheaper to query); other properties stay empty
};

struct ObservationBuildConfig {
    MatchKnowledgeConfig knowledge;
    StaticDeckContext own_deck;
    StaticDeckContext opponent_deck;
    std::uint64_t decision_index = 0;
    std::uint64_t engine_step_index = 0;
    // The public events to include in the observation: the whole stream, or with
    // `event_summary` set, just its most recent part.
    std::vector<VisibleGameEvent> visible_events;
    // Turn, phase and result derived from the whole stream (ObservationSession::summary());
    // without it they are derived from visible_events.
    std::optional<EventSummary> event_summary;
    std::optional<DecisionContext> decision_context;
    ObservationDetail detail = ObservationDetail::Full;
};

PlayerObservation build_player_observation(const ygo::core::CoreHost& host,
                                           std::uint8_t perspective_player,
                                           ObservationBuildConfig config = {});

}  // namespace ygo::observation
