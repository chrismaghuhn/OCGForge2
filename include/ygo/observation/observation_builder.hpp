#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ygo/observation/player_observation.hpp"


namespace ygo::core {
class CoreHost;
}

namespace ygo::observation {

struct ObservationBuildConfig {
    MatchKnowledgeConfig knowledge;
    StaticDeckContext own_deck;
    StaticDeckContext opponent_deck;
    std::uint64_t decision_index = 0;
    std::uint64_t engine_step_index = 0;
    std::vector<VisibleGameEvent> visible_events;
    std::optional<DecisionContext> decision_context;
};

PlayerObservation build_player_observation(const ygo::core::CoreHost& host,
                                           std::uint8_t perspective_player,
                                           const ObservationBuildConfig& config = {});

}  // namespace ygo::observation
