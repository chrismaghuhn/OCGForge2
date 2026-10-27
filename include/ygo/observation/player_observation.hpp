#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ygo/observation/chain_state.hpp"
#include "ygo/observation/match_context.hpp"
#include "ygo/observation/observed_card.hpp"
#include "ygo/observation/observed_player_globals.hpp"
#include "ygo/observation/observed_zone.hpp"
#include "ygo/observation/relationship.hpp"
#include "ygo/observation/visible_event.hpp"

namespace ygo::observation {

struct DecisionContext {
    std::optional<std::string> decision_id;
    std::optional<std::string> kind;
    std::optional<std::uint64_t> engine_step_index;
    std::optional<std::uint8_t> player;
    std::optional<std::uint8_t> engine_message_type;
    std::optional<std::string> engine_message_name;
    std::optional<std::string> continuation_id;
    std::vector<ObservationLocator> referenced_entities;
};

struct PlayerObservation {
    std::string schema_version = "ygo.player_observation.v1";
    std::uint8_t perspective_player = 0;
    std::uint64_t decision_index = 0;
    std::uint64_t engine_step_index = 0;
    ObservedPlayerGlobals globals;
    std::vector<ObservedZone> zones;
    std::vector<ObservedCard> entities;
    std::vector<Relationship> relationships;
    ChainState chain;
    std::vector<VisibleGameEvent> visible_events;
    DecisionContext decision_context;
    MatchContext match_context;
};

}  // namespace ygo::observation
