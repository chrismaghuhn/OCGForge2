#pragma once

#include <cstdint>
#include <optional>

#include "query_decoder.hpp"
#include "ygo/core/card_data.hpp"
#include "ygo/observation/observed_card.hpp"

namespace ygo::observation::detail {

struct CardProjectionInput {
    RawCardQuery query;
    std::optional<ygo::core::StaticCardData> printed;
    std::uint8_t owner = 0;
    std::uint8_t controller = 0;
    SemanticZone zone = SemanticZone::Unknown;
    std::optional<std::uint32_t> sequence;
    std::optional<std::uint32_t> overlay_sequence;
    ObservationLocator locator;
    bool identity_visible = false;
    bool current_features_visible = false;
    bool sequence_visible = false;
};

ObservedCard project_card(const CardProjectionInput& input);

}  // namespace ygo::observation::detail
