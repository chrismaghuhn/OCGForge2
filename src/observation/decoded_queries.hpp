#pragma once

#include <cstdint>
#include <optional>

#include "query_decoder.hpp"
#include "ygo/core/card_data.hpp"
#include "ygo/observation/observation_builder.hpp"

namespace ygo::observation::detail {

// What observation building reads about a duel, already decoded. The engine path answers it from
// CoreHost queries; client::ClientView from the messages an EDOPro server sends one player.
class DecodedQueries {
public:
    virtual ~DecodedQueries() = default;
    virtual RawFieldSnapshot field() const = 0;
    virtual RawLocationSnapshot location(std::uint8_t controller, std::uint32_t location,
                                         std::uint32_t flags) const = 0;
    // One card; overlay_sequence names an Xyz material of the card at (controller, location, sequence).
    virtual RawCardQuery card(std::uint8_t controller, std::uint32_t location, std::uint32_t sequence,
                              std::optional<std::uint32_t> overlay_sequence, std::uint32_t flags) const = 0;
    virtual std::optional<core::StaticCardData> printed(std::uint32_t code) const = 0;
};

PlayerObservation build_player_observation(const DecodedQueries& queries, std::uint8_t perspective_player,
                                           ObservationBuildConfig config);

}  // namespace ygo::observation::detail
