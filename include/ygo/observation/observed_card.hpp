#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ygo/observation/observed_zone.hpp"

namespace ygo::observation {

enum class LinkMarker {
    BottomLeft,
    Bottom,
    BottomRight,
    Left,
    Right,
    TopLeft,
    Top,
    TopRight,
};

std::string link_marker_name(LinkMarker marker);

struct Counter {
    std::uint32_t type = 0;
    std::uint32_t count = 0;
};

struct CardProperties {
    std::optional<std::uint32_t> type;
    std::optional<std::uint32_t> attribute;
    std::optional<std::uint64_t> race;
    std::optional<std::int32_t> attack;
    std::optional<std::int32_t> defense;
    std::optional<std::int32_t> base_attack;
    std::optional<std::int32_t> base_defense;
    std::optional<std::uint32_t> level;
    std::optional<std::uint32_t> rank;
    std::optional<std::uint32_t> link_rating;
    std::vector<LinkMarker> link_markers;
    std::optional<std::uint32_t> left_scale;
    std::optional<std::uint32_t> right_scale;
    std::optional<std::uint32_t> status_flags;
    std::vector<Counter> counters;
};

struct ObservedCard {
    ObservationLocator locator;
    bool identity_known = false;
    std::optional<std::uint32_t> passcode;
    std::optional<std::uint8_t> owner;
    std::optional<std::uint8_t> controller;
    SemanticZone zone = SemanticZone::Unknown;
    std::optional<std::uint32_t> sequence;
    std::optional<std::uint32_t> overlay_sequence;
    Position position = Position::Unknown;
    bool face_up = false;
    bool face_down = false;
    std::optional<CardProperties> printed;
    std::optional<CardProperties> current;
};

}  // namespace ygo::observation
