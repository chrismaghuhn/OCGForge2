#include "card_projection.hpp"

#include <array>

#include "common.h"

namespace ygo::observation::detail {
namespace {

std::vector<LinkMarker> link_markers_from_bits(std::uint32_t bits) {
    const std::array<std::pair<std::uint32_t, LinkMarker>, 8> markers = {{
        {LINK_MARKER_BOTTOM_LEFT, LinkMarker::BottomLeft},
        {LINK_MARKER_BOTTOM, LinkMarker::Bottom},
        {LINK_MARKER_BOTTOM_RIGHT, LinkMarker::BottomRight},
        {LINK_MARKER_LEFT, LinkMarker::Left},
        {LINK_MARKER_RIGHT, LinkMarker::Right},
        {LINK_MARKER_TOP_LEFT, LinkMarker::TopLeft},
        {LINK_MARKER_TOP, LinkMarker::Top},
        {LINK_MARKER_TOP_RIGHT, LinkMarker::TopRight},
    }};
    std::vector<LinkMarker> result;
    for (const auto& marker : markers) {
        if ((bits & marker.first) != 0) {
            result.push_back(marker.second);
        }
    }
    return result;
}

Position position_from_raw(const std::optional<std::uint32_t>& value) {
    if (!value.has_value()) {
        return Position::Unknown;
    }
    switch (*value) {
    case POS_FACEUP_ATTACK: return Position::FaceUpAttack;
    case POS_FACEDOWN_ATTACK: return Position::FaceDownAttack;
    case POS_FACEUP_DEFENSE: return Position::FaceUpDefense;
    case POS_FACEDOWN_DEFENSE: return Position::FaceDownDefense;
    default: return Position::Unknown;
    }
}

void set_static_card_properties(const ygo::core::StaticCardData& data, CardProperties& output) {
    output.type = data.type;
    output.attribute = data.attribute;
    output.race = data.race;
    output.attack = data.attack;
    if ((data.type & TYPE_LINK) == 0) {
        output.defense = data.defense;
    }
    if ((data.type & TYPE_XYZ) != 0) {
        output.rank = data.level;
    } else if ((data.type & TYPE_LINK) == 0) {
        output.level = data.level;
    }
    if ((data.type & TYPE_PENDULUM) != 0) {
        output.left_scale = data.left_scale;
        output.right_scale = data.right_scale;
    }
    if ((data.type & TYPE_LINK) != 0) {
        output.link_rating = data.level;
        output.link_markers = link_markers_from_bits(data.link_marker);
    }
}

void set_current_card_properties(const RawCardQuery& query, CardProperties& output) {
    output.type = query.type;
    output.attribute = query.attribute;
    output.race = query.race;
    output.attack = query.attack.has_value() ? std::optional<std::int32_t>(static_cast<std::int32_t>(*query.attack))
                                             : std::nullopt;
    const bool link = query.type.has_value() && ((*query.type & TYPE_LINK) != 0);
    if (!link) {
        output.defense = query.defense.has_value()
                             ? std::optional<std::int32_t>(static_cast<std::int32_t>(*query.defense))
                             : std::nullopt;
        output.base_defense = query.base_defense.has_value()
                                  ? std::optional<std::int32_t>(static_cast<std::int32_t>(*query.base_defense))
                                  : std::nullopt;
    }
    output.base_attack = query.base_attack.has_value()
                             ? std::optional<std::int32_t>(static_cast<std::int32_t>(*query.base_attack))
                             : std::nullopt;
    if (query.type.has_value() && ((*query.type & TYPE_XYZ) != 0)) {
        output.rank = query.rank;
    } else {
        output.level = query.level;
    }
    output.link_rating = query.link_rating;
    if (query.link_markers.has_value()) {
        output.link_markers = link_markers_from_bits(*query.link_markers);
    }
    output.left_scale = query.left_scale;
    output.right_scale = query.right_scale;
    output.status_flags = query.status;
    output.counters.reserve(query.counters.size());
    for (const auto& counter : query.counters) {
        output.counters.push_back({counter.type, counter.count});
    }
}

}  // namespace

ObservedCard project_card(const CardProjectionInput& input) {
    ObservedCard result;
    result.locator = input.locator;
    result.owner = input.owner;
    result.controller = input.controller;
    result.zone = input.zone;
    result.position = position_from_raw(input.query.position);
    result.face_up = result.position == Position::FaceUpAttack || result.position == Position::FaceUpDefense;
    result.face_down = result.position == Position::FaceDownAttack || result.position == Position::FaceDownDefense;
    if (input.sequence_visible) {
        result.sequence = input.sequence;
    }
    result.overlay_sequence = input.overlay_sequence;

    if (!input.identity_visible || !input.query.code.has_value()) {
        return result;
    }
    result.identity_known = true;
    result.passcode = input.query.code;
    if (input.printed.has_value()) {
        result.printed.emplace();
        set_static_card_properties(*input.printed, *result.printed);
    }
    if (input.current_features_visible) {
        result.current.emplace();
        set_current_card_properties(input.query, *result.current);
    }
    return result;
}

}  // namespace ygo::observation::detail
