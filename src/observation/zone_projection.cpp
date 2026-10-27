#include "zone_projection.hpp"

#include "common.h"

namespace ygo::observation::detail {
namespace {

ZoneProjection field_zone(SemanticZone zone) {
    return {zone, true};
}

}  // namespace

ZoneProjection project_zone(std::uint32_t engine_location, std::uint8_t controller,
                            std::uint32_t sequence, std::uint64_t duel_flags) {
    (void)controller;
    // ocgcore reports Xyz materials as LOCATION_OVERLAY | <parent location>.
    if ((engine_location & LOCATION_OVERLAY) != 0) {
        return {SemanticZone::Overlay, false};
    }
    switch (engine_location) {
    case LOCATION_DECK:
        return {SemanticZone::MainDeck, false};
    case LOCATION_HAND:
        return {SemanticZone::Hand, false};
    case LOCATION_MZONE:
    case LOCATION_MMZONE:
    case LOCATION_EMZONE:
        return field_zone(SemanticZone::MonsterZone);
    case LOCATION_SZONE:
        if (sequence == 5) {
            return field_zone(SemanticZone::FieldZone);
        }
        if ((duel_flags & DUEL_PZONE) != 0) {
            if ((duel_flags & DUEL_SEPARATE_PZONE) != 0 && (sequence == 6 || sequence == 7)) {
                return field_zone(SemanticZone::PendulumRelevant);
            }
            if ((duel_flags & DUEL_3_COLUMNS_FIELD) != 0 && (sequence == 1 || sequence == 3)) {
                return field_zone(SemanticZone::PendulumRelevant);
            }
            if ((duel_flags & DUEL_SEPARATE_PZONE) == 0 &&
                (duel_flags & DUEL_3_COLUMNS_FIELD) == 0 && (sequence == 0 || sequence == 4)) {
                return field_zone(SemanticZone::PendulumRelevant);
            }
        }
        return field_zone(SemanticZone::SpellTrapZone);
    case LOCATION_STZONE:
        return field_zone(SemanticZone::SpellTrapZone);
    case LOCATION_FZONE:
        return field_zone(SemanticZone::FieldZone);
    case LOCATION_PZONE:
        return field_zone(SemanticZone::PendulumRelevant);
    case LOCATION_GRAVE:
        return {SemanticZone::Graveyard, false};
    case LOCATION_REMOVED:
        return {SemanticZone::Banished, false};
    case LOCATION_EXTRA:
        return {SemanticZone::ExtraDeck, false};
    case LOCATION_OVERLAY:
        return {SemanticZone::Overlay, false};
    default:
        return {SemanticZone::Unknown, false};
    }
}

}  // namespace ygo::observation::detail
