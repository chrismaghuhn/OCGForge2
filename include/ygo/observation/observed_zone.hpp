#pragma once

#include <cstdint>
#include <string>

namespace ygo::observation {

enum class SemanticZone {
    Unknown,
    MainDeck,
    Hand,
    MonsterZone,
    SpellTrapZone,
    Graveyard,
    Banished,
    ExtraDeck,
    FieldZone,
    PendulumRelevant,
    Overlay,
};

enum class Position {
    Unknown = 0,
    FaceUpAttack = 0x1,
    FaceDownAttack = 0x2,
    FaceUpDefense = 0x4,
    FaceDownDefense = 0x8,
};

std::string semantic_zone_name(SemanticZone zone);
std::string position_name(Position position);

struct ObservationLocator {
    std::string value;

    bool empty() const noexcept { return value.empty(); }
    bool operator==(const ObservationLocator& other) const noexcept { return value == other.value; }
    bool operator!=(const ObservationLocator& other) const noexcept { return !(*this == other); }
    bool operator<(const ObservationLocator& other) const noexcept { return value < other.value; }
};

struct ObservedZone {
    std::uint8_t player = 0;
    SemanticZone kind = SemanticZone::Unknown;
    std::uint32_t total_count = 0;
    std::uint32_t public_identity_count = 0;
    std::uint32_t hidden_count = 0;
    bool player_observable_order = false;
};

}  // namespace ygo::observation
