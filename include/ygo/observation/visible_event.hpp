#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ygo/observation/observed_zone.hpp"

namespace ygo::observation {

enum class VisibleEventKind {
    Unknown,
    TurnStarted,
    PhaseChanged,
    CardMoved,
    CardRevealed,
    Summoned,
    Set,
    Draw,
    Shuffle,
    RandomizationBoundary,
    LifePointsChanged,
    ChainActivated,
    ChainResolved,
    ChainEnded,
    CardDestroyed,
    CardBanished,
    CardReturned,
    PositionChanged,
    CounterChanged,
    Equipped,
    Unequipped,
    Targeted,
    Win,
};

std::string visible_event_kind_name(VisibleEventKind kind);

struct VisibleGameEvent {
    std::uint64_t event_index = 0;
    std::uint64_t engine_step_index = 0;
    VisibleEventKind kind = VisibleEventKind::Unknown;
    std::optional<std::uint8_t> player;
    std::optional<ObservationLocator> entity;
    std::optional<std::uint32_t> public_passcode;
    std::optional<SemanticZone> from_zone;
    std::optional<SemanticZone> to_zone;
    std::optional<std::uint32_t> count;
    std::optional<std::int32_t> amount;
    std::optional<std::uint32_t> counter_type;
    std::optional<std::uint32_t> phase;
    std::optional<std::uint8_t> winner;
    std::optional<std::uint8_t> win_reason;
    std::optional<std::uint64_t> effect_description;
    std::vector<ObservationLocator> targets;
};

}  // namespace ygo::observation
