#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ygo::observation::detail {

class QueryDecodeError final : public std::runtime_error {
public:
    explicit QueryDecodeError(const char* message) : std::runtime_error(message) {}
    explicit QueryDecodeError(const std::string& message) : std::runtime_error(message) {}
};

struct RawLocation {
    std::uint8_t controller = 0;
    std::uint32_t location = 0;
    std::uint32_t sequence = 0;
    std::uint32_t position = 0;
};

struct RawCounter {
    std::uint32_t type = 0;
    std::uint32_t count = 0;
};

struct RawQueryRecord {
    std::uint32_t flag = 0;
    std::vector<std::uint8_t> payload;
};

struct RawCardQuery {
    std::vector<RawQueryRecord> records;
    std::optional<std::uint32_t> code;
    std::optional<std::uint32_t> position;
    std::optional<std::uint32_t> alias;
    std::optional<std::uint32_t> type;
    std::optional<std::uint32_t> level;
    std::optional<std::uint32_t> rank;
    std::optional<std::uint32_t> attribute;
    std::optional<std::uint64_t> race;
    std::optional<std::uint32_t> attack;
    std::optional<std::uint32_t> defense;
    std::optional<std::uint32_t> base_attack;
    std::optional<std::uint32_t> base_defense;
    std::optional<std::uint32_t> reason;
    std::optional<RawLocation> reason_card;
    std::optional<RawLocation> equip_card;
    std::vector<RawLocation> targets;
    std::vector<std::uint32_t> overlay_codes;
    std::vector<RawCounter> counters;
    std::optional<std::uint8_t> owner;
    std::optional<std::uint32_t> status;
    std::optional<std::uint8_t> is_public;
    std::optional<std::uint32_t> left_scale;
    std::optional<std::uint32_t> right_scale;
    std::optional<std::uint32_t> link_rating;
    std::optional<std::uint32_t> link_markers;
    std::optional<std::uint8_t> is_hidden;
    std::optional<std::uint32_t> cover;
};

struct RawLocationSnapshot {
    std::vector<std::optional<RawCardQuery>> entries;
};

struct RawFieldSlot {
    bool occupied = false;
    std::uint8_t position = 0;
    std::uint32_t overlay_count = 0;
};

struct RawFieldPlayer {
    std::uint32_t life_points = 0;
    std::array<RawFieldSlot, 7> monster_slots{};
    std::array<RawFieldSlot, 8> spell_trap_slots{};
    std::uint32_t main_deck_count = 0;
    std::uint32_t hand_count = 0;
    std::uint32_t graveyard_count = 0;
    std::uint32_t banished_count = 0;
    std::uint32_t extra_deck_count = 0;
    std::uint32_t face_up_extra_deck_count = 0;
};

struct RawChainLink {
    std::uint32_t source_code = 0;
    RawLocation source;
    std::uint8_t triggering_player = 0;
    std::uint8_t triggering_location = 0;
    std::uint32_t triggering_sequence = 0;
    std::uint64_t description = 0;
};

struct RawFieldSnapshot {
    std::uint32_t duel_options = 0;
    std::array<RawFieldPlayer, 2> players{};
    std::vector<RawChainLink> chain;
};

RawCardQuery decode_card_query(const std::vector<std::uint8_t>& bytes);
RawLocationSnapshot decode_location_query(const std::vector<std::uint8_t>& bytes);
RawFieldSnapshot decode_field_query(const std::vector<std::uint8_t>& bytes);

}  // namespace ygo::observation::detail
