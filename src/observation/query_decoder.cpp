#include "query_decoder.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <string>

#include "ocgapi_constants.h"

namespace ygo::observation::detail {
namespace {

class Cursor final {
public:
    explicit Cursor(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

    std::size_t remaining() const noexcept { return bytes_.size() - offset_; }
    bool empty() const noexcept { return remaining() == 0; }

    std::uint16_t peek_u16() const {
        if (remaining() < 2) {
            throw QueryDecodeError("truncated pinned query buffer");
        }
        return static_cast<std::uint16_t>(bytes_[offset_]) |
               static_cast<std::uint16_t>(bytes_[offset_ + 1]) << 8;
    }

    std::uint8_t u8() {
        require(1);
        return bytes_[offset_++];
    }

    std::uint16_t u16() {
        require(2);
        const auto value = static_cast<std::uint16_t>(bytes_[offset_]) |
                           static_cast<std::uint16_t>(bytes_[offset_ + 1]) << 8;
        offset_ += 2;
        return value;
    }

    std::uint32_t u32() {
        require(4);
        const auto value = static_cast<std::uint32_t>(bytes_[offset_]) |
                           static_cast<std::uint32_t>(bytes_[offset_ + 1]) << 8 |
                           static_cast<std::uint32_t>(bytes_[offset_ + 2]) << 16 |
                           static_cast<std::uint32_t>(bytes_[offset_ + 3]) << 24;
        offset_ += 4;
        return value;
    }

    std::uint64_t u64() {
        require(8);
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 64; shift += 8) {
            value |= static_cast<std::uint64_t>(bytes_[offset_++]) << shift;
        }
        return value;
    }

    std::vector<std::uint8_t> bytes(std::size_t count) {
        require(count);
        std::vector<std::uint8_t> result(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
                                         bytes_.begin() + static_cast<std::ptrdiff_t>(offset_ + count));
        offset_ += count;
        return result;
    }

private:
    void require(std::size_t count) const {
        if (count > remaining()) {
            throw QueryDecodeError("truncated pinned query buffer");
        }
    }

    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_ = 0;
};

void require_exact(const Cursor& cursor, std::size_t expected, const char* field) {
    if (cursor.remaining() != expected) {
        throw QueryDecodeError(std::string("invalid ") + field + " payload length");
    }
}

void require_at_least(const Cursor& cursor, std::size_t minimum, const char* field) {
    if (cursor.remaining() < minimum) {
        throw QueryDecodeError(std::string("truncated ") + field + " payload");
    }
}

RawLocation read_location(Cursor& cursor) {
    RawLocation result;
    result.controller = cursor.u8();
    result.location = cursor.u8();
    result.sequence = cursor.u32();
    result.position = cursor.u32();
    return result;
}

std::optional<RawLocation> read_optional_location(Cursor& cursor) {
    const auto controller = cursor.u8();
    const auto location = cursor.u8();
    const auto sequence = cursor.u32();
    const auto position = cursor.u32();
    if (controller == 0 && location == 0 && sequence == 0 && position == 0) {
        return std::nullopt;
    }
    return RawLocation{controller, location, sequence, position};
}

bool is_scalar_u32(std::uint32_t flag) {
    switch (flag) {
    case QUERY_CODE:
    case QUERY_POSITION:
    case QUERY_ALIAS:
    case QUERY_TYPE:
    case QUERY_LEVEL:
    case QUERY_RANK:
    case QUERY_ATTRIBUTE:
    case QUERY_ATTACK:
    case QUERY_DEFENSE:
    case QUERY_BASE_ATTACK:
    case QUERY_BASE_DEFENSE:
    case QUERY_REASON:
    case QUERY_STATUS:
    case QUERY_LSCALE:
    case QUERY_RSCALE:
    case QUERY_COVER:
        return true;
    default:
        return false;
    }
}

void decode_record_payload(std::uint32_t flag, const std::vector<std::uint8_t>& payload,
                           RawCardQuery& result) {
    Cursor cursor(payload);
    if (is_scalar_u32(flag)) {
        require_exact(cursor, 4, "scalar query");
        const auto value = cursor.u32();
        switch (flag) {
        case QUERY_CODE: result.code = value; break;
        case QUERY_POSITION: result.position = value; break;
        case QUERY_ALIAS: result.alias = value; break;
        case QUERY_TYPE: result.type = value; break;
        case QUERY_LEVEL: result.level = value; break;
        case QUERY_RANK: result.rank = value; break;
        case QUERY_ATTRIBUTE: result.attribute = value; break;
        case QUERY_ATTACK: result.attack = value; break;
        case QUERY_DEFENSE: result.defense = value; break;
        case QUERY_BASE_ATTACK: result.base_attack = value; break;
        case QUERY_BASE_DEFENSE: result.base_defense = value; break;
        case QUERY_REASON: result.reason = value; break;
        case QUERY_STATUS: result.status = value; break;
        case QUERY_LSCALE: result.left_scale = value; break;
        case QUERY_RSCALE: result.right_scale = value; break;
        case QUERY_COVER: result.cover = value; break;
        default: break;
        }
        return;
    }
    switch (flag) {
    case QUERY_RACE:
        require_exact(cursor, 8, "QUERY_RACE");
        result.race = cursor.u64();
        return;
    case QUERY_REASON_CARD:
        require_exact(cursor, 10, "QUERY_REASON_CARD");
        result.reason_card = read_optional_location(cursor);
        return;
    case QUERY_EQUIP_CARD:
        require_exact(cursor, 10, "QUERY_EQUIP_CARD");
        result.equip_card = read_optional_location(cursor);
        return;
    case QUERY_TARGET_CARD: {
        require_at_least(cursor, 4, "QUERY_TARGET_CARD");
        const auto count = cursor.u32();
        if (count > cursor.remaining() / 10) {
            throw QueryDecodeError("QUERY_TARGET_CARD count exceeds payload");
        }
        result.targets.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            result.targets.push_back(read_location(cursor));
        }
        require_exact(cursor, 0, "QUERY_TARGET_CARD");
        return;
    }
    case QUERY_OVERLAY_CARD: {
        require_at_least(cursor, 4, "QUERY_OVERLAY_CARD");
        const auto count = cursor.u32();
        if (count > cursor.remaining() / 4) {
            throw QueryDecodeError("QUERY_OVERLAY_CARD count exceeds payload");
        }
        result.overlay_codes.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            result.overlay_codes.push_back(cursor.u32());
        }
        require_exact(cursor, 0, "QUERY_OVERLAY_CARD");
        return;
    }
    case QUERY_COUNTERS: {
        require_at_least(cursor, 4, "QUERY_COUNTERS");
        const auto count = cursor.u32();
        if (count > cursor.remaining() / 4) {
            throw QueryDecodeError("QUERY_COUNTERS count exceeds payload");
        }
        result.counters.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto packed = cursor.u32();
            result.counters.push_back({packed & 0xffffU, packed >> 16});
        }
        require_exact(cursor, 0, "QUERY_COUNTERS");
        return;
    }
    case QUERY_OWNER:
        require_exact(cursor, 1, "QUERY_OWNER");
        result.owner = cursor.u8();
        return;
    case QUERY_IS_PUBLIC:
        require_exact(cursor, 1, "QUERY_IS_PUBLIC");
        result.is_public = cursor.u8();
        return;
    case QUERY_LINK:
        require_exact(cursor, 8, "QUERY_LINK");
        result.link_rating = cursor.u32();
        result.link_markers = cursor.u32();
        return;
    case QUERY_IS_HIDDEN:
        require_exact(cursor, 1, "QUERY_IS_HIDDEN");
        result.is_hidden = cursor.u8();
        return;
    default:
        throw QueryDecodeError("unsupported pinned card query flag");
    }
}

RawCardQuery decode_card_query_from_cursor(Cursor& cursor) {
    RawCardQuery result;
    std::set<std::uint32_t> seen_flags;
    bool ended = false;
    while (!cursor.empty()) {
        const auto record_length = cursor.u16();
        if (record_length < sizeof(std::uint32_t) || record_length - sizeof(std::uint32_t) > cursor.remaining()) {
            throw QueryDecodeError("invalid pinned card query record length");
        }
        const auto flag = cursor.u32();
        const auto payload = cursor.bytes(record_length - sizeof(std::uint32_t));
        if (!seen_flags.insert(flag).second) {
            throw QueryDecodeError("duplicate pinned card query flag");
        }
        result.records.push_back({flag, payload});
        if (flag == QUERY_END) {
            if (!payload.empty()) {
                throw QueryDecodeError("QUERY_END has a payload");
            }
            ended = true;
            break;
        }
        decode_record_payload(flag, payload, result);
    }
    if (!ended) {
        throw QueryDecodeError("pinned card query has no QUERY_END after " +
                               std::to_string(result.records.size()) + " records");
    }
    return result;
}

}  // namespace

RawCardQuery decode_card_query(const std::vector<std::uint8_t>& bytes) {
    Cursor cursor(bytes);
    auto result = decode_card_query_from_cursor(cursor);
    if (!cursor.empty()) {
        throw QueryDecodeError("trailing bytes after pinned card query");
    }
    return result;
}

RawLocationSnapshot decode_location_query(const std::vector<std::uint8_t>& bytes) {
    Cursor cursor(bytes);
    if (cursor.remaining() < sizeof(std::uint32_t)) {
        throw QueryDecodeError("truncated query location prefix");
    }
    const auto payload_size = cursor.u32();
    if (payload_size != cursor.remaining()) {
        throw QueryDecodeError("query location prefix does not match payload");
    }

    RawLocationSnapshot result;
    while (!cursor.empty()) {
        if (cursor.remaining() < sizeof(std::uint16_t)) {
            throw QueryDecodeError("truncated query location slot marker");
        }
        if (cursor.peek_u16() == 0) {
            (void)cursor.u16();
            result.entries.emplace_back(std::nullopt);
            continue;
        }
        result.entries.emplace_back(decode_card_query_from_cursor(cursor));
    }
    return result;
}

RawFieldSnapshot decode_field_query(const std::vector<std::uint8_t>& bytes) {
    Cursor cursor(bytes);
    RawFieldSnapshot result;
    result.duel_options = cursor.u32();
    for (auto& player : result.players) {
        player.life_points = cursor.u32();
        for (auto& slot : player.monster_slots) {
            slot.occupied = cursor.u8() != 0;
            if (slot.occupied) {
                slot.position = cursor.u8();
                slot.overlay_count = cursor.u32();
            }
        }
        for (auto& slot : player.spell_trap_slots) {
            slot.occupied = cursor.u8() != 0;
            if (slot.occupied) {
                slot.position = cursor.u8();
                slot.overlay_count = cursor.u32();
            }
        }
        player.main_deck_count = cursor.u32();
        player.hand_count = cursor.u32();
        player.graveyard_count = cursor.u32();
        player.banished_count = cursor.u32();
        player.extra_deck_count = cursor.u32();
        player.face_up_extra_deck_count = cursor.u32();
    }

    const auto chain_count = cursor.u32();
    if (chain_count > cursor.remaining() / 28) {
        throw QueryDecodeError("field chain count exceeds payload");
    }
    result.chain.reserve(chain_count);
    for (std::uint32_t index = 0; index < chain_count; ++index) {
        RawChainLink chain;
        chain.source_code = cursor.u32();
        chain.source = read_location(cursor);
        chain.triggering_player = cursor.u8();
        chain.triggering_location = cursor.u8();
        chain.triggering_sequence = cursor.u32();
        chain.description = cursor.u64();
        result.chain.push_back(chain);
    }
    if (!cursor.empty()) {
        throw QueryDecodeError("trailing bytes after pinned field query");
    }
    return result;
}

}  // namespace ygo::observation::detail
