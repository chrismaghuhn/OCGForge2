#include "event_projection.hpp"

#include <cstddef>
#include <limits>
#include <string>

#include "common.h"
#include "ocgapi_constants.h"
#include "ygo/observation/observed_zone.hpp"
#include "zone_projection.hpp"

namespace ygo::observation::detail {
namespace {

struct MessageLocation {
    std::uint8_t controller = 0;
    std::uint8_t location = 0;
    std::uint32_t sequence = 0;
    std::uint32_t position = 0;
};

class Reader final {
public:
    explicit Reader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

    std::size_t remaining() const noexcept { return bytes_.size() - offset_; }
    bool empty() const noexcept { return remaining() == 0; }

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
    void finish(const char* context) const {
        if (!empty()) {
            throw EventDecodeError(std::string("trailing bytes in ") + context + " event");
        }
    }

private:
    void require(std::size_t count) const {
        if (count > remaining()) {
            throw EventDecodeError("truncated pinned event message");
        }
    }
    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_ = 0;
};

MessageLocation read_location(Reader& reader) {
    MessageLocation result;
    result.controller = reader.u8();
    result.location = reader.u8();
    result.sequence = reader.u32();
    result.position = reader.u32();
    return result;
}

// MSG_CONFIRM_* carries a compact controller/location/sequence tuple and no
// position field. It is distinct from get_info_location(), which is the
// ten-byte wire shape used by movement and chain messages.
MessageLocation read_confirm_location(Reader& reader) {
    MessageLocation result;
    result.controller = reader.u8();
    result.location = reader.u8();
    result.sequence = reader.u32();
    result.position = POS_FACEUP;
    return result;
}

SemanticZone project_event_zone(const MessageLocation& location, std::uint32_t duel_flags) {
    return project_zone(location.location, location.controller, location.sequence, duel_flags).zone;
}

bool event_identity_visible(const MessageLocation& location, std::uint32_t code,
                            std::uint8_t perspective, bool explicit_reveal = false) {
    if (code == 0 || location.controller == 2) {
        return false;
    }
    if (explicit_reveal) {
        return true;
    }
    if ((location.position & POS_FACEUP) != 0) {
        return true;
    }
    // The event packet does not carry the card owner. A controller match is
    // therefore not enough to prove that a face-down card is known (for
    // example, a stolen face-down card). The snapshot query is the authority
    // for own private identity; the history stream stays conservative.
    (void)perspective;
    return false;
}

std::optional<ObservationLocator> safe_locator(const MessageLocation& location, std::uint8_t perspective,
                                               std::uint32_t duel_flags, bool explicit_reveal = false) {
    const auto zone = project_event_zone(location, duel_flags);
    if (zone == SemanticZone::MainDeck ||
        (zone == SemanticZone::Hand && location.controller != perspective && !explicit_reveal)) {
        return std::nullopt;
    }
    if (zone == SemanticZone::ExtraDeck && !explicit_reveal && location.controller != perspective &&
        (location.position & POS_FACEUP) == 0) {
        return std::nullopt;
    }
    return ObservationLocator{"p" + std::to_string(location.controller) + ":" +
                              semantic_zone_name(zone) + ":" + std::to_string(location.sequence)};
}

std::uint32_t read_frame_length(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
    if (bytes.size() - offset < 4) {
        throw EventDecodeError("truncated event frame length");
    }
    const auto result = static_cast<std::uint32_t>(bytes[offset]) |
                        static_cast<std::uint32_t>(bytes[offset + 1]) << 8 |
                        static_cast<std::uint32_t>(bytes[offset + 2]) << 16 |
                        static_cast<std::uint32_t>(bytes[offset + 3]) << 24;
    offset += 4;
    return result;
}

}  // namespace

std::vector<VisibleGameEvent> project_visible_events(const std::vector<std::uint8_t>& framed_messages,
                                                     std::uint8_t perspective_player,
                                                     std::uint64_t engine_step_index,
                                                     std::uint32_t duel_flags,
                                                     std::uint64_t first_event_index) {
    if (perspective_player > 1) {
        throw EventDecodeError("event perspective must be player 0 or 1");
    }
    std::vector<VisibleGameEvent> events;
    std::uint64_t next_index = first_event_index;
    const auto emit = [&events, &next_index, engine_step_index](VisibleEventKind kind) -> VisibleGameEvent& {
        events.push_back({});
        auto& event = events.back();
        event.event_index = next_index++;
        event.engine_step_index = engine_step_index;
        event.kind = kind;
        return event;
    };
    std::size_t offset = 0;
    while (offset < framed_messages.size()) {
        const auto frame_length = read_frame_length(framed_messages, offset);
        if (frame_length == 0 || frame_length > framed_messages.size() - offset) {
            throw EventDecodeError("invalid pinned event frame length");
        }
        const auto frame = std::vector<std::uint8_t>(framed_messages.begin() + static_cast<std::ptrdiff_t>(offset),
                                                     framed_messages.begin() + static_cast<std::ptrdiff_t>(offset + frame_length));
        offset += frame_length;
        Reader reader(frame);
        const auto type = reader.u8();
        switch (type) {
        case MSG_NEW_TURN: {
            auto& event = emit(VisibleEventKind::TurnStarted);
            event.player = reader.u8();
            reader.finish("MSG_NEW_TURN");
            break;
        }
        case MSG_NEW_PHASE: {
            auto& event = emit(VisibleEventKind::PhaseChanged);
            event.phase = reader.u16();
            reader.finish("MSG_NEW_PHASE");
            break;
        }
        case MSG_WIN: {
            auto& event = emit(VisibleEventKind::Win);
            event.winner = reader.u8();
            event.win_reason = reader.u8();
            reader.finish("MSG_WIN");
            break;
        }
        case MSG_SHUFFLE_DECK:
        case MSG_SHUFFLE_HAND:
        case MSG_SHUFFLE_EXTRA:
        case MSG_SHUFFLE_SET_CARD:
        case MSG_REVERSE_DECK: {
            auto& event = emit(VisibleEventKind::Shuffle);
            if (type == MSG_SHUFFLE_DECK || type == MSG_SHUFFLE_HAND || type == MSG_SHUFFLE_EXTRA) {
                event.player = reader.u8();
            } else if (type == MSG_SHUFFLE_SET_CARD) {
                // The pinned core writes LOCATION, count, then matching
                // Previous and Current location vectors. The vectors are
                // source facts for validation only; neither is public event
                // data because this message crosses a knowledge boundary.
                const auto shuffle_location = reader.u8();
                const auto count = reader.u8();
                if (shuffle_location != LOCATION_MZONE && shuffle_location != LOCATION_SZONE) {
                    throw EventDecodeError("invalid shuffle location");
                }
                if (count == 0) {
                    throw EventDecodeError("shuffle count must be positive");
                }
                std::optional<std::uint8_t> shuffle_player;
                for (std::uint32_t index = 0; index < count; ++index) {
                    const auto previous = read_location(reader);
                    if (previous.controller > 1) {
                        throw EventDecodeError("invalid shuffle controller");
                    }
                    if (previous.location != shuffle_location) {
                        throw EventDecodeError("shuffle location does not match header");
                    }
                    if (!shuffle_player.has_value()) {
                        shuffle_player = previous.controller;
                    } else if (shuffle_player.value() != previous.controller) {
                        throw EventDecodeError("inconsistent shuffle controller");
                    }
                }
                for (std::uint32_t index = 0; index < count; ++index) {
                    const auto current = read_location(reader);
                    const bool is_zero = current.controller == 0 &&
                                         current.location == 0 &&
                                         current.sequence == 0 &&
                                         current.position == 0;
                    if (is_zero) {
                        continue;
                    }
                    if (current.controller > 1) {
                        throw EventDecodeError("invalid shuffle current controller");
                    }
                    if (current.controller != shuffle_player.value() ||
                        current.location != shuffle_location) {
                        throw EventDecodeError("shuffle current location is inconsistent");
                    }
                }
                event.player = shuffle_player.value();
            }
            // Hand/Extra shuffle packets carry codes. Their identities are
            // intentionally discarded here; the boundary event is enough to
            // invalidate positional knowledge.
            if (type == MSG_SHUFFLE_HAND || type == MSG_SHUFFLE_EXTRA) {
                const auto count = reader.u32();
                if (count > reader.remaining() / sizeof(std::uint32_t)) {
                    throw EventDecodeError("shuffle card count exceeds event");
                }
                for (std::uint32_t index = 0; index < count; ++index) {
                    (void)reader.u32();
                }
            }
            reader.finish("shuffle");
            const auto shuffle_player = event.player;
            auto& boundary = emit(VisibleEventKind::RandomizationBoundary);
            boundary.player = shuffle_player;
            boundary.count = 0;
            break;
        }
        case MSG_MOVE: {
            const auto code = reader.u32();
            const auto from = read_location(reader);
            const auto to = read_location(reader);
            const auto reason = reader.u32();
            reader.finish("MSG_MOVE");
            const auto from_zone = project_event_zone(from, duel_flags);
            const auto to_zone = project_event_zone(to, duel_flags);
            auto kind = VisibleEventKind::CardMoved;
            if (to_zone == SemanticZone::Banished) {
                kind = VisibleEventKind::CardBanished;
            } else if (to_zone == SemanticZone::Graveyard && (reason & REASON_DESTROY) != 0) {
                kind = VisibleEventKind::CardDestroyed;
            } else if (from_zone == SemanticZone::Banished || from_zone == SemanticZone::Graveyard) {
                kind = VisibleEventKind::CardReturned;
            }
            auto& event = emit(kind);
            event.player = to.controller;
            event.from_zone = from_zone;
            event.to_zone = to_zone;
            if (event_identity_visible(from, code, perspective_player) ||
                event_identity_visible(to, code, perspective_player)) {
                event.public_passcode = code;
            }
            event.entity = safe_locator(to, perspective_player, duel_flags);
            break;
        }
        case MSG_POS_CHANGE: {
            const auto code = reader.u32();
            MessageLocation location;
            location.controller = reader.u8();
            location.location = reader.u8();
            location.sequence = reader.u8();
            const auto old_position = reader.u8();
            const auto new_position = reader.u8();
            location.position = new_position;
            reader.finish("MSG_POS_CHANGE");
            auto& event = emit(VisibleEventKind::PositionChanged);
            event.player = location.controller;
            event.to_zone = project_event_zone(location, duel_flags);
            if (event_identity_visible(location, code, perspective_player) ||
                (old_position & POS_FACEUP) != 0 || (new_position & POS_FACEUP) != 0) {
                event.public_passcode = code;
            }
            event.entity = safe_locator(location, perspective_player, duel_flags);
            break;
        }
        case MSG_SET:
        case MSG_SUMMONING:
        case MSG_SPSUMMONING:
        case MSG_FLIPSUMMONING: {
            const auto code = reader.u32();
            const auto location = read_location(reader);
            reader.finish("summon/set event");
            auto& event = emit(type == MSG_SET ? VisibleEventKind::Set : VisibleEventKind::Summoned);
            event.player = location.controller;
            event.to_zone = project_event_zone(location, duel_flags);
            if (event_identity_visible(location, code, perspective_player)) {
                event.public_passcode = code;
            }
            event.entity = safe_locator(location, perspective_player, duel_flags);
            break;
        }
        case MSG_SUMMONED:
        case MSG_SPSUMMONED:
        case MSG_FLIPSUMMONED:
            reader.finish("summoned event");
            (void)emit(VisibleEventKind::Summoned);
            break;
        case MSG_DRAW: {
            const auto player = reader.u8();
            const auto count = reader.u32();
            if (count > reader.remaining() / 8) {
                throw EventDecodeError("draw count exceeds event");
            }
            auto& event = emit(VisibleEventKind::Draw);
            event.player = player;
            event.count = count;
            for (std::uint32_t index = 0; index < count; ++index) {
                const auto code = reader.u32();
                const auto position = reader.u32();
                if (player == perspective_player || (position & POS_FACEUP) != 0) {
                    auto& reveal = emit(VisibleEventKind::CardRevealed);
                    reveal.player = player;
                    reveal.public_passcode = code;
                }
            }
            reader.finish("MSG_DRAW");
            break;
        }
        case MSG_CONFIRM_CARDS:
        case MSG_CONFIRM_DECKTOP:
        case MSG_CONFIRM_EXTRATOP: {
            const auto recipient = reader.u8();
            const auto count = reader.u32();
            // The pinned bundle emits the compact six-byte location for the
            // real confirm path.  The existing public event contract also
            // accepts the older ten-byte location used by the M0-M2 tests.
            // Select the shape from the exact framed byte count so malformed
            // packets remain fail-closed instead of being partially parsed.
            constexpr std::size_t compact_entry_size = sizeof(std::uint32_t) + 6u;
            constexpr std::size_t extended_entry_size = sizeof(std::uint32_t) + 10u;
            const auto payload_size = reader.remaining();
            enum class ConfirmLocationShape { Compact, Extended };
            ConfirmLocationShape shape;
            if (count == 0 && payload_size == 0) {
                shape = ConfirmLocationShape::Compact;
            } else if (count > 0 && count <= payload_size / extended_entry_size &&
                       payload_size == static_cast<std::size_t>(count) * extended_entry_size) {
                shape = ConfirmLocationShape::Extended;
            } else if (count > 0 && count <= payload_size / compact_entry_size &&
                       payload_size == static_cast<std::size_t>(count) * compact_entry_size) {
                shape = ConfirmLocationShape::Compact;
            } else {
                throw EventDecodeError("confirm count exceeds event");
            }
            for (std::uint32_t index = 0; index < count; ++index) {
                const auto code = reader.u32();
                MessageLocation location = shape == ConfirmLocationShape::Extended
                                                ? read_location(reader)
                                                : read_confirm_location(reader);
                auto& event = emit(VisibleEventKind::CardRevealed);
                event.player = location.controller;
                event.to_zone = project_event_zone(location, duel_flags);
                if (recipient == perspective_player) {
                    event.public_passcode = code;
                    event.entity = safe_locator(location, perspective_player, duel_flags, true);
                }
            }
            reader.finish("confirm event");
            break;
        }
        case MSG_LPUPDATE: {
            auto& event = emit(VisibleEventKind::LifePointsChanged);
            event.player = reader.u8();
            event.amount = static_cast<std::int32_t>(reader.u32());
            reader.finish("MSG_LPUPDATE");
            break;
        }
        case MSG_DAMAGE:
        case MSG_RECOVER: {
            auto& event = emit(VisibleEventKind::LifePointsChanged);
            event.player = reader.u8();
            const auto amount = static_cast<std::int32_t>(reader.u32());
            event.amount = type == MSG_DAMAGE ? -amount : amount;
            reader.finish("damage/recover event");
            break;
        }
        case MSG_CHAINING: {
            const auto code = reader.u32();
            const auto location = read_location(reader);
            const auto triggering_player = reader.u8();
            (void)reader.u8();
            (void)reader.u32();
            const auto description = reader.u64();
            const auto chain_count = reader.u32();
            auto& event = emit(VisibleEventKind::ChainActivated);
            event.player = triggering_player;
            event.to_zone = project_event_zone(location, duel_flags);
            event.count = chain_count;
            if (event_identity_visible(location, code, perspective_player)) {
                event.public_passcode = code;
                event.effect_description = description;
            }
            event.entity = safe_locator(location, perspective_player, duel_flags);
            reader.finish("MSG_CHAINING");
            break;
        }
        case MSG_CHAINED: {
            auto& event = emit(VisibleEventKind::ChainActivated);
            event.count = reader.u8();
            reader.finish("MSG_CHAINED");
            break;
        }
        case MSG_CHAIN_SOLVING:
        case MSG_CHAIN_SOLVED: {
            auto& event = emit(VisibleEventKind::ChainResolved);
            event.count = reader.u8();
            reader.finish("chain resolved event");
            break;
        }
        case MSG_CHAIN_END:
            reader.finish("MSG_CHAIN_END");
            (void)emit(VisibleEventKind::ChainEnded);
            break;
        case MSG_BECOME_TARGET: {
            const auto count = reader.u32();
            if (count > reader.remaining() / 10) {
                throw EventDecodeError("target count exceeds event");
            }
            auto& event = emit(VisibleEventKind::Targeted);
            for (std::uint32_t index = 0; index < count; ++index) {
                const auto location = read_location(reader);
                if (const auto target = safe_locator(location, perspective_player, duel_flags);
                    target.has_value()) {
                    event.targets.push_back(*target);
                }
            }
            reader.finish("MSG_BECOME_TARGET");
            break;
        }
        case MSG_EQUIP:
        case MSG_UNEQUIP:
        case MSG_CARD_TARGET:
        case MSG_CANCEL_TARGET: {
            const auto source = read_location(reader);
            const auto target = read_location(reader);
            reader.finish("relationship event");
            auto& event = emit(type == MSG_EQUIP ? VisibleEventKind::Equipped :
                               type == MSG_UNEQUIP ? VisibleEventKind::Unequipped : VisibleEventKind::Targeted);
            event.player = source.controller;
            event.entity = safe_locator(source, perspective_player, duel_flags);
            if (const auto target_locator = safe_locator(target, perspective_player, duel_flags); target_locator.has_value()) {
                event.targets.push_back(*target_locator);
            }
            break;
        }
        case MSG_ADD_COUNTER:
        case MSG_REMOVE_COUNTER: {
            auto& event = emit(VisibleEventKind::CounterChanged);
            event.counter_type = reader.u16();
            MessageLocation location;
            location.controller = reader.u8();
            location.location = reader.u8();
            location.sequence = reader.u8();
            location.position = POS_FACEUP;
            event.player = location.controller;
            event.amount = reader.u16();
            event.entity = safe_locator(location, perspective_player, duel_flags);
            reader.finish("counter event");
            break;
        }
        default:
            // Unsupported/private event families are deliberately not copied
            // into the model-facing stream. The event inventory records them.
            break;
        }
    }
    return events;
}

}  // namespace ygo::observation::detail
