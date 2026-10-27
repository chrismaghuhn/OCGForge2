#include "ygo/protocol/message_decoder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#include "ocgapi_constants.h"
#include "ygo/protocol/protocol_error.hpp"
#include "ygo/protocol/response_builder.hpp"
#include "ygo/util/sha256.hpp"

namespace ygo::protocol {
namespace {

class ByteReader final {
public:
    explicit ByteReader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

    std::size_t remaining() const noexcept { return bytes_.size() - offset_; }

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
        for (std::size_t index = 0; index < 8; ++index) {
            value |= static_cast<std::uint64_t>(bytes_[offset_ + index]) << (index * 8);
        }
        offset_ += 8;
        return value;
    }

    void finish() const {
        if (offset_ != bytes_.size()) {
            throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                                "interactive message has trailing bytes");
        }
    }

private:
    void require(std::size_t length) const {
        if (length > remaining()) {
            throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                                "engine message is shorter than its declared fields");
        }
    }

    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_ = 0;
};

std::vector<std::uint8_t> response_i32(std::uint32_t value) {
    return {static_cast<std::uint8_t>(value & 0xffu), static_cast<std::uint8_t>((value >> 8) & 0xffu),
            static_cast<std::uint8_t>((value >> 16) & 0xffu), static_cast<std::uint8_t>((value >> 24) & 0xffu)};
}

std::string card_key(const char* family, std::uint32_t command, std::uint32_t index,
                     std::uint32_t code, std::uint8_t controller, std::uint32_t location,
                     std::uint32_t sequence) {
    std::ostringstream key;
    key << family << "." << command << "." << index << "." << code << "."
        << static_cast<unsigned>(controller) << "." << location << "." << sequence;
    return key.str();
}

std::string command_key(const char* family, std::uint32_t command, std::uint32_t index) {
    std::ostringstream key;
    key << family << "." << command << "." << index;
    return key.str();
}

ContinuationItem read_card_item(ByteReader& reader, std::uint32_t source_index, bool include_position,
                                bool include_contribution, bool include_tribute_value) {
    ContinuationItem item;
    item.card.code = reader.u32();
    item.card.controller = reader.u8();
    item.card.location = reader.u8();
    item.card.sequence = reader.u32();
    item.card.position = include_position ? reader.u32() : 0;
    item.source_index = source_index;
    if (include_contribution) {
        const auto packed = reader.u32();
        item.primary_value = packed & 0xffffu;
        const auto signed_packed = static_cast<std::int32_t>(packed);
        const auto secondary = signed_packed >> 16;
        item.secondary_value = secondary > 0 ? static_cast<std::uint32_t>(secondary) : 0;
    }
    if (include_tribute_value) {
        item.primary_value = reader.u8();
    }
    return item;
}

void validate_count_against_remaining(const ByteReader& reader, std::uint32_t count, std::size_t entry_size,
                                      const char* context) {
    if (entry_size == 0 || count > reader.remaining() / entry_size) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            std::string(context) + " list exceeds message");
    }
}

DecisionRequest finalize_request(DecisionRequest request, const std::vector<std::uint8_t>& frame,
                                 std::uint64_t engine_step_index) {
    const auto hash = ygo::util::sha256_bytes(frame);
    if (request.continuation.has_value()) {
        auto continuation = *request.continuation;
        continuation.raw_message_hash = hash;
        return make_continuation_request(request.kind, request.player, request.engine_message_type,
                                         request.engine_message_name, engine_step_index,
                                         std::move(continuation));
    }
    request.raw_message_hash = hash;
    request.engine_step_index = engine_step_index;
    request.decision_id = hash + ".decision." + std::to_string(engine_step_index);
    return request;
}

void add_idle_card(DecisionRequest& request, std::uint32_t command, std::uint32_t index,
                   std::uint32_t code, std::uint8_t controller, std::uint32_t location,
                   std::uint32_t sequence) {
    ActionCandidate candidate;
    candidate.action_kind = ActionKind::IdleCommand;
    candidate.semantic_key = card_key("idle", command, index, code, controller, location, sequence);
    candidate.source_card = code;
    candidate.source_controller = controller;
    candidate.source_location = location;
    candidate.source_sequence = sequence;
    candidate.phase = command;
    candidate.source_index = index;
    candidate.choice_index = index;
    candidate.exact_response_bytes = response_i32(command | (index << 16));
    request.candidates.push_back(std::move(candidate));
}

void add_battle_card(DecisionRequest& request, std::uint32_t command, std::uint32_t index,
                     std::uint32_t code, std::uint8_t controller, std::uint32_t location,
                     std::uint32_t sequence) {
    ActionCandidate candidate;
    candidate.action_kind = ActionKind::BattleCommand;
    candidate.semantic_key = card_key("battle", command, index, code, controller, location, sequence);
    candidate.source_card = code;
    candidate.source_controller = controller;
    candidate.source_location = location;
    candidate.source_sequence = sequence;
    candidate.phase = command;
    candidate.source_index = index;
    candidate.choice_index = index;
    candidate.exact_response_bytes = response_i32(command | (index << 16));
    request.candidates.push_back(std::move(candidate));
}

void read_idle_card_list(DecisionRequest& request, ByteReader& reader, std::uint32_t command,
                         std::uint32_t count, std::size_t entry_size, const char* context) {
    if (entry_size != 0 && count > reader.remaining() / entry_size) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            std::string("idle ") + context + " list exceeds message");
    }
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto code = reader.u32();
        const auto controller = reader.u8();
        const auto location = reader.u8();
        std::uint32_t sequence = 0;
        if (entry_size == 7) {
            sequence = reader.u8();
        } else {
            sequence = reader.u32();
        }
        add_idle_card(request, command, index, code, controller, location, sequence);
    }
}

DecisionRequest decode_idle(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_SELECT_IDLECMD) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "idle decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::IdleCommand;
    request.engine_message_type = MSG_SELECT_IDLECMD;
    request.engine_message_name = "MSG_SELECT_IDLECMD";
    request.player = reader.u8();

    const auto summonable = reader.u32();
    for (std::uint32_t index = 0; index < summonable; ++index) {
        // Read the wire fields before calling the helper. The reader mutates its
        // cursor, and function-argument evaluation order is not portable across
        // the compilers used by the supported Windows build paths.
        const auto code = reader.u32();
        const auto controller = reader.u8();
        const auto location = reader.u8();
        const auto sequence = reader.u32();
        add_idle_card(request, 0, index, code, controller, location, sequence);
    }

    const auto special = reader.u32();
    read_idle_card_list(request, reader, 1, special, 10, "special-summon");

    const auto reposition = reader.u32();
    read_idle_card_list(request, reader, 2, reposition, 7, "reposition");

    const auto mset = reader.u32();
    read_idle_card_list(request, reader, 3, mset, 10, "monster-set");

    const auto sset = reader.u32();
    read_idle_card_list(request, reader, 4, sset, 10, "spell-set");

    const auto activate = reader.u32();
    if (activate > reader.remaining() / 19) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "idle activation list exceeds message");
    }
    for (std::uint32_t index = 0; index < activate; ++index) {
        const auto code = reader.u32();
        const auto controller = reader.u8();
        const auto location = reader.u8();
        const auto sequence = reader.u32();
        const auto description = reader.u64();
        (void)reader.u8();
        add_idle_card(request, 5, index, code, controller, location, sequence);
        request.candidates.back().choice_value = description;
    }

    if (reader.u8() != 0) {
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::IdleCommand;
        candidate.semantic_key = command_key("idle", 6, 0);
        candidate.phase = 6;
        candidate.exact_response_bytes = response_i32(6);
        request.candidates.push_back(std::move(candidate));
    }
    if (reader.u8() != 0) {
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::IdleCommand;
        candidate.semantic_key = command_key("idle", 7, 0);
        candidate.phase = 7;
        candidate.exact_response_bytes = response_i32(7);
        request.candidates.push_back(std::move(candidate));
    }
    if (reader.u8() != 0) {
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::IdleCommand;
        candidate.semantic_key = command_key("idle", 8, 0);
        candidate.phase = 8;
        candidate.exact_response_bytes = response_i32(8);
        request.candidates.push_back(std::move(candidate));
    }
    reader.finish();
    validate_candidate_set(request);
    return request;
}

DecisionRequest decode_battle(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_SELECT_BATTLECMD) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "battle decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::BattleCommand;
    request.engine_message_type = MSG_SELECT_BATTLECMD;
    request.engine_message_name = "MSG_SELECT_BATTLECMD";
    request.player = reader.u8();

    const auto activatable = reader.u32();
    if (activatable > reader.remaining() / 19) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "battle activation list exceeds message");
    }
    for (std::uint32_t index = 0; index < activatable; ++index) {
        const auto code = reader.u32();
        const auto controller = reader.u8();
        const auto location = reader.u8();
        const auto sequence = reader.u32();
        const auto description = reader.u64();
        (void)reader.u8();
        add_battle_card(request, 0, index, code, controller, location, sequence);
        request.candidates.back().choice_value = description;
    }

    const auto attackable = reader.u32();
    if (attackable > reader.remaining() / 8) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "battle attack list exceeds message");
    }
    for (std::uint32_t index = 0; index < attackable; ++index) {
        const auto code = reader.u32();
        const auto controller = reader.u8();
        const auto location = reader.u8();
        const auto sequence = reader.u8();
        (void)reader.u8();
        add_battle_card(request, 1, index, code, controller, location, sequence);
    }
    if (reader.u8() != 0) {
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::BattleCommand;
        candidate.semantic_key = command_key("battle", 2, 0);
        candidate.phase = 2;
        candidate.exact_response_bytes = response_i32(2);
        request.candidates.push_back(std::move(candidate));
    }
    if (reader.u8() != 0) {
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::BattleCommand;
        candidate.semantic_key = command_key("battle", 3, 0);
        candidate.phase = 3;
        candidate.exact_response_bytes = response_i32(3);
        request.candidates.push_back(std::move(candidate));
    }
    reader.finish();
    validate_candidate_set(request);
    return request;
}

DecisionRequest decode_yes_no(const std::vector<std::uint8_t>& frame, bool effect) {
    ByteReader reader(frame);
    const auto expected = effect ? MSG_SELECT_EFFECTYN : MSG_SELECT_YESNO;
    if (reader.u8() != expected) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "yes/no decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::YesNo;
    request.engine_message_type = expected;
    request.engine_message_name = effect ? "MSG_SELECT_EFFECTYN" : "MSG_SELECT_YESNO";
    request.player = reader.u8();
    if (effect) {
        const auto code = reader.u32();
        const auto controller = reader.u8();
        const auto location = reader.u8();
        const auto sequence = reader.u32();
        (void)reader.u32();
        (void)reader.u64();
        for (const auto choice : {0u, 1u}) {
            ActionCandidate candidate;
            candidate.action_kind = ActionKind::YesNo;
            candidate.semantic_key = choice == 0 ? "yes_no.no" : "yes_no.yes";
            candidate.choice_value = choice;
            candidate.source_card = code;
            candidate.source_controller = controller;
            candidate.source_location = location;
            candidate.source_sequence = sequence;
            candidate.exact_response_bytes = response_i32(choice);
            request.candidates.push_back(std::move(candidate));
        }
    } else {
        (void)reader.u64();
        for (const auto choice : {0u, 1u}) {
            ActionCandidate candidate;
            candidate.action_kind = ActionKind::YesNo;
            candidate.semantic_key = choice == 0 ? "yes_no.no" : "yes_no.yes";
            candidate.choice_value = choice;
            candidate.exact_response_bytes = response_i32(choice);
            request.candidates.push_back(std::move(candidate));
        }
    }
    reader.finish();
    validate_candidate_set(request);
    return request;
}

DecisionRequest decode_position(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_SELECT_POSITION) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "position decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::Position;
    request.engine_message_type = MSG_SELECT_POSITION;
    request.engine_message_name = "MSG_SELECT_POSITION";
    request.player = reader.u8();
    const auto code = reader.u32();
    const auto positions = static_cast<std::uint8_t>(reader.u8() & 0xfu);
    constexpr std::array<std::uint8_t, 4> choices = {POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK,
                                                     POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE};
    for (const auto position : choices) {
        if ((positions & position) == 0) {
            continue;
        }
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::Position;
        candidate.semantic_key = "position." + std::to_string(code) + "." + std::to_string(position);
        candidate.choice_value = position;
        candidate.source_card = code;
        candidate.position = position;
        candidate.exact_response_bytes = response_i32(position);
        request.candidates.push_back(std::move(candidate));
    }
    reader.finish();
    validate_candidate_set(request);
    return request;
}

DecisionRequest decode_chain(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_SELECT_CHAIN) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "chain decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::Chain;
    request.engine_message_type = MSG_SELECT_CHAIN;
    request.engine_message_name = "MSG_SELECT_CHAIN";
    request.player = reader.u8();
    (void)reader.u8();
    const auto forced = reader.u8() != 0;
    (void)reader.u32();
    (void)reader.u32();
    const auto count = reader.u32();
    if (count > reader.remaining() / 23) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "chain activation list exceeds message");
    }
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto code = reader.u32();
        const auto controller = reader.u8();
        const auto location = reader.u8();
        const auto sequence = reader.u32();
        (void)reader.u32();
        const auto description = reader.u64();
        (void)reader.u8();
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::Chain;
        candidate.choice_value = description;
        candidate.semantic_key = card_key("chain", 0, index, code, controller, location, sequence);
        candidate.choice_index = index;
        candidate.source_card = code;
        candidate.source_controller = controller;
        candidate.source_location = location;
        candidate.source_sequence = sequence;
        candidate.exact_response_bytes = response_i32(index);
        request.candidates.push_back(std::move(candidate));
    }
    if (!forced) {
        ActionCandidate pass;
        pass.action_kind = ActionKind::Chain;
        pass.semantic_key = "chain.pass";
        pass.phase = 1;
        pass.exact_response_bytes = response_i32(std::numeric_limits<std::uint32_t>::max());
        request.candidates.push_back(std::move(pass));
    }
    reader.finish();
    validate_candidate_set(request);
    return request;
}

DecisionRequest decode_card_selection(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_SELECT_CARD) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "card-selection decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::CardSelection;
    request.engine_message_type = MSG_SELECT_CARD;
    request.engine_message_name = "MSG_SELECT_CARD";
    request.player = reader.u8();
    const auto cancelable = reader.u8() != 0;
    const auto min = reader.u32();
    const auto max = reader.u32();
    const auto count = reader.u32();
    validate_count_against_remaining(reader, count, 14, "card-selection");
    if (count == 0 || min > max || max > count) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "card-selection cardinality is outside the pinned wire domain", MSG_SELECT_CARD,
                            request.player, frame);
    }
    std::vector<ContinuationItem> items;
    items.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        items.push_back(read_card_item(reader, index, true, false, false));
    }
    reader.finish();
    if (min == 1 && max == 1) {
        for (const auto& item : items) {
            ActionCandidate candidate;
            candidate.action_kind = ActionKind::CardSelection;
            candidate.semantic_key = card_key("card", 0, item.source_index, item.card.code, item.card.controller,
                                              item.card.location, item.card.sequence);
            candidate.source_card = item.card.code;
            candidate.source_controller = item.card.controller;
            candidate.source_location = item.card.location;
            candidate.source_sequence = item.card.sequence;
            candidate.source_position = item.card.position;
            candidate.source_index = item.source_index;
            candidate.exact_response_bytes = encode_card_index_response({item.source_index});
            request.candidates.push_back(std::move(candidate));
        }
        if (cancelable) {
            ActionCandidate cancel;
            cancel.action_kind = ActionKind::Cancel;
            cancel.semantic_key = "card.cancel";
            cancel.exact_response_bytes = encode_int32_response(-1);
            request.candidates.push_back(std::move(cancel));
        }
        validate_candidate_set(request);
        return request;
    }
    SelectionContinuation continuation;
    continuation.continuation_kind = ContinuationKind::UnorderedSelection;
    continuation.original_message_type = MSG_SELECT_CARD;
    continuation.items = std::move(items);
    continuation.min_count = min;
    continuation.max_count = max;
    continuation.can_cancel = cancelable;
    request.continuation = std::move(continuation);
    return request;
}

DecisionRequest decode_select_option(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_SELECT_OPTION) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "option decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::Option;
    request.engine_message_type = MSG_SELECT_OPTION;
    request.engine_message_name = "MSG_SELECT_OPTION";
    request.player = reader.u8();
    const auto count = reader.u8();
    if (count == 0) {
        throw ProtocolError(ProtocolErrorCode::UnsupportedDecision,
                            "MSG_SELECT_OPTION has no selectable options", MSG_SELECT_OPTION, request.player, frame);
    }
    validate_count_against_remaining(reader, count, 8, "option");
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto value = reader.u64();
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::Option;
        candidate.semantic_key = "option." + std::to_string(index) + "." + std::to_string(value);
        candidate.choice_value = value;
        candidate.choice_index = index;
        candidate.phase = index;
        candidate.exact_response_bytes = encode_int32_response(static_cast<std::int32_t>(index));
        request.candidates.push_back(std::move(candidate));
    }
    reader.finish();
    validate_candidate_set(request);
    return request;
}

DecisionRequest decode_tribute(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_SELECT_TRIBUTE) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "tribute decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::Tribute;
    request.engine_message_type = MSG_SELECT_TRIBUTE;
    request.engine_message_name = "MSG_SELECT_TRIBUTE";
    request.player = reader.u8();
    const auto cancelable = reader.u8() != 0;
    const auto min = reader.u32();
    const auto max = reader.u32();
    const auto count = reader.u32();
    validate_count_against_remaining(reader, count, 11, "tribute");
    if (count == 0 || max == 0 || min > max) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "tribute cardinality is outside the pinned wire domain", MSG_SELECT_TRIBUTE,
                            request.player, frame);
    }
    SelectionContinuation continuation;
    continuation.continuation_kind = ContinuationKind::Tribute;
    continuation.original_message_type = MSG_SELECT_TRIBUTE;
    continuation.required_amount = min;
    continuation.max_count = max;
    continuation.can_cancel = cancelable;
    continuation.items.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        continuation.items.push_back(read_card_item(reader, index, false, false, true));
    }
    reader.finish();
    request.continuation = std::move(continuation);
    return request;
}

DecisionRequest decode_sum(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_SELECT_SUM) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "sum decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::Sum;
    request.engine_message_type = MSG_SELECT_SUM;
    request.engine_message_name = "MSG_SELECT_SUM";
    request.player = reader.u8();
    const auto mode = reader.u8();
    if (mode > 1) {
        throw ProtocolError(ProtocolErrorCode::UnsupportedDecision,
                            "MSG_SELECT_SUM mode is not implemented by the pinned core contract", MSG_SELECT_SUM,
                            request.player, frame);
    }
    const auto target = reader.u32();
    const auto min = reader.u32();
    const auto max = reader.u32();
    const auto mandatory_count = reader.u32();
    validate_count_against_remaining(reader, mandatory_count, 18, "sum mandatory");
    SelectionContinuation continuation;
    continuation.continuation_kind = ContinuationKind::Sum;
    continuation.original_message_type = MSG_SELECT_SUM;
    continuation.target_sum = target;
    continuation.min_count = min;
    continuation.max_count = max;
    continuation.exact_sum = mode == 0;
    continuation.greater_sum = mode == 1;
    continuation.mandatory_items.reserve(mandatory_count);
    for (std::uint32_t index = 0; index < mandatory_count; ++index) {
        continuation.mandatory_items.push_back(read_card_item(reader, index, true, true, false));
    }
    const auto optional_count = reader.u32();
    validate_count_against_remaining(reader, optional_count, 18, "sum optional");
    if (optional_count == 0) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "sum message contains no optional candidates", MSG_SELECT_SUM, request.player, frame);
    }
    continuation.items.reserve(optional_count);
    for (std::uint32_t index = 0; index < optional_count; ++index) {
        continuation.items.push_back(read_card_item(reader, index, true, true, false));
    }
    reader.finish();
    request.continuation = std::move(continuation);
    return request;
}

std::vector<ContinuationItem> decode_zone_items(std::uint8_t player, std::uint32_t flag) {
    std::vector<ContinuationItem> items;
    for (std::uint32_t bit = 0; bit < 32; ++bit) {
        std::uint8_t controller = player;
        std::uint8_t location = LOCATION_SZONE;
        std::uint8_t sequence = 0;
        bool valid = true;
        if (bit <= 6) {
            location = LOCATION_MZONE;
            sequence = static_cast<std::uint8_t>(bit);
        } else if (bit >= 8 && bit <= 15) {
            sequence = static_cast<std::uint8_t>(bit - 8);
        } else if (bit >= 16 && bit <= 22) {
            controller = static_cast<std::uint8_t>(1 - player);
            location = LOCATION_MZONE;
            sequence = static_cast<std::uint8_t>(bit - 16);
        } else if (bit >= 24 && bit <= 31) {
            controller = static_cast<std::uint8_t>(1 - player);
            sequence = static_cast<std::uint8_t>(bit - 24);
        } else {
            valid = false;
        }
        if (valid && (flag & (1u << bit)) == 0) {
            ContinuationItem item;
            item.source_index = bit;
            item.card.controller = controller;
            item.card.location = location;
            item.card.sequence = sequence;
            item.mask_value = 1ull << bit;
            items.push_back(std::move(item));
        }
    }
    return items;
}

DecisionRequest decode_place(const std::vector<std::uint8_t>& frame, bool disfield) {
    ByteReader reader(frame);
    const auto expected = disfield ? MSG_SELECT_DISFIELD : MSG_SELECT_PLACE;
    if (reader.u8() != expected) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "place decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::Place;
    request.engine_message_type = expected;
    request.engine_message_name = disfield ? "MSG_SELECT_DISFIELD" : "MSG_SELECT_PLACE";
    request.player = reader.u8();
    const auto count = reader.u8();
    const auto flag = reader.u32();
    if (count == 0) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "zero-count place message should not be interactive", expected, request.player, frame);
    }
    auto items = decode_zone_items(request.player, flag);
    if (count > items.size()) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "place count exceeds the complete free-zone domain", expected, request.player, frame);
    }
    reader.finish();
    if (count == 1) {
        for (const auto& item : items) {
            ActionCandidate candidate;
            candidate.action_kind = ActionKind::Place;
            candidate.semantic_key = "place." + std::to_string(item.card.controller) + "." +
                                     std::to_string(item.card.location) + "." + std::to_string(item.card.sequence);
            candidate.source_controller = item.card.controller;
            candidate.source_location = item.card.location;
            candidate.source_sequence = item.card.sequence;
            candidate.source_index = item.source_index;
            candidate.exact_response_bytes = encode_zone_response(
                {{item.card.controller, static_cast<std::uint8_t>(item.card.location),
                  static_cast<std::uint8_t>(item.card.sequence)}});
            request.candidates.push_back(std::move(candidate));
        }
        validate_candidate_set(request);
        return request;
    }
    SelectionContinuation continuation;
    continuation.continuation_kind = ContinuationKind::ZonePlacement;
    continuation.original_message_type = expected;
    continuation.items = std::move(items);
    continuation.min_count = count;
    continuation.max_count = count;
    request.continuation = std::move(continuation);
    return request;
}

DecisionRequest decode_counter(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_SELECT_COUNTER) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "counter decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::Counter;
    request.engine_message_type = MSG_SELECT_COUNTER;
    request.engine_message_name = "MSG_SELECT_COUNTER";
    request.player = reader.u8();
    (void)reader.u16();
    const auto required = reader.u16();
    const auto count = reader.u32();
    validate_count_against_remaining(reader, count, 9, "counter");
    if (count == 0) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "zero-card counter message should not be interactive", MSG_SELECT_COUNTER,
                            request.player, frame);
    }
    SelectionContinuation continuation;
    continuation.continuation_kind = ContinuationKind::CounterAllocation;
    continuation.original_message_type = MSG_SELECT_COUNTER;
    continuation.required_amount = required;
    continuation.items.reserve(count);
    std::uint32_t total_capacity = 0;
    for (std::uint32_t index = 0; index < count; ++index) {
        ContinuationItem item;
        item.card.code = reader.u32();
        item.card.controller = reader.u8();
        item.card.location = reader.u8();
        item.card.sequence = reader.u8();
        item.source_index = index;
        item.capacity = reader.u16();
        if (item.capacity > static_cast<std::uint32_t>(std::numeric_limits<std::int16_t>::max())) {
            throw ProtocolError(ProtocolErrorCode::UnsupportedDecision,
                                "counter capacity exceeds the pinned signed response domain", MSG_SELECT_COUNTER,
                                request.player, frame);
        }
        total_capacity += item.capacity;
        continuation.items.push_back(std::move(item));
    }
    reader.finish();
    if (required > total_capacity) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "counter requirement exceeds the complete capacity domain", MSG_SELECT_COUNTER,
                            request.player, frame);
    }
    request.continuation = std::move(continuation);
    return request;
}

DecisionRequest decode_ordering(const std::vector<std::uint8_t>& frame, bool chain) {
    ByteReader reader(frame);
    const auto expected = chain ? MSG_SORT_CHAIN : MSG_SORT_CARD;
    if (reader.u8() != expected) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "ordering decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::Ordering;
    request.engine_message_type = expected;
    request.engine_message_name = chain ? "MSG_SORT_CHAIN" : "MSG_SORT_CARD";
    request.player = reader.u8();
    const auto count = reader.u32();
    validate_count_against_remaining(reader, count, 13, "ordering");
    if (count == 0 || count > 128) {
        throw ProtocolError(ProtocolErrorCode::UnsupportedDecision,
                            "ordering count is outside the pinned signed-byte response domain", expected,
                            request.player, frame);
    }
    SelectionContinuation continuation;
    continuation.continuation_kind = ContinuationKind::Ordering;
    continuation.original_message_type = expected;
    continuation.items.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        ContinuationItem item;
        item.card.code = reader.u32();
        item.card.controller = reader.u8();
        item.card.location = reader.u32();
        item.card.sequence = reader.u32();
        item.source_index = index;
        continuation.items.push_back(std::move(item));
    }
    reader.finish();
    request.continuation = std::move(continuation);
    return request;
}

DecisionRequest decode_unselect(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_SELECT_UNSELECT_CARD) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "unselect decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::UnselectCard;
    request.engine_message_type = MSG_SELECT_UNSELECT_CARD;
    request.engine_message_name = "MSG_SELECT_UNSELECT_CARD";
    request.player = reader.u8();
    const auto finishable = reader.u8() != 0;
    const auto cancelable = reader.u8() != 0;
    (void)reader.u32();
    (void)reader.u32();
    // ocgcore writes the cards that can still be selected first, then the
    // cards that are already selected and can be unselected (playerop.cpp).
    const auto select_count = reader.u32();
    validate_count_against_remaining(reader, select_count, 14, "selectable-card");
    std::uint32_t combined_index = 0;
    for (std::uint32_t index = 0; index < select_count; ++index, ++combined_index) {
        const auto item = read_card_item(reader, index, true, false, false);
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::CardSelection;
        candidate.card_selection_operation = CardSelectionOperation::Select;
        candidate.semantic_key = "unselect.select." + std::to_string(item.source_index) + "." +
                                 std::to_string(item.card.code) + "." + std::to_string(item.card.controller) + "." +
                                 std::to_string(item.card.location) + "." + std::to_string(item.card.sequence);
        candidate.source_card = item.card.code;
        candidate.source_controller = item.card.controller;
        candidate.source_location = item.card.location;
        candidate.source_sequence = item.card.sequence;
        candidate.source_position = item.card.position;
        candidate.source_index = combined_index;
        candidate.exact_response_bytes = encode_int32_response(1);
        const auto index_response = encode_int32_response(static_cast<std::int32_t>(combined_index));
        candidate.exact_response_bytes.insert(candidate.exact_response_bytes.end(), index_response.begin(),
                                              index_response.end());
        request.candidates.push_back(std::move(candidate));
    }
    const auto unselect_count = reader.u32();
    validate_count_against_remaining(reader, unselect_count, 14, "unselectable-card");
    for (std::uint32_t index = 0; index < unselect_count; ++index, ++combined_index) {
        const auto item = read_card_item(reader, index, true, false, false);
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::CardSelection;
        candidate.card_selection_operation = CardSelectionOperation::Unselect;
        candidate.semantic_key = "unselect.unselect." + std::to_string(item.source_index) + "." +
                                 std::to_string(item.card.code) + "." + std::to_string(item.card.controller) + "." +
                                 std::to_string(item.card.location) + "." + std::to_string(item.card.sequence);
        candidate.source_card = item.card.code;
        candidate.source_controller = item.card.controller;
        candidate.source_location = item.card.location;
        candidate.source_sequence = item.card.sequence;
        candidate.source_position = item.card.position;
        candidate.source_index = combined_index;
        candidate.exact_response_bytes = encode_int32_response(1);
        const auto index_response = encode_int32_response(static_cast<std::int32_t>(combined_index));
        candidate.exact_response_bytes.insert(candidate.exact_response_bytes.end(), index_response.begin(),
                                              index_response.end());
        request.candidates.push_back(std::move(candidate));
    }
    reader.finish();
    if (finishable || cancelable) {
        ActionCandidate finish;
        finish.action_kind = finishable ? ActionKind::Finish : ActionKind::Cancel;
        finish.semantic_key = finishable ? "unselect.finish" : "unselect.cancel";
        finish.exact_response_bytes = encode_int32_response(-1);
        request.candidates.push_back(std::move(finish));
    }
    validate_candidate_set(request);
    return request;
}

DecisionRequest decode_announce_number(const std::vector<std::uint8_t>& frame) {
    ByteReader reader(frame);
    if (reader.u8() != MSG_ANNOUNCE_NUMBER) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "announce-number decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::Announcement;
    request.engine_message_type = MSG_ANNOUNCE_NUMBER;
    request.engine_message_name = "MSG_ANNOUNCE_NUMBER";
    request.player = reader.u8();
    const auto count = reader.u8();
    validate_count_against_remaining(reader, count, 8, "announce-number");
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto value = reader.u64();
        ActionCandidate candidate;
        candidate.action_kind = ActionKind::Announcement;
        candidate.semantic_key = "announce_number." + std::to_string(index) + "." + std::to_string(value);
        candidate.choice_value = value;
        candidate.choice_index = index;
        candidate.phase = index;
        candidate.exact_response_bytes = encode_int32_response(static_cast<std::int32_t>(index));
        request.candidates.push_back(std::move(candidate));
    }
    reader.finish();
    validate_candidate_set(request);
    return request;
}

std::uint32_t popcount64(std::uint64_t value) {
    std::uint32_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

DecisionRequest decode_announce_mask(const std::vector<std::uint8_t>& frame, bool race) {
    ByteReader reader(frame);
    const auto expected = race ? MSG_ANNOUNCE_RACE : MSG_ANNOUNCE_ATTRIB;
    if (reader.u8() != expected) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "announcement-mask decoder received another message type");
    }
    DecisionRequest request;
    request.kind = DecisionRequestKind::Announcement;
    request.engine_message_type = expected;
    request.engine_message_name = race ? "MSG_ANNOUNCE_RACE" : "MSG_ANNOUNCE_ATTRIB";
    request.player = reader.u8();
    const auto count = reader.u8();
    const auto available = race ? reader.u64() : reader.u32();
    if (count == 0 || count > popcount64(available)) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage,
                            "announcement count exceeds the engine-provided mask domain", expected,
                            request.player, frame);
    }
    std::vector<ContinuationItem> items;
    for (std::uint32_t bit = 0; bit < (race ? 64u : 32u); ++bit) {
        if ((available & (1ull << bit)) == 0) {
            continue;
        }
        ContinuationItem item;
        item.source_index = bit;
        item.mask_value = 1ull << bit;
        items.push_back(std::move(item));
    }
    reader.finish();
    if (count == 1) {
        for (const auto& item : items) {
            ActionCandidate candidate;
            candidate.action_kind = ActionKind::Announcement;
            candidate.semantic_key = "announce_mask." + std::to_string(item.source_index);
            candidate.choice_index = item.source_index;
            candidate.source_index = item.source_index;
            candidate.exact_response_bytes = race ? encode_uint64_response(item.mask_value)
                                                  : encode_uint32_response(static_cast<std::uint32_t>(item.mask_value));
            request.candidates.push_back(std::move(candidate));
        }
        validate_candidate_set(request);
        return request;
    }
    SelectionContinuation continuation;
    continuation.continuation_kind = ContinuationKind::AnnouncementMask;
    continuation.original_message_type = expected;
    continuation.items = std::move(items);
    continuation.min_count = count;
    continuation.max_count = count;
    continuation.available_mask = available;
    request.continuation = std::move(continuation);
    return request;
}

bool is_unsupported_interactive(std::uint8_t type) {
    switch (type) {
    case MSG_REQUEST_DECK:
    case MSG_ROCK_PAPER_SCISSORS:
    case MSG_ANNOUNCE_CARD:
        return true;
    default:
        return false;
    }
}

bool is_known_noninteractive(std::uint8_t type) {
    switch (type) {
    case MSG_RETRY:
    case MSG_HINT:
    case MSG_WAITING:
    case MSG_START:
    case MSG_UPDATE_DATA:
    case MSG_UPDATE_CARD:
    case MSG_CONFIRM_DECKTOP:
    case MSG_CONFIRM_CARDS:
    case MSG_SHUFFLE_DECK:
    case MSG_SHUFFLE_HAND:
    case MSG_REFRESH_DECK:
    case MSG_SWAP_GRAVE_DECK:
    case MSG_SHUFFLE_SET_CARD:
    case MSG_REVERSE_DECK:
    case MSG_DECK_TOP:
    case MSG_SHUFFLE_EXTRA:
    case MSG_NEW_TURN:
    case MSG_NEW_PHASE:
    case MSG_CONFIRM_EXTRATOP:
    case MSG_MOVE:
    case MSG_POS_CHANGE:
    case MSG_SET:
    case MSG_SWAP:
    case MSG_FIELD_DISABLED:
    case MSG_SUMMONING:
    case MSG_SUMMONED:
    case MSG_SPSUMMONING:
    case MSG_SPSUMMONED:
    case MSG_FLIPSUMMONING:
    case MSG_FLIPSUMMONED:
    case MSG_CHAINING:
    case MSG_CHAINED:
    case MSG_CHAIN_SOLVING:
    case MSG_CHAIN_SOLVED:
    case MSG_CHAIN_END:
    case MSG_CHAIN_NEGATED:
    case MSG_CHAIN_DISABLED:
    case MSG_CARD_SELECTED:
    case MSG_RANDOM_SELECTED:
    case MSG_BECOME_TARGET:
    case MSG_DRAW:
    case MSG_DAMAGE:
    case MSG_RECOVER:
    case MSG_EQUIP:
    case MSG_LPUPDATE:
    case MSG_UNEQUIP:
    case MSG_CARD_TARGET:
    case MSG_CANCEL_TARGET:
    case MSG_PAY_LPCOST:
    case MSG_ADD_COUNTER:
    case MSG_REMOVE_COUNTER:
    case MSG_ATTACK:
    case MSG_BATTLE:
    case MSG_ATTACK_DISABLED:
    case MSG_DAMAGE_STEP_START:
    case MSG_DAMAGE_STEP_END:
    case MSG_MISSED_EFFECT:
    case MSG_BE_CHAIN_TARGET:
    case MSG_CREATE_RELATION:
    case MSG_RELEASE_RELATION:
    case MSG_TOSS_COIN:
    case MSG_TOSS_DICE:
    case MSG_HAND_RES:
    case MSG_CARD_HINT:
    case MSG_TAG_SWAP:
    case MSG_RELOAD_FIELD:
    case MSG_AI_NAME:
    case MSG_SHOW_HINT:
    case MSG_PLAYER_HINT:
    case MSG_MATCH_KILL:
    case MSG_CUSTOM_MSG:
    case MSG_REMOVE_CARDS:
        return true;
    default:
        return false;
    }
}

std::vector<std::uint8_t> read_frame(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
    if (bytes.size() - offset < 4) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "engine message stream has an incomplete frame length");
    }
    const auto size = static_cast<std::uint32_t>(bytes[offset]) |
                      static_cast<std::uint32_t>(bytes[offset + 1]) << 8 |
                      static_cast<std::uint32_t>(bytes[offset + 2]) << 16 |
                      static_cast<std::uint32_t>(bytes[offset + 3]) << 24;
    offset += 4;
    if (size == 0 || size > bytes.size() - offset) {
        throw ProtocolError(ProtocolErrorCode::MalformedMessage, "engine message frame length is invalid");
    }
    std::vector<std::uint8_t> frame(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                                    bytes.begin() + static_cast<std::ptrdiff_t>(offset + size));
    offset += size;
    return frame;
}

[[noreturn]] void unsupported(const std::vector<std::uint8_t>& frame, std::uint8_t type, std::uint8_t player,
                              const char* name) {
    throw ProtocolError(ProtocolErrorCode::UnsupportedDecision,
                        std::string("unsupported interactive engine message: ") + name, type, player, frame);
}

}  // namespace

DecodedMessage decode_messages(const std::vector<std::uint8_t>& bytes, std::uint64_t engine_step_index) {
    DecodedMessage decoded;
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        auto frame = read_frame(bytes, offset);
        const auto type = frame.front();
        decoded.message_type = type;
        if (type == MSG_RETRY) {
            decoded.retry = true;
            continue;
        }
        if (type == MSG_WIN) {
            if (frame.size() != 3) {
                throw ProtocolError(ProtocolErrorCode::MalformedMessage, "MSG_WIN has unexpected length", type);
            }
            decoded.terminal = true;
            decoded.winner = frame[1];
            decoded.win_reason = frame[2];
            continue;
        }
        if (type == MSG_SELECT_IDLECMD) {
            decoded.decisions.push_back(finalize_request(decode_idle(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_BATTLECMD) {
            decoded.decisions.push_back(finalize_request(decode_battle(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_EFFECTYN) {
            decoded.decisions.push_back(finalize_request(decode_yes_no(frame, true), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_YESNO) {
            decoded.decisions.push_back(finalize_request(decode_yes_no(frame, false), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_POSITION) {
            decoded.decisions.push_back(finalize_request(decode_position(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_PLACE) {
            decoded.decisions.push_back(finalize_request(decode_place(frame, false), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_DISFIELD) {
            decoded.decisions.push_back(finalize_request(decode_place(frame, true), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_CHAIN) {
            decoded.decisions.push_back(finalize_request(decode_chain(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_CARD) {
            decoded.decisions.push_back(finalize_request(decode_card_selection(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_OPTION) {
            decoded.decisions.push_back(finalize_request(decode_select_option(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_TRIBUTE) {
            decoded.decisions.push_back(finalize_request(decode_tribute(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_SUM) {
            decoded.decisions.push_back(finalize_request(decode_sum(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_COUNTER) {
            decoded.decisions.push_back(finalize_request(decode_counter(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SORT_CARD) {
            decoded.decisions.push_back(finalize_request(decode_ordering(frame, false), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SORT_CHAIN) {
            decoded.decisions.push_back(finalize_request(decode_ordering(frame, true), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_SELECT_UNSELECT_CARD) {
            decoded.decisions.push_back(finalize_request(decode_unselect(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_ANNOUNCE_NUMBER) {
            decoded.decisions.push_back(finalize_request(decode_announce_number(frame), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_ANNOUNCE_RACE) {
            decoded.decisions.push_back(finalize_request(decode_announce_mask(frame, true), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (type == MSG_ANNOUNCE_ATTRIB) {
            decoded.decisions.push_back(finalize_request(decode_announce_mask(frame, false), frame, engine_step_index));
            decoded.interactive = true;
            continue;
        }
        if (is_unsupported_interactive(type)) {
            const auto player = frame.size() > 1 ? frame[1] : 255;
            unsupported(frame, type, player, "unsupported-selection");
        }
        if (!is_known_noninteractive(type)) {
            unsupported(frame, type, frame.size() > 1 ? frame[1] : 255, "unknown-engine-message");
        }
    }
    return decoded;
}

std::string action_kind_name(ActionKind kind) {
    switch (kind) {
    case ActionKind::IdleCommand:
        return "idle_command";
    case ActionKind::BattleCommand:
        return "battle_command";
    case ActionKind::Chain:
        return "chain";
    case ActionKind::Option:
        return "option";
    case ActionKind::CardSelection:
        return "card_selection";
    case ActionKind::Announcement:
        return "announcement";
    case ActionKind::Place:
        return "place";
    case ActionKind::Position:
        return "position";
    case ActionKind::YesNo:
        return "yes_no";
    case ActionKind::Pick:
        return "pick";
    case ActionKind::Finish:
        return "finish";
    case ActionKind::Cancel:
        return "cancel";
    case ActionKind::AssignAmount:
        return "assign_amount";
    }
    return "unknown";
}

std::string decision_kind_name(DecisionRequestKind kind) {
    switch (kind) {
    case DecisionRequestKind::IdleCommand:
        return "idle_command";
    case DecisionRequestKind::BattleCommand:
        return "battle_command";
    case DecisionRequestKind::Chain:
        return "chain";
    case DecisionRequestKind::Option:
        return "option";
    case DecisionRequestKind::CardSelection:
        return "card_selection";
    case DecisionRequestKind::Tribute:
        return "tribute";
    case DecisionRequestKind::Sum:
        return "sum";
    case DecisionRequestKind::Place:
        return "place";
    case DecisionRequestKind::Counter:
        return "counter";
    case DecisionRequestKind::Ordering:
        return "ordering";
    case DecisionRequestKind::Announcement:
        return "announcement";
    case DecisionRequestKind::UnselectCard:
        return "unselect_card";
    case DecisionRequestKind::Position:
        return "position";
    case DecisionRequestKind::YesNo:
        return "yes_no";
    case DecisionRequestKind::Unsupported:
        return "unsupported";
    }
    return "unknown";
}

void validate_candidate_set(const DecisionRequest& request) {
    if (request.candidates.empty()) {
        throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                            "interactive request contains no legal candidates");
    }
    for (std::size_t i = 0; i < request.candidates.size(); ++i) {
        const auto& candidate = request.candidates[i];
        if (candidate.semantic_key.empty()) {
            throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                                "interactive candidate is missing a semantic key");
        }
        if (request.kind == DecisionRequestKind::UnselectCard &&
            candidate.action_kind == ActionKind::CardSelection) {
            if (candidate.card_selection_operation != CardSelectionOperation::Select &&
                candidate.card_selection_operation != CardSelectionOperation::Unselect) {
                throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                                    "unselect-card candidate has an invalid selection operation");
            }
        } else if (candidate.card_selection_operation != CardSelectionOperation::None) {
            throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                                "card selection operation is invalid for this candidate");
        }
        if (candidate.submits_engine_response && candidate.exact_response_bytes.empty()) {
            throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                                "terminal interactive candidate is missing an exact response");
        }
        if (!candidate.submits_engine_response && !candidate.exact_response_bytes.empty()) {
            throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                                "intermediate interactive candidate unexpectedly contains a response");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (candidate.semantic_key == request.candidates[j].semantic_key) {
                throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                                    "interactive request contains duplicate semantic keys");
            }
        }
    }
}

const ActionCandidate& select_candidate(const DecisionRequest& request, const std::string& semantic_key) {
    validate_candidate_set(request);
    const auto it = std::find_if(request.candidates.begin(), request.candidates.end(),
                                 [&semantic_key](const ActionCandidate& candidate) {
                                     return candidate.semantic_key == semantic_key;
                                 });
    if (it == request.candidates.end()) {
        throw ProtocolError(ProtocolErrorCode::InvalidSemanticKey,
                            "unknown or stale semantic action key: " + semantic_key);
    }
    return *it;
}

}  // namespace ygo::protocol
