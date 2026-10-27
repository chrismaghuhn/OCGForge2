#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ygo/protocol/action_candidate.hpp"

namespace ygo::protocol {

struct CardLocator {
    std::uint32_t code = 0;
    std::uint8_t controller = 0;
    std::uint32_t location = 0;
    std::uint32_t sequence = 0;
    std::uint32_t position = 0;
};

struct ContinuationItem {
    CardLocator card;
    std::uint32_t source_index = 0;
    std::uint32_t primary_value = 0;
    std::uint32_t secondary_value = 0;
    std::uint32_t capacity = 0;
    std::uint64_t mask_value = 0;
};

enum class ContinuationKind {
    UnorderedSelection,
    Tribute,
    Sum,
    ZonePlacement,
    CounterAllocation,
    Ordering,
    AnnouncementMask,
};

struct SelectionContinuation {
    std::string continuation_id;
    ContinuationKind continuation_kind = ContinuationKind::UnorderedSelection;
    std::uint32_t continuation_step = 0;
    std::uint8_t original_message_type = 0;
    std::string raw_message_hash;
    std::vector<ContinuationItem> items;
    std::vector<ContinuationItem> mandatory_items;
    std::vector<std::uint32_t> selected_indices;
    std::vector<std::uint32_t> remaining_indices;
    std::vector<std::uint16_t> assigned_amounts;
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
    std::uint32_t target_sum = 0;
    std::uint32_t required_amount = 0;
    std::uint64_t available_mask = 0;
    std::uint64_t selected_mask = 0;
    std::uint32_t continuation_steps = 0;
    std::size_t peak_candidate_count = 0;
    std::uint64_t terminal_solution_count = 0;
    bool exact_sum = true;
    bool greater_sum = false;
    bool can_finish = false;
    bool can_cancel = false;
};

enum class DecisionRequestKind {
    IdleCommand,
    BattleCommand,
    Chain,
    Option,
    CardSelection,
    Tribute,
    Sum,
    Place,
    Counter,
    Ordering,
    Announcement,
    UnselectCard,
    Position,
    YesNo,
    Unsupported,
};

struct DecisionRequest {
    DecisionRequestKind kind = DecisionRequestKind::Unsupported;
    std::string decision_id;
    std::uint64_t engine_step_index = 0;
    std::uint8_t player = 0;
    std::uint8_t engine_message_type = 0;
    std::string engine_message_name;
    std::string raw_message_hash;
    std::vector<ActionCandidate> candidates;
    std::optional<SelectionContinuation> continuation;
};

DecisionRequest make_continuation_request(DecisionRequestKind request_kind, std::uint8_t player,
                                          std::uint8_t engine_message_type, std::string engine_message_name,
                                          std::uint64_t engine_step_index, SelectionContinuation continuation);

struct ContinuationTransition {
    DecisionRequest request;
    bool terminal = false;
    bool engine_advanced = false;
    std::vector<std::uint8_t> engine_response;
};

std::string continuation_kind_name(ContinuationKind kind);
std::string_view continuation_operation_name(ContinuationOperation operation) noexcept;

DecisionRequest begin_unordered_continuation(std::uint8_t player, std::uint8_t engine_message_type,
                                             std::string engine_message_name, std::string raw_message_hash,
                                             std::vector<ContinuationItem> items, std::uint32_t min_count,
                                             std::uint32_t max_count, bool can_cancel);

ContinuationTransition apply_continuation_action(const DecisionRequest& request,
                                                 const std::string& semantic_key);

}  // namespace ygo::protocol
