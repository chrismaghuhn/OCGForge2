#include "ygo/protocol/continuation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <iterator>
#include <sstream>
#include <utility>
#include <vector>

#include "ygo/protocol/protocol_error.hpp"
#include "ygo/protocol/response_builder.hpp"

namespace ygo::protocol {
namespace {

bool contains_index(const std::vector<std::uint32_t>& indices, std::uint32_t value) {
    return std::find(indices.begin(), indices.end(), value) != indices.end();
}

std::uint32_t popcount64(std::uint64_t value) {
    std::uint32_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

void validate_items(const std::vector<ContinuationItem>& items) {
    for (std::size_t i = 0; i < items.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (items[i].source_index == items[j].source_index) {
                throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                                    "continuation contains duplicate source indices");
            }
        }
        if (i != 0 && items[i - 1].source_index >= items[i].source_index) {
            throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                                "continuation source indices are not in canonical order");
        }
    }
}

std::string state_id(const SelectionContinuation& continuation, std::uint64_t engine_step_index) {
    std::ostringstream id;
    id << "cont." << continuation.raw_message_hash << "." << continuation_kind_name(continuation.continuation_kind)
       << ".engine." << engine_step_index << ".step." << continuation.continuation_step << ".selected.";
    for (const auto index : continuation.selected_indices) {
        id << index << ",";
    }
    id << ".amounts.";
    for (const auto amount : continuation.assigned_amounts) {
        id << amount << ",";
    }
    id << ".mask." << continuation.selected_mask;
    return id.str();
}

std::uint32_t last_selected_index(const SelectionContinuation& continuation) {
    if (continuation.selected_indices.empty()) {
        return 0;
    }
    return continuation.selected_indices.back();
}

const ContinuationItem* find_item(const SelectionContinuation& continuation, std::uint32_t source_index) {
    const auto it = std::find_if(continuation.items.begin(), continuation.items.end(),
                                 [source_index](const ContinuationItem& item) {
                                     return item.source_index == source_index;
                                 });
    return it == continuation.items.end() ? nullptr : &*it;
}

void copy_card_fields(ActionCandidate& candidate, const ContinuationItem& item) {
    candidate.source_card = item.card.code;
    candidate.source_controller = item.card.controller;
    candidate.source_location = item.card.location;
    candidate.source_sequence = item.card.sequence;
    candidate.source_position = item.card.position;
    candidate.source_index = item.source_index;
}

std::vector<ContinuationItem> selected_items(const SelectionContinuation& continuation) {
    std::vector<ContinuationItem> result;
    result.reserve(continuation.selected_indices.size() + continuation.mandatory_items.size());
    result.insert(result.end(), continuation.mandatory_items.begin(), continuation.mandatory_items.end());
    for (const auto index : continuation.selected_indices) {
        const auto* item = find_item(continuation, index);
        if (item == nullptr) {
            throw ProtocolError(ProtocolErrorCode::InvalidSemanticKey,
                                "continuation references an unknown selected item");
        }
        result.push_back(*item);
    }
    return result;
}

std::uint32_t contribution_min(const ContinuationItem& item) {
    if (item.secondary_value == 0) {
        return item.primary_value;
    }
    return std::min(item.primary_value, item.secondary_value);
}

std::uint32_t contribution_max(const ContinuationItem& item) {
    return std::max(item.primary_value, item.secondary_value);
}

bool exact_sum_reachable(const std::vector<ContinuationItem>& items, std::size_t position,
                         std::int64_t remaining) {
    if (remaining <= 0 || position >= items.size()) {
        return false;
    }
    const auto& item = items[position];
    if (position == items.size() - 1) {
        return remaining == static_cast<std::int64_t>(item.primary_value) ||
               (item.secondary_value > 0 && remaining == static_cast<std::int64_t>(item.secondary_value));
    }
    if (remaining > static_cast<std::int64_t>(item.primary_value) &&
        exact_sum_reachable(items, position + 1,
                            remaining - static_cast<std::int64_t>(item.primary_value))) {
        return true;
    }
    return item.secondary_value > 0 && remaining > static_cast<std::int64_t>(item.secondary_value) &&
           exact_sum_reachable(items, position + 1,
                               remaining - static_cast<std::int64_t>(item.secondary_value));
}

bool greater_sum_legal(const SelectionContinuation& continuation) {
    const auto cards = selected_items(continuation);
    if (cards.empty()) {
        return false;
    }
    std::int64_t minimum_sum = 0;
    std::int64_t maximum_sum = 0;
    std::uint32_t minimum_value = std::numeric_limits<std::uint32_t>::max();
    for (const auto& item : cards) {
        minimum_sum += contribution_min(item);
        maximum_sum += contribution_max(item);
        minimum_value = std::min(minimum_value, contribution_min(item));
    }
    return maximum_sum >= continuation.target_sum && minimum_sum - minimum_value < continuation.target_sum;
}

bool sum_terminal(const SelectionContinuation& continuation) {
    if (continuation.greater_sum) {
        return greater_sum_legal(continuation);
    }
    const auto count = static_cast<std::uint32_t>(continuation.selected_indices.size());
    if (count < continuation.min_count || count > continuation.max_count) {
        return false;
    }
    const auto cards = selected_items(continuation);
    return !cards.empty() && exact_sum_reachable(cards, 0, continuation.target_sum);
}

bool sum_completion_search(const SelectionContinuation& continuation, std::vector<std::uint32_t>& selected,
                           const std::vector<ContinuationItem>& tail, std::size_t position) {
    auto candidate = continuation;
    candidate.selected_indices = selected;
    if (sum_terminal(candidate)) {
        return true;
    }
    if (!candidate.greater_sum && selected.size() >= candidate.max_count) {
        return false;
    }
    for (std::size_t index = position; index < tail.size(); ++index) {
        selected.push_back(tail[index].source_index);
        if (sum_completion_search(continuation, selected, tail, index + 1)) {
            return true;
        }
        selected.pop_back();
    }
    return false;
}

bool sum_can_complete_after(const SelectionContinuation& continuation, std::uint32_t picked_index) {
    std::vector<ContinuationItem> tail;
    for (const auto& item : continuation.items) {
        if (item.source_index > picked_index && !contains_index(continuation.selected_indices, item.source_index)) {
            tail.push_back(item);
        }
    }
    auto selected = continuation.selected_indices;
    selected.push_back(picked_index);
    return sum_completion_search(continuation, selected, tail, 0);
}

std::uint64_t selected_tribute_value(const SelectionContinuation& continuation) {
    std::uint64_t total = 0;
    for (const auto index : continuation.selected_indices) {
        const auto* item = find_item(continuation, index);
        if (item != nullptr) {
            total += item->primary_value;
        }
    }
    return total;
}

bool tribute_terminal(const SelectionContinuation& continuation) {
    const auto count = static_cast<std::uint32_t>(continuation.selected_indices.size());
    return count <= continuation.max_count && selected_tribute_value(continuation) >= continuation.required_amount;
}

bool tribute_completion_search(const SelectionContinuation& continuation, std::size_t position,
                               std::uint32_t selected_count, std::uint64_t selected_value) {
    if (selected_count > continuation.max_count) {
        return false;
    }
    if (selected_value >= continuation.required_amount) {
        return true;
    }
    if (position == continuation.items.size() || selected_count == continuation.max_count) {
        return false;
    }
    for (std::size_t index = position; index < continuation.items.size(); ++index) {
        if (contains_index(continuation.selected_indices, continuation.items[index].source_index)) {
            continue;
        }
        if (tribute_completion_search(continuation, index + 1, selected_count + 1,
                                       selected_value + continuation.items[index].primary_value)) {
            return true;
        }
    }
    return false;
}

bool tribute_can_complete_after(const SelectionContinuation& continuation, std::uint32_t picked_index) {
    const auto selected_count = static_cast<std::uint32_t>(continuation.selected_indices.size() + 1);
    if (selected_count > continuation.max_count) {
        return false;
    }
    const auto* picked = find_item(continuation, picked_index);
    if (picked == nullptr) {
        return false;
    }
    const auto picked_position = std::find_if(
        continuation.items.begin(), continuation.items.end(),
        [picked_index](const ContinuationItem& item) { return item.source_index == picked_index; });
    if (picked_position == continuation.items.end()) {
        return false;
    }
    return tribute_completion_search(continuation,
                                     static_cast<std::size_t>(std::distance(continuation.items.begin(), picked_position)) + 1,
                                     selected_count, selected_tribute_value(continuation) + picked->primary_value);
}

bool cardinality_can_complete_after(const SelectionContinuation& continuation, std::uint32_t picked_index) {
    const auto selected_count = static_cast<std::uint32_t>(continuation.selected_indices.size() + 1);
    if (selected_count > continuation.max_count) {
        return false;
    }
    std::uint32_t later_count = 0;
    for (const auto& item : continuation.items) {
        if (item.source_index > picked_index && !contains_index(continuation.selected_indices, item.source_index)) {
            ++later_count;
        }
    }
    return selected_count + later_count >= continuation.min_count;
}

bool cardinality_terminal(const SelectionContinuation& continuation) {
    const auto count = static_cast<std::uint32_t>(continuation.selected_indices.size());
    return count >= continuation.min_count && count <= continuation.max_count;
}

std::vector<std::uint8_t> terminal_response(const SelectionContinuation& continuation) {
    switch (continuation.continuation_kind) {
    case ContinuationKind::UnorderedSelection:
    case ContinuationKind::Tribute:
    case ContinuationKind::Sum:
        return encode_card_index_response(continuation.selected_indices);
    case ContinuationKind::ZonePlacement: {
        std::vector<ZoneSlot> zones;
        for (const auto index : continuation.selected_indices) {
            const auto* item = find_item(continuation, index);
            if (item == nullptr) {
                throw ProtocolError(ProtocolErrorCode::InvalidSemanticKey,
                                    "zone continuation references an unknown item");
            }
            zones.push_back({item->card.controller, static_cast<std::uint8_t>(item->card.location),
                             static_cast<std::uint8_t>(item->card.sequence)});
        }
        return encode_zone_response(zones);
    }
    case ContinuationKind::CounterAllocation:
        return encode_counter_response(continuation.assigned_amounts);
    case ContinuationKind::Ordering:
        return encode_order_response(continuation.selected_indices);
    case ContinuationKind::AnnouncementMask:
        return continuation.original_message_type == 140 ? encode_uint64_response(continuation.selected_mask)
                                                         : encode_uint32_response(
                                                               static_cast<std::uint32_t>(continuation.selected_mask));
    }
    return {};
}

ActionCandidate make_pick_candidate(const SelectionContinuation& continuation, const ContinuationItem& item,
                                    ActionKind kind = ActionKind::Pick) {
    ActionCandidate candidate;
    candidate.action_kind = kind;
    candidate.semantic_key = continuation.continuation_id + ".pick." + std::to_string(item.source_index);
    copy_card_fields(candidate, item);
    candidate.choice_index = item.source_index;
    candidate.continuation_id = continuation.continuation_id;
    candidate.continuation_operation = kind == ActionKind::AssignAmount
                                           ? ContinuationOperation::AssignAmount
                                           : ContinuationOperation::Pick;
    candidate.submits_engine_response = false;
    return candidate;
}

void add_finish_and_cancel(DecisionRequest& request) {
    auto& continuation = *request.continuation;
    if (continuation.can_finish) {
        ActionCandidate finish;
        finish.action_kind = ActionKind::Finish;
        finish.semantic_key = continuation.continuation_id + ".finish";
        finish.continuation_id = continuation.continuation_id;
        finish.continuation_operation = ContinuationOperation::Finish;
        finish.submits_engine_response = true;
        finish.exact_response_bytes = terminal_response(continuation);
        request.candidates.push_back(std::move(finish));
    }
    if (continuation.can_cancel) {
        ActionCandidate cancel;
        cancel.action_kind = ActionKind::Cancel;
        cancel.semantic_key = continuation.continuation_id + ".cancel";
        cancel.continuation_id = continuation.continuation_id;
        cancel.continuation_operation = ContinuationOperation::Cancel;
        cancel.submits_engine_response = true;
        cancel.exact_response_bytes = encode_int32_response(-1);
        request.candidates.push_back(std::move(cancel));
    }
}

void add_monotonic_candidates(DecisionRequest& request,
                              bool (*completion)(const SelectionContinuation&, std::uint32_t)) {
    auto& continuation = *request.continuation;
    const auto last = last_selected_index(continuation);
    for (const auto& item : continuation.items) {
        if (contains_index(continuation.selected_indices, item.source_index) ||
            (!continuation.selected_indices.empty() && item.source_index <= last) ||
            !completion(continuation, item.source_index)) {
            continue;
        }
        request.candidates.push_back(make_pick_candidate(continuation, item));
    }
    add_finish_and_cancel(request);
}

void add_unordered_candidates(DecisionRequest& request) {
    auto& continuation = *request.continuation;
    continuation.can_finish = cardinality_terminal(continuation);
    add_monotonic_candidates(request, cardinality_can_complete_after);
}

void add_tribute_candidates(DecisionRequest& request) {
    auto& continuation = *request.continuation;
    continuation.can_finish = tribute_terminal(continuation);
    add_monotonic_candidates(request, tribute_can_complete_after);
}

void add_sum_candidates(DecisionRequest& request) {
    auto& continuation = *request.continuation;
    continuation.can_finish = sum_terminal(continuation);
    add_monotonic_candidates(request, sum_can_complete_after);
}

void add_zone_candidates(DecisionRequest& request) {
    auto& continuation = *request.continuation;
    continuation.can_finish = cardinality_terminal(continuation);
    add_monotonic_candidates(request, cardinality_can_complete_after);
}

void add_ordering_candidates(DecisionRequest& request) {
    auto& continuation = *request.continuation;
    continuation.remaining_indices.clear();
    for (const auto& item : continuation.items) {
        if (!contains_index(continuation.selected_indices, item.source_index)) {
            continuation.remaining_indices.push_back(item.source_index);
            request.candidates.push_back(make_pick_candidate(continuation, item));
        }
    }
    ActionCandidate bypass;
    bypass.action_kind = ActionKind::Cancel;
    bypass.semantic_key = continuation.continuation_id + ".bypass";
    bypass.continuation_id = continuation.continuation_id;
    bypass.continuation_operation = ContinuationOperation::Bypass;
    bypass.exact_response_bytes = encode_order_bypass_response();
    request.candidates.push_back(std::move(bypass));
    continuation.can_finish = continuation.remaining_indices.empty();
    add_finish_and_cancel(request);
}

void add_counter_candidates(DecisionRequest& request) {
    auto& continuation = *request.continuation;
    const auto assigned = continuation.assigned_amounts.size();
    continuation.remaining_indices.clear();
    for (std::size_t index = assigned; index < continuation.items.size(); ++index) {
        continuation.remaining_indices.push_back(continuation.items[index].source_index);
    }
    const auto used = std::accumulate(continuation.assigned_amounts.begin(), continuation.assigned_amounts.end(), 0u);
    if (assigned == continuation.items.size()) {
        continuation.can_finish = used == continuation.required_amount;
        add_finish_and_cancel(request);
        return;
    }
    continuation.can_finish = false;
    const auto& item = continuation.items[assigned];
    std::uint32_t future_capacity = 0;
    for (std::size_t index = assigned + 1; index < continuation.items.size(); ++index) {
        future_capacity += continuation.items[index].capacity;
    }
    if (used > continuation.required_amount) {
        return;
    }
    const auto remaining = continuation.required_amount - used;
    const auto upper = std::min(item.capacity, remaining);
    for (std::uint32_t amount = 0; amount <= upper; ++amount) {
        if (remaining - amount > future_capacity) {
            continue;
        }
        ActionCandidate candidate = make_pick_candidate(continuation, item, ActionKind::AssignAmount);
        candidate.amount = static_cast<std::int32_t>(amount);
        candidate.choice_value = amount;
        candidate.semantic_key = continuation.continuation_id + ".amount." + std::to_string(item.source_index) + "." +
                                 std::to_string(amount);
        request.candidates.push_back(std::move(candidate));
    }
    add_finish_and_cancel(request);
}

void add_mask_candidates(DecisionRequest& request) {
    auto& continuation = *request.continuation;
    const auto selected_count = popcount64(continuation.selected_mask);
    continuation.can_finish = selected_count == continuation.min_count;
    const auto last = last_selected_index(continuation);
    for (const auto& item : continuation.items) {
        if (selected_count >= continuation.min_count || contains_index(continuation.selected_indices, item.source_index) ||
            (!continuation.selected_indices.empty() && item.source_index <= last) ||
            !cardinality_can_complete_after(continuation, item.source_index)) {
            continue;
        }
        request.candidates.push_back(make_pick_candidate(continuation, item));
    }
    add_finish_and_cancel(request);
}

void add_candidates(DecisionRequest& request) {
    auto& continuation = *request.continuation;
    continuation.remaining_indices.clear();
    for (const auto& item : continuation.items) {
        if (!contains_index(continuation.selected_indices, item.source_index)) {
            continuation.remaining_indices.push_back(item.source_index);
        }
    }
    switch (continuation.continuation_kind) {
    case ContinuationKind::UnorderedSelection:
        add_unordered_candidates(request);
        break;
    case ContinuationKind::Tribute:
        add_tribute_candidates(request);
        break;
    case ContinuationKind::Sum:
        add_sum_candidates(request);
        break;
    case ContinuationKind::ZonePlacement:
        add_zone_candidates(request);
        break;
    case ContinuationKind::CounterAllocation:
        add_counter_candidates(request);
        break;
    case ContinuationKind::Ordering:
        add_ordering_candidates(request);
        break;
    case ContinuationKind::AnnouncementMask:
        add_mask_candidates(request);
        break;
    }
}

void validate_generated_candidates(const DecisionRequest& request) {
    for (std::size_t index = 0; index < request.candidates.size(); ++index) {
        const auto& candidate = request.candidates[index];
        if (candidate.semantic_key.empty() ||
            (candidate.submits_engine_response && candidate.exact_response_bytes.empty()) ||
            (!candidate.submits_engine_response && !candidate.exact_response_bytes.empty())) {
            throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                                "generated continuation candidate is incomplete");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (candidate.semantic_key == request.candidates[previous].semantic_key) {
                throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                                    "generated continuation candidate keys are not unique");
            }
        }
    }
    if (request.candidates.empty()) {
        throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                            "continuation has no reachable action");
    }
}

}  // namespace

std::string continuation_kind_name(ContinuationKind kind) {
    switch (kind) {
    case ContinuationKind::UnorderedSelection:
        return "unordered";
    case ContinuationKind::Tribute:
        return "tribute";
    case ContinuationKind::Sum:
        return "sum";
    case ContinuationKind::ZonePlacement:
        return "zone";
    case ContinuationKind::CounterAllocation:
        return "counter";
    case ContinuationKind::Ordering:
        return "ordering";
    case ContinuationKind::AnnouncementMask:
        return "announce_mask";
    }
    return "unknown";
}

std::string_view continuation_operation_name(const ContinuationOperation operation) noexcept {
    switch (operation) {
    case ContinuationOperation::None:
        return {};
    case ContinuationOperation::Pick:
        return "pick";
    case ContinuationOperation::AssignAmount:
        // Preserve the established public descriptor spelling while keeping
        // the owning source metadata typed.
        return "amount";
    case ContinuationOperation::Finish:
        return "finish";
    case ContinuationOperation::Cancel:
        return "cancel";
    case ContinuationOperation::Bypass:
        return "bypass";
    }
    return {};
}

DecisionRequest make_continuation_request(DecisionRequestKind request_kind, std::uint8_t player,
                                          std::uint8_t engine_message_type, std::string engine_message_name,
                                          std::uint64_t engine_step_index, SelectionContinuation continuation) {
    validate_items(continuation.items);
    continuation.continuation_id = state_id(continuation, engine_step_index);
    DecisionRequest request;
    request.kind = request_kind;
    request.decision_id = continuation.continuation_id + ".decision";
    request.engine_step_index = engine_step_index;
    request.player = player;
    request.engine_message_type = engine_message_type;
    request.engine_message_name = std::move(engine_message_name);
    request.raw_message_hash = continuation.raw_message_hash;
    request.continuation = std::move(continuation);
    add_candidates(request);
    auto& metrics = *request.continuation;
    metrics.continuation_steps = metrics.continuation_step;
    metrics.peak_candidate_count = std::max(metrics.peak_candidate_count, request.candidates.size());
    metrics.terminal_solution_count = 0;
    for (const auto& candidate : request.candidates) {
        if (candidate.action_kind == ActionKind::Finish || candidate.action_kind == ActionKind::Cancel) {
            ++metrics.terminal_solution_count;
        }
    }
    validate_generated_candidates(request);
    return request;
}

DecisionRequest begin_unordered_continuation(std::uint8_t player, std::uint8_t engine_message_type,
                                             std::string engine_message_name, std::string raw_message_hash,
                                             std::vector<ContinuationItem> items, std::uint32_t min_count,
                                             std::uint32_t max_count, bool can_cancel) {
    validate_items(items);
    if (items.empty() || min_count > max_count || max_count > items.size()) {
        throw ProtocolError(ProtocolErrorCode::IncompleteCandidates,
                            "unordered continuation has invalid cardinality constraints");
    }
    SelectionContinuation continuation;
    continuation.continuation_kind = ContinuationKind::UnorderedSelection;
    continuation.original_message_type = engine_message_type;
    continuation.raw_message_hash = std::move(raw_message_hash);
    continuation.items = std::move(items);
    continuation.min_count = min_count;
    continuation.max_count = max_count;
    continuation.can_cancel = can_cancel;
    return make_continuation_request(DecisionRequestKind::CardSelection, player, engine_message_type,
                                     std::move(engine_message_name), 0, std::move(continuation));
}

ContinuationTransition apply_continuation_action(const DecisionRequest& request, const std::string& semantic_key) {
    if (!request.continuation.has_value()) {
        throw ProtocolError(ProtocolErrorCode::InvalidSemanticKey,
                            "decision request does not have an active continuation");
    }
    const auto& continuation = *request.continuation;
    const auto candidate_it = std::find_if(request.candidates.begin(), request.candidates.end(),
                                           [&semantic_key](const ActionCandidate& candidate) {
                                               return candidate.semantic_key == semantic_key;
                                           });
    if (candidate_it == request.candidates.end() || candidate_it->continuation_id != continuation.continuation_id) {
        throw ProtocolError(ProtocolErrorCode::InvalidSemanticKey,
                            "unknown or stale continuation semantic key: " + semantic_key);
    }

    if (candidate_it->action_kind == ActionKind::Pick || candidate_it->action_kind == ActionKind::AssignAmount) {
        auto next = continuation;
        switch (continuation.continuation_kind) {
        case ContinuationKind::CounterAllocation:
            {
            const auto assigned = next.assigned_amounts.size();
            const auto used = std::accumulate(next.assigned_amounts.begin(), next.assigned_amounts.end(), 0u);
            std::uint32_t future_capacity = 0;
            for (std::size_t index = assigned + 1; index < next.items.size(); ++index) {
                future_capacity += next.items[index].capacity;
            }
            if (candidate_it->action_kind != ActionKind::AssignAmount ||
                assigned >= next.items.size() || next.items[assigned].source_index != candidate_it->source_index ||
                candidate_it->amount < 0 || used > next.required_amount ||
                static_cast<std::uint32_t>(candidate_it->amount) > next.items[assigned].capacity ||
                static_cast<std::uint32_t>(candidate_it->amount) > next.required_amount - used ||
                next.required_amount - used - static_cast<std::uint32_t>(candidate_it->amount) > future_capacity) {
                throw ProtocolError(ProtocolErrorCode::InvalidSemanticKey,
                                    "counter allocation action does not match the current counter item");
            }
            next.assigned_amounts.push_back(static_cast<std::uint16_t>(candidate_it->amount));
            }
            break;
        case ContinuationKind::AnnouncementMask: {
            const auto* item = find_item(next, candidate_it->source_index);
            if (item == nullptr || contains_index(next.selected_indices, candidate_it->source_index) ||
                (!next.selected_indices.empty() &&
                 candidate_it->source_index <= next.selected_indices.back())) {
                throw ProtocolError(ProtocolErrorCode::InvalidSemanticKey,
                                    "announcement action does not match the canonical remaining bit domain");
            }
            next.selected_indices.push_back(candidate_it->source_index);
            next.selected_mask |= item->mask_value;
            break;
        }
        case ContinuationKind::Ordering:
            if (candidate_it->action_kind != ActionKind::Pick ||
                find_item(next, candidate_it->source_index) == nullptr ||
                contains_index(next.selected_indices, candidate_it->source_index)) {
                throw ProtocolError(ProtocolErrorCode::InvalidSemanticKey,
                                    "ordering pick does not match the remaining permutation domain");
            }
            next.selected_indices.push_back(candidate_it->source_index);
            break;
        default:
            if (candidate_it->action_kind != ActionKind::Pick ||
                find_item(next, candidate_it->source_index) == nullptr ||
                contains_index(next.selected_indices, candidate_it->source_index) ||
                (!next.selected_indices.empty() &&
                 candidate_it->source_index <= next.selected_indices.back())) {
                throw ProtocolError(ProtocolErrorCode::InvalidSemanticKey,
                                    "continuation pick does not match the canonical remaining domain");
            }
            next.selected_indices.push_back(candidate_it->source_index);
            break;
        }
        ++next.continuation_step;
        auto next_request = make_continuation_request(request.kind, request.player, request.engine_message_type,
                                                      request.engine_message_name, request.engine_step_index,
                                                      std::move(next));
        return {std::move(next_request), false, false, {}};
    }

    if (candidate_it->action_kind != ActionKind::Finish && candidate_it->action_kind != ActionKind::Cancel) {
        throw ProtocolError(ProtocolErrorCode::InvalidSemanticKey,
                            "continuation action kind is not supported by this transition");
    }
    return {request, true, true, candidate_it->exact_response_bytes};
}

}  // namespace ygo::protocol
