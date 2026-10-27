#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ygo/protocol/decision_request.hpp"

namespace ygo::protocol {

struct DecodedMessage {
    bool interactive = false;
    bool terminal = false;
    bool retry = false;
    std::uint8_t message_type = 0;
    std::uint8_t winner = 255;
    std::uint8_t win_reason = 255;
    std::vector<DecisionRequest> decisions;
};

DecodedMessage decode_messages(const std::vector<std::uint8_t>& bytes, std::uint64_t engine_step_index = 0);
std::string action_kind_name(ActionKind kind);
std::string decision_kind_name(DecisionRequestKind kind);
void validate_candidate_set(const DecisionRequest& request);
const ActionCandidate& select_candidate(const DecisionRequest& request, const std::string& semantic_key);

}  // namespace ygo::protocol
