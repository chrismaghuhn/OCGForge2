#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ygo/observation/player_observation.hpp"
#include "ygo/protocol/decision_request.hpp"

namespace ygo::encode {

// Model input for one decision. Field names, order and vocabularies must
// match python/src/ocgforge/features.py; tests/test_core.py compares them
// through the *_names() functions below.
inline constexpr std::size_t kMaxTargets = 4;

struct EncodedDecision {
    std::uint8_t player = 0;
    std::vector<std::int64_t> token_card;
    std::vector<std::int64_t> token_zone;
    std::vector<std::int64_t> token_side;
    std::vector<float> token_numeric;  // [tokens x token_numeric_names().size()]
    std::vector<float> global_numeric;
    std::int64_t phase = 0;
    std::int64_t decision = 0;
    std::vector<std::int64_t> candidate_kind;
    std::vector<std::int64_t> candidate_source;   // token index or -1
    std::vector<std::int64_t> candidate_card;     // card index when there is no source token (own cards only), else 0
    std::vector<std::int64_t> candidate_zone;     // zone of the candidate's card or place, 0 = none
    std::vector<std::int64_t> candidate_targets;  // [candidates x kMaxTargets], token index or -1
    std::vector<float> candidate_numeric;         // [candidates x candidate_numeric_names().size()]
};

// Passcode (including alternate artworks) -> model card index; anything else
// is encoded as unknown (1). Comes from the model checkpoint's cards.json.
using CardIndex = std::unordered_map<std::uint32_t, std::int64_t>;

// Encodes the decision from the deciding player's observation. Throws
// std::invalid_argument for a request the model cannot represent.
EncodedDecision encode(const observation::PlayerObservation& observation, const protocol::DecisionRequest& request,
                       const CardIndex& card_index);

const std::vector<std::string>& decision_names();
const std::vector<std::string>& zone_names();
const std::vector<std::string>& phase_names();
const std::vector<std::string>& candidate_kind_names();
const std::vector<std::string>& token_numeric_names();
const std::vector<std::string>& global_numeric_names();
const std::vector<std::string>& candidate_numeric_names();

}  // namespace ygo::encode
