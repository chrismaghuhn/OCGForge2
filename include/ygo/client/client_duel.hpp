#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ygo/protocol/decision_request.hpp"

namespace ygo::observation {
class ObservationSession;
}

namespace ygo::client {

// One player's side of a duel run by someone else's engine (an EDOPro server): fed the game
// messages that player receives, it turns each question into a DecisionRequest and an answer into
// the response bytes to send back. No engine and no rules of its own; the server stays the authority.
//
// Messages it cannot read are recorded in skipped() and ignored, never fatal: a server may send
// messages (or variants) the training engine never produced. Only a question it cannot decode is
// an error, reported by feed().
class ClientDuel final {
public:
    explicit ClientDuel(std::uint8_t player);
    ~ClientDuel();

    // One game message (type byte + payload, as in STOC_GAME_MSG). Returns true when it asks this
    // player for a decision; request() then holds it.
    bool feed(const std::vector<std::uint8_t>& message);

    bool pending() const noexcept { return request_.has_value(); }
    const protocol::DecisionRequest& request() const;

    // Answers the pending question. Returns the response to send, or nothing when the answer was one
    // step of a multi-step selection and request() holds the next step.
    std::optional<std::vector<std::uint8_t>> act(const std::string& semantic_key);

    bool finished() const noexcept { return winner_.has_value(); }
    std::optional<std::uint8_t> winner() const noexcept { return winner_; }
    std::uint64_t messages() const noexcept { return messages_; }
    // "type: reason" for every message that could not be read.
    const std::vector<std::string>& skipped() const noexcept { return skipped_; }

private:
    std::uint8_t player_;
    std::uint64_t messages_ = 0;
    std::unique_ptr<observation::ObservationSession> session_;
    std::optional<protocol::DecisionRequest> request_;
    std::optional<std::uint8_t> winner_;
    std::vector<std::string> skipped_;
};

}  // namespace ygo::client
