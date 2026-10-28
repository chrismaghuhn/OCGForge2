#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <filesystem>

#include "ygo/observation/player_observation.hpp"
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
class ClientView;

class ClientDuel final {
public:
    // Without card_data_tsv no observations are built (request() and act() still work).
    // duel_flags: the room's rules (DUEL_MODE_MR5 for Master Rule 2020); main_deck: our own deck.
    explicit ClientDuel(std::uint8_t player, std::uint64_t duel_flags = 0,
                        const std::filesystem::path& card_data_tsv = {},
                        std::vector<std::uint32_t> main_deck = {});
    ~ClientDuel();

    // One game message (type byte + payload, as in STOC_GAME_MSG). Returns true when it asks this
    // player for a decision; request() then holds it.
    bool feed(const std::vector<std::uint8_t>& message);

    bool pending() const noexcept { return request_.has_value(); }
    const protocol::DecisionRequest& request() const;
    // The pending question as this player sees the duel, built like the training engine builds it.
    const observation::PlayerObservation* observation() const noexcept { return observation_ ? &*observation_ : nullptr; }

    // Answers the pending question. Returns the response to send, or nothing when the answer was one
    // step of a multi-step selection and request() holds the next step.
    std::optional<std::vector<std::uint8_t>> act(const std::string& semantic_key);

    bool finished() const noexcept { return winner_.has_value(); }
    std::optional<std::uint8_t> winner() const noexcept { return winner_; }
    std::uint64_t messages() const noexcept { return messages_; }
    // "type: reason" for every message that could not be read.
    const std::vector<std::string>& skipped() const noexcept { return skipped_; }

private:
    void observe();

    std::uint8_t player_;
    std::uint64_t duel_flags_;
    std::vector<std::uint32_t> main_deck_;
    std::unique_ptr<ClientView> view_;
    std::optional<observation::PlayerObservation> observation_;
    std::uint64_t messages_ = 0;
    std::uint64_t decisions_ = 0;
    std::unique_ptr<observation::ObservationSession> session_;
    std::optional<protocol::DecisionRequest> request_;
    std::optional<std::uint8_t> winner_;
    std::vector<std::string> skipped_;
};

}  // namespace ygo::client
