#include "ygo/client/client_duel.hpp"

#include <stdexcept>
#include <utility>

#include "ocgapi_constants.h"
#include "ygo/observation/observation_session.hpp"
#include "ygo/protocol/continuation.hpp"
#include "ygo/protocol/message_decoder.hpp"
#include "ygo/protocol/protocol_error.hpp"

namespace ygo::client {
namespace {

// decode_messages and ObservationSession read the engine's framing: u32 length, then the message.
std::vector<std::uint8_t> frame(const std::vector<std::uint8_t>& message) {
    const auto size = static_cast<std::uint32_t>(message.size());
    std::vector<std::uint8_t> framed = {static_cast<std::uint8_t>(size), static_cast<std::uint8_t>(size >> 8),
                                        static_cast<std::uint8_t>(size >> 16), static_cast<std::uint8_t>(size >> 24)};
    framed.insert(framed.end(), message.begin(), message.end());
    return framed;
}

}  // namespace

ClientDuel::ClientDuel(std::uint8_t player)
    : player_(player), session_(std::make_unique<observation::ObservationSession>(player, DUEL_MODE_MR5)) {
    if (player > 1) {
        throw std::invalid_argument("ClientDuel player must be 0 or 1");
    }
}

ClientDuel::~ClientDuel() = default;

const protocol::DecisionRequest& ClientDuel::request() const {
    if (!request_) {
        throw std::logic_error("ClientDuel::request() without a pending question");
    }
    return *request_;
}

bool ClientDuel::feed(const std::vector<std::uint8_t>& message) {
    if (message.empty()) {
        return false;
    }
    const auto step = messages_++;
    const auto framed = frame(message);
    const auto type = std::to_string(message.front());
    try {
        session_->ingest(framed, step);
    } catch (const std::exception& error) {
        skipped_.push_back(type + ": observation: " + error.what());
    }
    protocol::DecodedMessage decoded;
    try {
        decoded = protocol::decode_messages(framed, step);
    } catch (const protocol::ProtocolError& error) {
        if (error.code() == protocol::ProtocolErrorCode::UnsupportedDecision && error.player() == player_ &&
            std::string(error.what()).find("unknown-engine-message") == std::string::npos) {
            throw;  // a question for us that we cannot answer
        }
        skipped_.push_back(type + ": " + error.what());
        return false;
    }
    if (decoded.terminal) {
        winner_ = decoded.winner;
    }
    if (!decoded.interactive) {
        return false;
    }
    auto& request = decoded.decisions.front();
    if (request.player != player_) {
        skipped_.push_back(type + ": question for the other player");
        return false;
    }
    request_ = std::move(request);
    return true;
}

std::optional<std::vector<std::uint8_t>> ClientDuel::act(const std::string& semantic_key) {
    const auto& candidate = protocol::select_candidate(request(), semantic_key);
    if (request_->continuation) {
        auto transition = protocol::apply_continuation_action(*request_, candidate.semantic_key);
        if (!transition.terminal) {
            request_ = std::move(transition.request);
            return std::nullopt;
        }
        request_.reset();
        return std::move(transition.engine_response);
    }
    auto response = candidate.exact_response_bytes;
    request_.reset();
    return response;
}

}  // namespace ygo::client
