#include "ygo/client/client_duel.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "ocgapi_constants.h"
#include "card_data_reader.hpp"
#include "client_view.hpp"
#include "decoded_queries.hpp"
#include "ygo/observation/decision_integration.hpp"
#include "ygo/observation/observation_session.hpp"
#include "ygo/protocol/continuation.hpp"
#include "ygo/protocol/message_decoder.hpp"
#include "ygo/protocol/protocol_error.hpp"
#include "ygo/util/sha256.hpp"

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

// Public events copied into each observation (at least; always back to the turn start), as in Duel.
constexpr std::size_t kEventTail = 512;

ClientDuel::ClientDuel(std::uint8_t player, std::uint64_t duel_flags, const std::filesystem::path& card_data_tsv,
                       std::vector<std::uint32_t> main_deck)
    : player_(player), duel_flags_(duel_flags != 0 ? duel_flags : DUEL_MODE_MR5), main_deck_(std::move(main_deck)),
      session_(std::make_unique<observation::ObservationSession>(player, static_cast<std::uint32_t>(duel_flags_))) {
    if (player > 1) {
        throw std::invalid_argument("ClientDuel player must be 0 or 1");
    }
    if (!card_data_tsv.empty()) {
        view_ = std::make_unique<ClientView>(static_cast<std::uint32_t>(duel_flags_),
                                             core::detail::CardDataStore::shared(card_data_tsv));
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
    if (view_) {
        try {
            view_->apply(message);
        } catch (const std::exception& error) {
            skipped_.push_back(type + ": field: " + error.what());
        }
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
    state_.clear();
    if (view_ && !request.continuation && request.candidates.size() > 1) {
        auto state = view_->digest();
        state.insert(state.end(), message.begin(), message.end());
        state_ = util::sha256_bytes(state);
        if (const auto answers = answers_.find(state_); answers != answers_.end()) {
            auto open = request.candidates;
            std::erase_if(open, [&](const protocol::ActionCandidate& candidate) {
                const auto given = answers->second.find(candidate.semantic_key);
                return given != answers->second.end() && given->second >= block_after_;
            });
            if (!open.empty()) {
                request.candidates = std::move(open);
            }
        }
    }
    request_ = std::move(request);
    observe();
    return true;
}

void ClientDuel::observe() {
    observation_.reset();
    if (!view_ || !request_) {
        return;
    }
    observation::ObservationBuildConfig build;
    build.decision_index = decisions_;
    build.engine_step_index = request_->engine_step_index;
    const auto& events = session_->visible_events();
    auto first = events.size() <= kEventTail ? 0 : events.size() - kEventTail;
    first = std::min(first, session_->turn_start());
    build.visible_events.assign(events.begin() + static_cast<std::ptrdiff_t>(first), events.end());
    build.event_summary = session_->summary();
    build.detail = observation::ObservationDetail::ModelInput;
    build.knowledge.own_decklist_known = true;
    build.own_deck.known = true;
    build.own_deck.main_deck = main_deck_;
    observation_ = observation::detail::build_player_observation(*view_, player_, std::move(build));
    observation::attach_decision_context(*observation_, *request_);
}

std::optional<std::vector<std::uint8_t>> ClientDuel::act(const std::string& semantic_key) {
    const auto& candidate = protocol::select_candidate(request(), semantic_key);
    ++decisions_;
    if (!state_.empty() && ++answers_[state_][candidate.semantic_key] == block_after_) {
        ++blocked_answers_;
    }
    state_.clear();
    if (request_->continuation) {
        auto transition = protocol::apply_continuation_action(*request_, candidate.semantic_key);
        if (!transition.terminal) {
            request_ = std::move(transition.request);
            observe();
            return std::nullopt;
        }
        request_.reset();
        observation_.reset();
        return std::move(transition.engine_response);
    }
    auto response = candidate.exact_response_bytes;
    request_.reset();
    observation_.reset();
    return response;
}

}  // namespace ygo::client
