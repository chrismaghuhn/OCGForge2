#pragma once

#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "ygo/core/deck.hpp"
#include "ygo/observation/player_observation.hpp"
#include "ygo/protocol/decision_request.hpp"

namespace ygo::core {
class CoreHost;
}
namespace ygo::observation {
class ObservationSession;
}

namespace ygo::duel {

struct DuelConfig {
    std::filesystem::path card_scripts_root;
    std::filesystem::path card_data_tsv;
    core::Deck deck0;
    core::Deck deck1;
    std::uint64_t seed = 0;
    std::uint8_t starting_player = 0;
    bool build_observations = true;
    // Model-input observations query only the card properties the encoder uses and carry only
    // the recent events; full ones have every card property and the whole event history.
    bool full_observations = false;

    // Loop protection: a duel that needs more decisions than this, or that
    // reaches the exact same engine question in the exact same field state
    // more than max_repeats times, is treated as stuck and throws DuelError.
    std::uint64_t max_decisions = 5000;
    std::uint32_t max_repeats = 16;
};

struct DuelResult {
    std::uint8_t winner = 255;
    std::uint8_t reason = 255;
};

// Thrown for engine/protocol failures and for stuck duels. what() includes the
// most recent decisions so a loop is visible directly in the test output.
class DuelError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Drives one duel. Usage:
//
//     Duel duel(config);
//     while (!duel.finished()) {
//         duel.act(choose(duel.request()));
//     }
//     auto result = *duel.result();
class Duel final {
public:
    explicit Duel(DuelConfig config);
    ~Duel();

    Duel(const Duel&) = delete;
    Duel& operator=(const Duel&) = delete;

    bool finished() const noexcept { return result_.has_value(); }
    const std::optional<DuelResult>& result() const noexcept { return result_; }

    // The pending decision. Only valid while !finished().
    const protocol::DecisionRequest& request() const;
    // The deciding player's view, or nullptr if observations are disabled.
    const observation::PlayerObservation* observation() const noexcept;

    // Applies one candidate (by semantic key) and runs the engine until the
    // next decision or the end of the duel.
    void act(const std::string& semantic_key);

    std::uint64_t decision_count() const noexcept { return decision_count_; }

private:
    void advance();
    void publish(protocol::DecisionRequest request);
    void submit(std::vector<std::uint8_t> response);
    [[noreturn]] void fail(const std::string& message) const;

    DuelConfig config_;
    std::unique_ptr<core::CoreHost> host_;
    std::unique_ptr<observation::ObservationSession> sessions_[2];
    std::uint64_t engine_step_ = 0;
    std::uint64_t decision_count_ = 0;
    std::optional<protocol::DecisionRequest> request_;
    std::optional<observation::PlayerObservation> observation_;
    std::optional<DuelResult> result_;
    std::map<std::string, std::uint32_t> seen_states_;
    std::deque<std::string> recent_;
};

}  // namespace ygo::duel
