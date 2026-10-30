#pragma once

#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

    // Loop protection. A duel that needs more than max_decisions decisions throws DuelError.
    // An answer given block_after times to the same engine question in the same field state is
    // removed from that question's candidates. When that leaves no candidate, or the question
    // comes back more than max_repeats times, the asking player loses (reason kLoopLoss).
    // Questions with a single candidate are not counted; multi-step (continuation) questions only
    // count towards max_repeats. block_after = 0 turns blocking off: visit max_repeats + 1 then
    // throws a "stuck" DuelError (the old behaviour also had max_repeats = 16).
    // When set, the decks are not loaded: this Lua script (a puzzle: Debug.ReloadFieldBegin, AddCard,
    // ReloadFieldEnd) builds the whole field before the duel starts, e.g. a sampled world for search.
    // deck0/deck1 then only list the cards whose scripts to preload.
    std::string fixture_script;
    std::uint64_t max_decisions = 5000;
    std::uint32_t block_after = 8;
    std::uint32_t max_repeats = 64;
};

// DuelResult::reason of a duel lost by looping (see DuelConfig::block_after).
constexpr std::uint8_t kLoopLoss = 0xF0;

// Every card of a duel as the engine knows it (hidden ones included): what search needs to build
// worlds from, after hiding what the searching player cannot see.
struct CardState {
    std::uint8_t controller = 0;
    std::uint8_t owner = 0;
    std::uint32_t location = 0;
    std::uint32_t sequence = 0;
    std::uint32_t code = 0;
    std::uint32_t position = 0;
    bool is_public = false;
    std::vector<std::uint32_t> materials;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> counters;  // type, count
};

struct DuelSnapshot {
    std::uint32_t life_points[2] = {0, 0};
    std::vector<CardState> cards;
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
    // Privileged: the full state, hidden cards included.
    DuelSnapshot snapshot() const;
    // Answers removed so far by loop blocking.
    std::uint64_t blocked_answers() const noexcept { return blocked_answers_; }

    // An independent copy of the duel at the pending decision: the engine state (ocgcore fork's
    // OCG_CloneDuel) and everything this wrapper keeps (observation sessions, the pending request
    // and observation, loop-guard counts, decision count, recent decisions, config). Given the same
    // answers, the copy plays on exactly as this duel would. With a seed, the engine's random state
    // is reseeded instead: config() stays the duel's origin (deck order and opening hands come from
    // config().seed) and the seed is appended to reseeds(). Hidden cards (deck order, the
    // opponent's hand) are copied as they are, so a search must not treat a reseeded clone as a
    // sampled world. Only while !finished(); the copy does not depend on this duel, which may be
    // destroyed first. Clones of one duel may be made from several threads at once, as long as
    // nothing acts on that duel meanwhile.
    std::unique_ptr<Duel> clone(std::optional<std::uint64_t> seed = std::nullopt) const;
    const DuelConfig& config() const noexcept { return config_; }
    // Seeds this duel's engine was reseeded with by clone(seed), oldest first; empty if never.
    const std::vector<std::uint64_t>& reseeds() const noexcept { return reseeds_; }

private:
    Duel(const Duel& source, std::unique_ptr<core::CoreHost> host);
    void advance();
    void publish(protocol::DecisionRequest request);
    void submit(std::vector<std::uint8_t> response);
    void lose_by_loop(std::uint8_t player);
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
    // Per question-and-state hash: how often each answer (semantic key) was given.
    std::map<std::string, std::map<std::string, std::uint32_t>> answers_;
    std::string state_;  // hash of the pending engine question; empty during a continuation
    std::uint64_t blocked_answers_ = 0;
    std::deque<std::string> recent_;
    std::vector<std::uint64_t> reseeds_;
};

}  // namespace ygo::duel
