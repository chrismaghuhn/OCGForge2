#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ygo/core/deck.hpp"
#include "ygo/duel/duel.hpp"
#include "ygo/encode/encoder.hpp"

namespace ygo::encode {

// Runs fn(i) for i in [0, n) on a fixed set of worker threads.
class ThreadPool final {
public:
    explicit ThreadPool(std::size_t threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn);

private:
    void work();

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<void(std::size_t)>* job_ = nullptr;
    std::size_t job_size_ = 0;
    std::size_t next_ = 0;
    std::size_t finished_ = 0;
    std::uint64_t generation_ = 0;
    bool stop_ = false;
};

struct VectorEnvConfig {
    std::filesystem::path card_scripts_root;
    std::filesystem::path card_data_tsv;
    std::vector<core::Deck> decks;  // each duel draws two of these (with replacement)
    std::size_t num_envs = 1;
    std::size_t threads = 0;  // 0 = hardware concurrency
    std::uint64_t seed = 0;
    std::uint64_t max_decisions = 5000;
    // Answer decisions with a single candidate on the worker, without building or encoding an
    // observation; such envs are never handed out. They still count towards max_decisions and are
    // listed in EnvInfo::actions, but skip the check that a candidate's card is visible to the
    // player (it runs while building an observation).
    bool skip_forced = false;
};

// What an env is doing, for diagnosing a duel that hangs: enough to replay it
// with duel::Duel (same seed, decks, starting player, then the actions).
struct EnvInfo {
    bool busy = false;  // inside step right now
    std::uint64_t seed = 0;
    std::uint8_t starting_player = 0;
    std::pair<std::size_t, std::size_t> decks;
    std::vector<std::string> actions;  // semantic keys played in this duel, the last one possibly still running
};

// An env whose decision is pending, from VectorEnv::recv: done/winner/error describe the
// move that got it there (when done, the env already holds a new duel).
struct Ready {
    std::size_t env = 0;
    std::uint8_t done = 0;
    std::uint8_t winner = 255;
    std::string error;
};

struct StepResult {
    std::vector<std::uint8_t> done;    // the duel of this env ended with this step
    std::vector<std::uint8_t> winner;  // when done: 0 or 1, 2 = draw, 255 = error (see errors)
    std::vector<std::string> errors;   // when done with an error: the DuelError message
};

// num_envs duels that always have a pending decision: a finished duel is
// replaced by a fresh one (new seed, new deck pair, random starting player).
//
// Two ways to drive it (pick one per env):
// - step: every env moves, in lockstep; deterministic, but each step waits for the
//   slowest duel.
// - recv/send: recv hands out envs whose decision is ready, send gives their moves
//   to worker threads and returns at once; the next recv returns whichever envs are
//   done first. Nothing waits for stragglers, and the caller (e.g. the GPU) works
//   while the engine does. Per-env play is still deterministic; batch composition
//   is not.
// Not thread-safe: one caller at a time. If a new duel cannot be started at all,
// the env throws and is unusable.
class VectorEnv final {
public:
    VectorEnv(VectorEnvConfig config, CardIndex card_index);
    ~VectorEnv();

    std::size_t size() const noexcept { return slots_.size(); }
    const EncodedDecision& decision(std::size_t env) const;
    // How many decisions this env's current duel has had before the pending one.
    std::uint64_t duel_decision(std::size_t env) const;
    // Deck indices (into config.decks) of player 0 and 1 in this env's current duel.
    std::pair<std::size_t, std::size_t> decks(std::size_t env) const;
    // Safe to call while another thread is inside step.
    EnvInfo info(std::size_t env) const;

    // actions[env] is the index of the chosen candidate in decision(env).
    StepResult step(const std::vector<std::int64_t>& actions);

    // Waits until at least min_count envs (or all that can still become ready) have a
    // pending decision, and hands them out. decision(env) is valid for them until send.
    std::vector<Ready> recv(std::size_t min_count);
    // Plays actions[i] in envs[i]; each env must have been handed out by recv.
    void send(const std::vector<std::size_t>& envs, const std::vector<std::int64_t>& actions);

    std::uint64_t duels_finished() const noexcept { return duels_finished_; }
    std::uint64_t duel_errors() const noexcept { return duel_errors_; }
    // Loop protection (duel::DuelConfig::block_after), summed over finished duels: answers
    // blocked, and duels lost by a player who looped anyway.
    std::uint64_t blocked_answers() const noexcept { return blocked_answers_; }
    std::uint64_t loop_losses() const noexcept { return loop_losses_; }
    // Decisions answered on the worker because they had a single candidate (config.skip_forced).
    std::uint64_t forced_decisions() const noexcept { return forced_decisions_; }
    // Duels that failed to start (and were replaced by another seed/deck pair), and the
    // last reason. Nonzero usually means a deck or card data problem.
    std::uint64_t start_failures() const noexcept { return start_failures_; }
    std::string last_start_failure() const;

private:
    struct Slot;
    void reset(Slot& slot);
    // reset, then (with skip_forced) the new duel's forced decisions; throws like reset.
    void restart(Slot& slot);
    void count_finished(const Slot& slot, const std::string& error);
    // With config.skip_forced: plays single-candidate decisions until a real choice or the end.
    void answer_forced(Slot& slot);
    // Plays one move in one env (restarting it when the duel ends). Throws only when a new
    // duel cannot be started.
    Ready play(std::size_t env, std::int64_t action);
    void check_action(std::size_t env, std::int64_t action) const;
    // In recv/send mode an env may only be read while recv has handed it out.
    void require_handed_out(std::size_t env) const;
    const duel::Duel& live_duel(std::size_t env) const;
    void start_workers();
    void work();

    VectorEnvConfig config_;
    CardIndex card_index_;
    std::vector<std::unique_ptr<Slot>> slots_;
    ThreadPool pool_;
    std::atomic<std::uint64_t> duels_finished_ = 0;
    std::atomic<std::uint64_t> duel_errors_ = 0;
    std::atomic<std::uint64_t> blocked_answers_ = 0;
    std::atomic<std::uint64_t> loop_losses_ = 0;
    std::atomic<std::uint64_t> forced_decisions_ = 0;

    // recv/send mode
    enum class EnvState : std::uint8_t { Queued, HandedOut, Playing };
    std::vector<std::thread> workers_;
    mutable std::mutex async_mutex_;
    std::condition_variable jobs_ready_;
    std::condition_variable envs_ready_;
    std::deque<std::pair<std::size_t, std::int64_t>> jobs_;
    std::deque<Ready> ready_;
    std::vector<EnvState> state_;
    std::size_t playing_ = 0;  // envs queued or being played by workers
    bool stop_workers_ = false;
    std::string broken_;
    std::atomic<std::uint64_t> start_failures_ = 0;
    mutable std::mutex start_failure_mutex_;
    std::string last_start_failure_;
    std::atomic<bool> stepping_ = false;
};

}  // namespace ygo::encode
