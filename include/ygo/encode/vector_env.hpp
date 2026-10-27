#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
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

struct StepResult {
    std::vector<std::uint8_t> done;    // the duel of this env ended with this step
    std::vector<std::uint8_t> winner;  // when done: 0 or 1, 2 = draw, 255 = error (see errors)
    std::vector<std::string> errors;   // when done with an error: the DuelError message
};

// num_envs duels that always have a pending decision: a finished duel is
// replaced by a fresh one (new seed, new deck pair, random starting player).
// Not thread-safe: one caller at a time (step throws if called concurrently).
// If a new duel cannot be started at all, step throws and the env is unusable.
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

    std::uint64_t duels_finished() const noexcept { return duels_finished_; }
    std::uint64_t duel_errors() const noexcept { return duel_errors_; }
    // Duels that failed to start (and were replaced by another seed/deck pair), and the
    // last reason. Nonzero usually means a deck or card data problem.
    std::uint64_t start_failures() const noexcept { return start_failures_; }
    std::string last_start_failure() const;

private:
    struct Slot;
    void reset(Slot& slot);

    VectorEnvConfig config_;
    CardIndex card_index_;
    std::vector<std::unique_ptr<Slot>> slots_;
    ThreadPool pool_;
    std::uint64_t duels_finished_ = 0;
    std::uint64_t duel_errors_ = 0;
    std::string broken_;
    std::atomic<std::uint64_t> start_failures_ = 0;
    mutable std::mutex start_failure_mutex_;
    std::string last_start_failure_;
    std::atomic<bool> stepping_ = false;
};

}  // namespace ygo::encode
