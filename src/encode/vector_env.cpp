#include "ygo/encode/vector_env.hpp"

#include <random>
#include <stdexcept>
#include <utility>

namespace ygo::encode {

ThreadPool::ThreadPool(std::size_t threads) {
    if (threads == 0) {
        threads = std::max(1u, std::thread::hardware_concurrency());
    }
    for (std::size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this] { work(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (auto& worker : workers_) {
        worker.join();
    }
}

void ThreadPool::parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn) {
    if (n == 0) {
        return;
    }
    std::unique_lock lock(mutex_);
    job_ = &fn;
    job_size_ = n;
    next_ = 0;
    finished_ = 0;
    ++generation_;
    wake_.notify_all();
    done_.wait(lock, [&] { return finished_ == job_size_; });
    job_ = nullptr;
}

void ThreadPool::work() {
    std::uint64_t seen = 0;
    std::unique_lock lock(mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return stop_ || (generation_ != seen && job_ != nullptr && next_ < job_size_); });
        if (stop_) {
            return;
        }
        seen = generation_;
        while (job_ != nullptr && next_ < job_size_) {
            const auto index = next_++;
            const auto* job = job_;
            lock.unlock();
            (*job)(index);  // jobs must not throw
            lock.lock();
            if (++finished_ == job_size_) {
                done_.notify_one();
            }
        }
    }
}

struct VectorEnv::Slot {
    mutable std::mutex mutex;  // guards info
    EnvInfo info;
    std::size_t index = 0;
    std::uint64_t episode = 0;
    std::mt19937_64 rng;
    std::unique_ptr<duel::Duel> duel;
    std::pair<std::size_t, std::size_t> decks;
    EncodedDecision decision;
};

VectorEnv::VectorEnv(VectorEnvConfig config, CardIndex card_index)
    : config_(std::move(config)), card_index_(std::move(card_index)), pool_(config_.threads) {
    if (config_.decks.empty()) {
        throw std::invalid_argument("VectorEnv needs at least one deck");
    }
    for (std::size_t i = 0; i < config_.num_envs; ++i) {
        auto slot = std::make_unique<Slot>();
        slot->index = i;
        // seed_seq takes 32-bit words; split the 64-bit seed so it is used in full.
        std::seed_seq seed{static_cast<std::uint32_t>(config_.seed), static_cast<std::uint32_t>(config_.seed >> 32),
                           static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(static_cast<std::uint64_t>(i) >> 32)};
        slot->rng.seed(seed);
        slots_.push_back(std::move(slot));
    }
    std::vector<std::string> errors(slots_.size());
    pool_.parallel_for(slots_.size(), [&](std::size_t i) {
        try {
            reset(*slots_[i]);
        } catch (const std::exception& error) {
            errors[i] = error.what();
        }
    });
    for (const auto& error : errors) {
        if (!error.empty()) {
            throw std::runtime_error("VectorEnv could not start a duel: " + error);
        }
    }
}

VectorEnv::~VectorEnv() {
    {
        std::lock_guard lock(async_mutex_);
        stop_workers_ = true;
    }
    jobs_ready_.notify_all();
    for (auto& worker : workers_) {
        worker.join();
    }
}

// Starts a fresh duel in the slot and encodes its first decision. A duel that
// fails before its first decision is retried with the next seed.
void VectorEnv::reset(Slot& slot) {
    for (int attempt = 0;; ++attempt) {
        duel::DuelConfig config;
        config.card_scripts_root = config_.card_scripts_root;
        config.card_data_tsv = config_.card_data_tsv;
        std::uniform_int_distribution<std::size_t> pick(0, config_.decks.size() - 1);
        slot.decks = {pick(slot.rng), pick(slot.rng)};
        config.deck0 = config_.decks[slot.decks.first];
        config.deck1 = config_.decks[slot.decks.second];
        config.seed = slot.rng();
        config.starting_player = static_cast<std::uint8_t>(slot.rng() & 1);
        config.max_decisions = config_.max_decisions;
        ++slot.episode;
        {
            std::lock_guard lock(slot.mutex);
            slot.info.seed = config.seed;
            slot.info.starting_player = config.starting_player;
            slot.info.decks = slot.decks;
            slot.info.actions.clear();
        }
        try {
            slot.duel = std::make_unique<duel::Duel>(std::move(config));
            if (slot.duel->finished()) {
                continue;  // ended without a single decision; just start another
            }
            slot.decision = encode(*slot.duel->observation(), slot.duel->request(), card_index_);
            return;
        } catch (const std::exception& error) {  // stuck duel or undecodable first decision
            ++start_failures_;
            {
                std::lock_guard lock(start_failure_mutex_);
                last_start_failure_ = error.what();
            }
            if (attempt >= 10) {
                throw;
            }
        }
    }
}

void VectorEnv::require_handed_out(std::size_t env) const {
    std::lock_guard lock(async_mutex_);
    if (!workers_.empty() && (env >= state_.size() || state_[env] != EnvState::HandedOut)) {
        throw std::logic_error("env " + std::to_string(env) + " is being played; in recv/send mode only envs "
                               "handed out by recv can be read");
    }
}

const EncodedDecision& VectorEnv::decision(std::size_t env) const {
    require_handed_out(env);
    return slots_.at(env)->decision;
}

std::uint64_t VectorEnv::duel_decision(std::size_t env) const {
    require_handed_out(env);
    return slots_.at(env)->duel->decision_count();
}

std::string VectorEnv::last_start_failure() const {
    std::lock_guard lock(start_failure_mutex_);
    return last_start_failure_;
}

EnvInfo VectorEnv::info(std::size_t env) const {
    const auto& slot = *slots_.at(env);
    std::lock_guard lock(slot.mutex);
    return slot.info;
}

std::pair<std::size_t, std::size_t> VectorEnv::decks(std::size_t env) const {
    require_handed_out(env);
    return slots_.at(env)->decks;
}

void VectorEnv::check_action(std::size_t env, std::int64_t action) const {
    const auto count = static_cast<std::int64_t>(slots_.at(env)->duel->request().candidates.size());
    if (action < 0 || action >= count) {
        throw std::out_of_range("action " + std::to_string(action) + " for env " + std::to_string(env) +
                                " is not a candidate index below " + std::to_string(count));
    }
}

Ready VectorEnv::play(std::size_t env, std::int64_t action) {
    auto& slot = *slots_[env];
    Ready result;
    result.env = env;
    try {
        const auto& request = slot.duel->request();
        const auto& key = request.candidates[static_cast<std::size_t>(action)].semantic_key;
        {
            std::lock_guard lock(slot.mutex);
            slot.info.actions.push_back(key);
            slot.info.busy = true;
        }
        struct NotBusy {
            Slot& slot;
            ~NotBusy() {
                std::lock_guard lock(slot.mutex);
                slot.info.busy = false;
            }
        } not_busy{slot};
        slot.duel->act(key);
        if (!slot.duel->finished()) {
            slot.decision = encode(*slot.duel->observation(), slot.duel->request(), card_index_);
            return result;
        }
        result.done = 1;
        result.winner = slot.duel->result()->winner;
    } catch (const std::exception& error) {
        // DuelError (stuck, engine failure) or a decision the encoder cannot
        // represent: this duel has no result, start another.
        result.done = 1;
        result.error = error.what();
    }
    ++duels_finished_;
    duel_errors_ += result.error.empty() ? 0 : 1;
    reset(slot);  // throws if no new duel can be started
    return result;
}

StepResult VectorEnv::step(const std::vector<std::int64_t>& actions) {
    {
        std::lock_guard lock(async_mutex_);
        if (!workers_.empty()) {
            throw std::logic_error("VectorEnv::step cannot be mixed with recv/send");
        }
    }
    if (!broken_.empty()) {
        throw std::logic_error("VectorEnv is unusable after an earlier error: " + broken_);
    }
    if (stepping_.exchange(true)) {
        throw std::logic_error("VectorEnv::step called while another step is running");
    }
    struct Release {
        std::atomic<bool>& flag;
        ~Release() { flag = false; }
    } release{stepping_};
    if (actions.size() != slots_.size()) {
        throw std::invalid_argument("step needs one action per env");
    }
    for (std::size_t i = 0; i < actions.size(); ++i) {
        check_action(i, actions[i]);
    }
    StepResult result;
    result.done.assign(slots_.size(), 0);
    result.winner.assign(slots_.size(), 255);
    result.errors.assign(slots_.size(), "");
    std::vector<std::string> fatal(slots_.size());
    pool_.parallel_for(slots_.size(), [&](std::size_t i) {
        try {
            auto ready = play(i, actions[i]);
            result.done[i] = ready.done;
            result.winner[i] = ready.winner;
            result.errors[i] = std::move(ready.error);
        } catch (const std::exception& error) {
            fatal[i] = error.what();
        }
    });
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (!fatal[i].empty()) {
            broken_ = "env " + std::to_string(i) + " could not start a new duel: " + fatal[i];
            throw std::runtime_error(broken_);
        }
    }
    return result;
}

void VectorEnv::start_workers() {
    state_.assign(slots_.size(), EnvState::Queued);
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        ready_.push_back(Ready{i, 0, 255, {}});
    }
    const auto threads = config_.threads != 0 ? config_.threads : std::max(1u, std::thread::hardware_concurrency());
    for (std::size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this] { work(); });
    }
}

void VectorEnv::work() {
    std::unique_lock lock(async_mutex_);
    for (;;) {
        jobs_ready_.wait(lock, [&] { return stop_workers_ || !jobs_.empty(); });
        if (stop_workers_) {
            return;
        }
        const auto [env, action] = jobs_.front();
        jobs_.pop_front();
        lock.unlock();
        Ready ready;
        std::string fatal;
        try {
            ready = play(env, action);
        } catch (const std::exception& error) {
            fatal = "env " + std::to_string(env) + " could not start a new duel: " + error.what();
        } catch (...) {
            fatal = "env " + std::to_string(env) + ": unknown error";
        }
        lock.lock();
        --playing_;
        if (!fatal.empty()) {
            broken_ = fatal;
        } else {
            state_[env] = EnvState::Queued;
            ready_.push_back(std::move(ready));
        }
        envs_ready_.notify_one();
    }
}

std::vector<Ready> VectorEnv::recv(std::size_t min_count) {
    if (stepping_) {
        throw std::logic_error("VectorEnv::recv called while step is running");
    }
    std::unique_lock lock(async_mutex_);
    if (workers_.empty()) {
        start_workers();
    }
    // Never wait for more envs than can still arrive.
    envs_ready_.wait(lock, [&] {
        return !broken_.empty() || ready_.size() >= std::min(min_count, ready_.size() + playing_);
    });
    if (!broken_.empty()) {
        throw std::runtime_error(broken_);
    }
    std::vector<Ready> out(std::make_move_iterator(ready_.begin()), std::make_move_iterator(ready_.end()));
    ready_.clear();
    for (const auto& ready : out) {
        state_[ready.env] = EnvState::HandedOut;
    }
    return out;
}

void VectorEnv::send(const std::vector<std::size_t>& envs, const std::vector<std::int64_t>& actions) {
    if (envs.size() != actions.size()) {
        throw std::invalid_argument("send needs one action per env");
    }
    std::lock_guard lock(async_mutex_);
    if (!broken_.empty()) {
        throw std::runtime_error(broken_);
    }
    std::vector<bool> seen(slots_.size());
    for (std::size_t i = 0; i < envs.size(); ++i) {
        if (envs[i] >= slots_.size() || state_.empty() || state_[envs[i]] != EnvState::HandedOut) {
            throw std::logic_error("send for env " + std::to_string(envs[i]) + ", which recv did not hand out");
        }
        if (seen[envs[i]]) {
            throw std::logic_error("send got env " + std::to_string(envs[i]) + " twice");
        }
        seen[envs[i]] = true;
        check_action(envs[i], actions[i]);
    }
    for (std::size_t i = 0; i < envs.size(); ++i) {
        state_[envs[i]] = EnvState::Playing;
        jobs_.emplace_back(envs[i], actions[i]);
        ++playing_;
    }
    jobs_ready_.notify_all();
}

}  // namespace ygo::encode
