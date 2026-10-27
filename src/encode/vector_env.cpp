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

VectorEnv::~VectorEnv() = default;

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
        try {
            slot.duel = std::make_unique<duel::Duel>(std::move(config));
            if (slot.duel->finished()) {
                continue;  // ended without a single decision; just start another
            }
            slot.decision = encode(*slot.duel->observation(), slot.duel->request(), card_index_);
            return;
        } catch (const std::exception&) {  // stuck duel or undecodable first decision
            if (attempt >= 10) {
                throw;
            }
        }
    }
}

const EncodedDecision& VectorEnv::decision(std::size_t env) const { return slots_.at(env)->decision; }

std::uint64_t VectorEnv::duel_decision(std::size_t env) const { return slots_.at(env)->duel->decision_count(); }

std::pair<std::size_t, std::size_t> VectorEnv::decks(std::size_t env) const { return slots_.at(env)->decks; }

StepResult VectorEnv::step(const std::vector<std::int64_t>& actions) {
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
        const auto count = static_cast<std::int64_t>(slots_[i]->duel->request().candidates.size());
        if (actions[i] < 0 || actions[i] >= count) {
            throw std::out_of_range("action " + std::to_string(actions[i]) + " for env " + std::to_string(i) +
                                    " is not a candidate index below " + std::to_string(count));
        }
    }
    StepResult result;
    result.done.assign(slots_.size(), 0);
    result.winner.assign(slots_.size(), 255);
    result.errors.assign(slots_.size(), "");
    std::vector<std::string> fatal(slots_.size());
    pool_.parallel_for(slots_.size(), [&](std::size_t i) {
        auto& slot = *slots_[i];
        try {
            try {
                const auto& request = slot.duel->request();
                slot.duel->act(request.candidates[static_cast<std::size_t>(actions[i])].semantic_key);
                if (!slot.duel->finished()) {
                    slot.decision = encode(*slot.duel->observation(), slot.duel->request(), card_index_);
                    return;
                }
                result.done[i] = 1;
                result.winner[i] = slot.duel->result()->winner;
            } catch (const std::exception& error) {
                // DuelError (stuck, engine failure) or a decision the encoder cannot
                // represent: this duel has no result, start another.
                result.done[i] = 1;
                result.errors[i] = error.what();
            }
            reset(slot);
        } catch (const std::exception& error) {
            fatal[i] = error.what();
        }
    });
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (!fatal[i].empty()) {
            broken_ = "env " + std::to_string(i) + " could not start a new duel: " + fatal[i];
            throw std::runtime_error(broken_);
        }
        duels_finished_ += result.done[i];
        duel_errors_ += result.errors[i].empty() ? 0 : 1;
    }
    return result;
}

}  // namespace ygo::encode
