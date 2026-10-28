#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "ygo/core/card_data.hpp"
#include "ygo/core/deck.hpp"
#include "ygo/core/seed_bundle.hpp"


struct OCG_QueryInfo;

namespace ygo::core {

struct CoreHostConfig {
    std::filesystem::path card_scripts_root;
    std::filesystem::path card_data_tsv;
    SeedBundle seed;
    std::uint64_t duel_flags = 0;
    std::uint32_t starting_lp = 8000;
    std::uint32_t starting_draw_count = 5;
    std::uint32_t draw_count_per_turn = 1;
    std::optional<std::uint8_t> starting_player;
    std::vector<std::uint32_t> required_script_codes;

    std::uint8_t effective_starting_player() const noexcept {
        return starting_player.value_or(0);
    }
};

struct ProcessResult {
    int status = 0;
    std::vector<std::uint8_t> message;
};

struct CoreHostMetrics {
    std::size_t duel_process_calls = 0;
    std::size_t duel_query_calls = 0;
    std::size_t duel_query_location_calls = 0;
    std::size_t duel_query_field_calls = 0;
    std::size_t duel_query_count_calls = 0;
    std::size_t script_reader_requests = 0;
    std::size_t script_loads = 0;
};

class CoreHost final {
public:
    explicit CoreHost(CoreHostConfig config);
    ~CoreHost();

    CoreHost(const CoreHost&) = delete;
    CoreHost& operator=(const CoreHost&) = delete;
    CoreHost(CoreHost&&) = delete;
    CoreHost& operator=(CoreHost&&) = delete;

    void load_deck(std::uint8_t team, const Deck& deck);
    void load_fixture_card(std::uint8_t team, std::uint32_t code, std::uint32_t location,
                           std::uint32_t sequence, std::uint32_t position);
    void load_fixture_script(const std::filesystem::path& path);
    // A Lua script run before the duel starts, e.g. a puzzle that builds the field (Debug.AddCard).
    void load_script_text(const std::string& name, const std::string& text);
    void start_duel();
    ProcessResult process();
    void submit_response(const std::vector<std::uint8_t>& response);

    std::uint32_t query_count(std::uint8_t team, std::uint32_t location) const;
    std::vector<std::uint8_t> query(const OCG_QueryInfo& info) const;
    std::vector<std::uint8_t> query_location(const OCG_QueryInfo& info) const;
    std::vector<std::uint8_t> query_field() const;
    std::optional<StaticCardData> static_card_data(std::uint32_t code) const;

    const CoreHostConfig& config() const noexcept { return config_; }
    std::size_t response_submission_count() const noexcept { return response_submission_count_; }
    std::size_t process_call_count() const noexcept { return process_call_count_; }
    CoreHostMetrics metrics() const noexcept;
    int api_major() const noexcept { return api_major_; }
    int api_minor() const noexcept { return api_minor_; }

private:
    struct Impl;
    void throw_if_callback_error(const char* operation) const;
    CoreHostConfig config_;
    Impl* impl_ = nullptr;
    std::size_t response_submission_count_ = 0;
    std::size_t process_call_count_ = 0;
    mutable std::size_t duel_query_calls_ = 0;
    mutable std::size_t duel_query_location_calls_ = 0;
    mutable std::size_t duel_query_field_calls_ = 0;
    mutable std::size_t duel_query_count_calls_ = 0;
    int api_major_ = 0;
    int api_minor_ = 0;
};

}  // namespace ygo::core
