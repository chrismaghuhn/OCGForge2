#include "ygo/core/core_host.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "ocgapi.h"
#include "ocgapi_constants.h"
#include "ygo/core/core_error.hpp"
#include "card_data_reader.hpp"
#include "script_reader.hpp"

namespace ygo::core {

struct CoreHost::Impl {
    OCG_DuelOptions options{};
    detail::CardDataStore card_data;
    detail::CardDataCallbackContext card_data_payload;
    detail::ScriptStore scripts;
    std::string callback_error;
    std::string log;
    std::pair<detail::ScriptStore*, std::string*> script_payload;
    OCG_Duel duel = nullptr;

    explicit Impl(const std::filesystem::path& script_root,
                  const std::vector<std::uint32_t>& required_script_codes
                  )
        : card_data_payload{&card_data, &callback_error},
          scripts(script_root, required_script_codes
          ),
          script_payload(&scripts, &callback_error) {}
};

void CoreHost::throw_if_callback_error(const char* operation) const {
    if (!impl_->callback_error.empty()) {
        throw CoreError(CoreErrorCode::Callback, operation, impl_->callback_error);
    }
}

CoreHost::CoreHost(CoreHostConfig config) : config_(std::move(config)) {
    OCG_GetVersion(&api_major_, &api_minor_);
    if (api_major_ != 11 || api_minor_ != 0) {
        throw CoreError(CoreErrorCode::Lifecycle, "OCG_GetVersion",
                        "expected 11.0, got " + std::to_string(api_major_) + "." + std::to_string(api_minor_));
    }
    impl_ = new Impl(config_.card_scripts_root, config_.required_script_codes
    );
    try {
        impl_->card_data.load(config_.card_data_tsv);
        // Plain Normal Monsters and tokens have no script (Normal Pendulum Monsters do).
        for (const auto code : config_.required_script_codes) {
            const auto data = impl_->card_data.snapshot(code);
            if (data && (((data->type & TYPE_NORMAL) != 0 && (data->type & TYPE_PENDULUM) == 0) ||
                         (data->type & TYPE_TOKEN) != 0)) {
                impl_->scripts.not_required(code);
            }
        }
        auto& options = impl_->options;
        for (std::size_t index = 0; index < config_.seed.words.size(); ++index) {
            options.seed[index] = config_.seed.words[index];
        }
        options.flags = config_.duel_flags;
        options.team1 = {config_.starting_lp, config_.starting_draw_count, config_.draw_count_per_turn};
        options.team2 = options.team1;
        options.cardReader = detail::card_data_callback;
        options.payload1 = &impl_->card_data_payload;
        options.scriptReader = detail::script_reader_callback;
        options.payload2 = &impl_->script_payload;
        options.logHandler = detail::log_callback;
        options.payload3 = &impl_->log;
        options.cardReaderDone = detail::card_data_done_callback;
        options.payload4 = &impl_->card_data;
        options.enableUnsafeLibraries = 0;

        const int status = OCG_CreateDuel(&impl_->duel, &options);
        if (status != OCG_DUEL_CREATION_SUCCESS || impl_->duel == nullptr) {
            throw CoreError(CoreErrorCode::Lifecycle, "OCG_CreateDuel", "status=" + std::to_string(status));
        }

        const auto load_global_script = [this](const std::string& name) {
            std::string error;
            if (impl_->scripts.load(impl_->duel, name.c_str(), &error) == 0) {
                throw CoreError(CoreErrorCode::Callback, "load_global_script", error.empty() ? name : error);
            }
        };
        load_global_script("constant.lua");
        load_global_script("utility.lua");
        load_global_script("proc_normal.lua");
        throw_if_callback_error("OCG_CreateDuel");
    } catch (...) {
        if (impl_->duel != nullptr) {
            OCG_DestroyDuel(impl_->duel);
            impl_->duel = nullptr;
        }
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

CoreHost::~CoreHost() {
    if (impl_ != nullptr) {
        if (impl_->duel != nullptr) {
            OCG_DestroyDuel(impl_->duel);
            impl_->duel = nullptr;
        }
        delete impl_;
        impl_ = nullptr;
    }
}

void CoreHost::load_deck(std::uint8_t team, const Deck& deck) {
    if (team > 1 || deck.main_deck.size() < 40) {
        throw CoreError(CoreErrorCode::Lifecycle, "OCG_DuelNewCard", "fixture deck must contain at least 40 entries");
    }
    // duelist=0 places the card in the requested player/location list. The
    // nonzero values are reserved for tag/relay deck lists by the public API.
    std::uint32_t sequence = 0;
    for (const std::uint32_t code : deck.main_deck) {
        OCG_NewCardInfo info{};
        info.team = team;
        info.duelist = 0;
        info.code = code;
        info.con = team;
        info.loc = LOCATION_DECK;
        info.seq = sequence++;
        info.pos = POS_FACEDOWN_DEFENSE;
        OCG_DuelNewCard(impl_->duel, &info);
        throw_if_callback_error("OCG_DuelNewCard");
    }
    sequence = 0;
    for (const std::uint32_t code : deck.extra_deck) {
        OCG_NewCardInfo info{};
        info.team = team;
        info.duelist = 0;
        info.code = code;
        info.con = team;
        info.loc = LOCATION_EXTRA;
        info.seq = sequence++;
        info.pos = POS_FACEDOWN_DEFENSE;
        OCG_DuelNewCard(impl_->duel, &info);
        throw_if_callback_error("OCG_DuelNewCard");
    }
}

void CoreHost::load_fixture_card(std::uint8_t team, std::uint32_t code, std::uint32_t location,
                                 std::uint32_t sequence, std::uint32_t position) {
    if (team > 1 || code == 0 || location == 0) {
        throw CoreError(CoreErrorCode::Lifecycle, "OCG_DuelNewCard", "invalid fixture card setup");
    }
    OCG_NewCardInfo info{};
    info.team = team;
    info.duelist = 0;
    info.code = code;
    info.con = team;
    info.loc = location;
    info.seq = sequence;
    info.pos = position;
    OCG_DuelNewCard(impl_->duel, &info);
    throw_if_callback_error("OCG_DuelNewCard");
}

void CoreHost::load_fixture_script(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw CoreError(CoreErrorCode::Callback, "OCG_LoadScript", "cannot open fixture script: " + path.string());
    }
    const std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    const auto name = path.filename().string();
    if (OCG_LoadScript(impl_->duel, bytes.data(), static_cast<std::uint32_t>(bytes.size()), name.c_str()) == 0) {
        throw CoreError(CoreErrorCode::Callback, "OCG_LoadScript", "fixture script rejected: " + path.string());
    }
    throw_if_callback_error("OCG_LoadScript");
}

void CoreHost::start_duel() {
    if (config_.starting_player.has_value()) {
        const int status = OCG_DuelSetStartingPlayer(impl_->duel, *config_.starting_player);
        if (status != 1) {
            throw CoreError(CoreErrorCode::Lifecycle, "OCG_DuelSetStartingPlayer",
                            "starting player was rejected before duel start");
        }
    }
    OCG_StartDuel(impl_->duel);
    throw_if_callback_error("OCG_StartDuel");
}

ProcessResult CoreHost::process() {
    ++process_call_count_;
    const int status = OCG_DuelProcess(impl_->duel);
    throw_if_callback_error("OCG_DuelProcess");
    if (OCG_DuelLuaBudgetExceeded(impl_->duel) != 0) {
        // A card script searched too many combinations and was cut off (our ocgcore
        // patch 0004); the duel state can no longer be trusted.
        throw CoreError(CoreErrorCode::Lifecycle, "OCG_DuelProcess", "Lua instruction budget exceeded");
    }
    std::uint32_t length = 0;
    auto* raw = static_cast<const std::uint8_t*>(OCG_DuelGetMessage(impl_->duel, &length));
    ProcessResult result;
    result.status = status;
    if (raw != nullptr && length != 0) {
        result.message.assign(raw, raw + length);
    }
    return result;
}

void CoreHost::submit_response(const std::vector<std::uint8_t>& response) {
    if (response.empty()) {
        throw CoreError(CoreErrorCode::Response, "OCG_DuelSetResponse", "empty response");
    }
    OCG_DuelSetResponse(impl_->duel, response.data(), static_cast<std::uint32_t>(response.size()));
    ++response_submission_count_;
    throw_if_callback_error("OCG_DuelSetResponse");
}

std::uint32_t CoreHost::query_count(std::uint8_t team, std::uint32_t location) const {
    ++duel_query_count_calls_;
    return OCG_DuelQueryCount(impl_->duel, team, location);
}

std::vector<std::uint8_t> CoreHost::query_location(const OCG_QueryInfo& info) const {
    ++duel_query_location_calls_;
    std::uint32_t length = 0;
    auto* raw = static_cast<const std::uint8_t*>(OCG_DuelQueryLocation(impl_->duel, &length, &info));
    if (raw == nullptr && length != 0) {
        throw CoreError(CoreErrorCode::Query, "OCG_DuelQueryLocation", "null result with nonzero length");
    }
    return raw == nullptr ? std::vector<std::uint8_t>{} : std::vector<std::uint8_t>(raw, raw + length);
}

std::vector<std::uint8_t> CoreHost::query(const OCG_QueryInfo& info) const {
    ++duel_query_calls_;
    std::uint32_t length = 0;
    auto* raw = static_cast<const std::uint8_t*>(OCG_DuelQuery(impl_->duel, &length, &info));
    if (raw == nullptr && length != 0) {
        throw CoreError(CoreErrorCode::Query, "OCG_DuelQuery", "null result with nonzero length");
    }
    return raw == nullptr ? std::vector<std::uint8_t>{} : std::vector<std::uint8_t>(raw, raw + length);
}

std::vector<std::uint8_t> CoreHost::query_field() const {
    ++duel_query_field_calls_;
    std::uint32_t length = 0;
    auto* raw = static_cast<const std::uint8_t*>(OCG_DuelQueryField(impl_->duel, &length));
    if (raw == nullptr && length != 0) {
        throw CoreError(CoreErrorCode::Query, "OCG_DuelQueryField", "null result with nonzero length");
    }
    return raw == nullptr ? std::vector<std::uint8_t>{} : std::vector<std::uint8_t>(raw, raw + length);
}

CoreHostMetrics CoreHost::metrics() const noexcept {
    return CoreHostMetrics{process_call_count_, duel_query_calls_, duel_query_location_calls_, duel_query_field_calls_,
                           duel_query_count_calls_, impl_->scripts.reader_requests(), impl_->scripts.successful_loads()};
}

std::optional<StaticCardData> CoreHost::static_card_data(std::uint32_t code) const {
    return impl_->card_data.snapshot(code);
}

}  // namespace ygo::core
