#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

#include "ocgapi_types.h"
#include "ygo/core/card_data.hpp"

namespace ygo::core::detail {

class CardDataStore final {
public:
    void load(const std::filesystem::path& path);
    void read(std::uint32_t code, OCG_CardData* output);
    std::optional<StaticCardData> snapshot(std::uint32_t code) const;
    void release(OCG_CardData*) noexcept {}

private:
    struct Record {
        OCG_CardData data{};
        std::array<std::uint16_t, 5> setcodes{};
    };
    std::unordered_map<std::uint32_t, Record> records_;
};

struct CardDataCallbackContext {
    CardDataStore* store = nullptr;
    std::string* error = nullptr;
};

void card_data_callback(void* payload, std::uint32_t code, OCG_CardData* output);
void card_data_done_callback(void* payload, OCG_CardData* data);

}  // namespace ygo::core::detail
