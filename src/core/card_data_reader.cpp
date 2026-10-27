#include "card_data_reader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ygo::core::detail {

void CardDataStore::load(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open generated card data: " + path.string());
    }

    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::stringstream parser(line);
        std::string field;
        std::uint64_t values[12]{};
        for (std::size_t index = 0; index < 12; ++index) {
            if (!std::getline(parser, field, '|')) {
                throw std::runtime_error("malformed generated card data row: " + line);
            }
            values[index] = std::stoull(field);
        }

        Record record;
        record.data.code = static_cast<std::uint32_t>(values[0]);
        record.data.alias = static_cast<std::uint32_t>(values[1]);
        record.data.type = static_cast<std::uint32_t>(values[3]);
        record.data.level = static_cast<std::uint32_t>(values[4]);
        record.data.attribute = static_cast<std::uint32_t>(values[5]);
        record.data.race = values[6];
        record.data.attack = static_cast<std::int32_t>(values[7]);
        record.data.defense = static_cast<std::int32_t>(values[8]);
        record.data.lscale = static_cast<std::uint32_t>(values[9]);
        record.data.rscale = static_cast<std::uint32_t>(values[10]);
        record.data.link_marker = static_cast<std::uint32_t>(values[11]);

        const std::uint64_t setcode = values[2];
        for (std::size_t index = 0; index < 4; ++index) {
            record.setcodes[index] = static_cast<std::uint16_t>((setcode >> (index * 16)) & 0xffff);
        }
        record.setcodes[4] = 0;
        record.data.setcodes = record.setcodes.data();
        records_[record.data.code] = record;
        records_[record.data.code].data.setcodes = records_[record.data.code].setcodes.data();
    }
}

void CardDataStore::read(std::uint32_t code, OCG_CardData* output) {
    if (output == nullptr) {
        throw std::runtime_error("card data callback received null output");
    }
    // The core constructs an internal temporary card with passcode zero while
    // creating a duel. It is not a database lookup and carries no card data.
    if (code == 0) {
        *output = OCG_CardData{};
        return;
    }
    const auto it = records_.find(code);
    if (it == records_.end()) {
        throw std::runtime_error("card passcode not present in generated fixture data: " + std::to_string(code));
    }
    *output = it->second.data;
}

std::optional<StaticCardData> CardDataStore::snapshot(std::uint32_t code) const {
    const auto it = records_.find(code);
    if (it == records_.end()) {
        return std::nullopt;
    }
    StaticCardData result;
    result.code = it->second.data.code;
    result.alias = it->second.data.alias;
    result.type = it->second.data.type;
    result.level = it->second.data.level;
    result.attribute = it->second.data.attribute;
    result.race = it->second.data.race;
    result.attack = it->second.data.attack;
    result.defense = it->second.data.defense;
    result.left_scale = it->second.data.lscale;
    result.right_scale = it->second.data.rscale;
    result.link_marker = it->second.data.link_marker;
    for (const auto setcode : it->second.setcodes) {
        if (setcode != 0) {
            result.setcodes.push_back(setcode);
        }
    }
    return result;
}

void card_data_callback(void* payload, std::uint32_t code, OCG_CardData* output) {
    auto* context = static_cast<CardDataCallbackContext*>(payload);
    try {
        if (context == nullptr || context->store == nullptr) {
            throw std::runtime_error("card callback context is null");
        }
        context->store->read(code, output);
    } catch (const std::exception& error) {
        if (output != nullptr) {
            *output = OCG_CardData{};
        }
        if (context != nullptr && context->error != nullptr && context->error->empty()) {
            *context->error = error.what();
        }
    } catch (...) {
        if (output != nullptr) {
            *output = OCG_CardData{};
        }
        if (context != nullptr && context->error != nullptr && context->error->empty()) {
            *context->error = "unknown card callback failure";
        }
    }
}

void card_data_done_callback(void* payload, OCG_CardData* data) {
    static_cast<CardDataStore*>(payload)->release(data);
}

}  // namespace ygo::core::detail
