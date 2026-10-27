#include "script_reader.hpp"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ocgapi.h"

namespace ygo::core::detail {

namespace {

std::optional<std::uint32_t> card_code_from_script_name(const std::string& name) {
    const auto separator = name.find_last_of("/\\");
    const auto base = separator == std::string::npos ? name : name.substr(separator + 1);
    if (base.size() < 6 || base.front() != 'c' || base.substr(base.size() - 4) != ".lua") {
        return std::nullopt;
    }
    const auto digits = base.substr(1, base.size() - 5);
    try {
        std::size_t consumed = 0;
        const auto value = std::stoull(digits, &consumed, 10);
        if (consumed != digits.size() || value > 0xffffffffULL) {
            return std::nullopt;
        }
        return static_cast<std::uint32_t>(value);
    } catch (...) {
        return std::nullopt;
    }
}

}  // namespace

int ScriptStore::load(OCG_Duel duel, const char* name, std::string* error) {
    if (name == nullptr) {
        if (error != nullptr) {
            *error = "script callback received null name";
        }
        return 0;
    }

    std::string requested(name);
    while (requested.rfind("./", 0) == 0) {
        requested.erase(0, 2);
    }
    auto path = root_ / requested;
    std::ifstream stream(path, std::ios::binary);
    if (!stream && requested.find('/') == std::string::npos && requested.find('\\') == std::string::npos) {
        // The pinned CardScripts tree keeps official card scripts and the
        // optional unofficial helper in canonical subdirectories, while the
        // core requests both card scripts and helpers by basename.
        path = root_ / "official" / requested;
        stream.clear();
        stream.open(path, std::ios::binary);
    }
    if (!stream && requested.find('/') == std::string::npos && requested.find('\\') == std::string::npos) {
        path = root_ / "unofficial" / requested;
        stream.clear();
        stream.open(path, std::ios::binary);
    }
    if (!stream) {
        // Normal monsters intentionally have no cXXXX.lua script. Required
        // effect/procedure cards must fail closed when their script is absent,
        // including requests that carry an official/ subdirectory prefix.
        const auto code = card_code_from_script_name(requested);
        if (code.has_value()) {
            if (required_codes_.find(*code) != required_codes_.end() && error != nullptr) {
                *error = "required card script not found: " + requested;
            }
            return 0;
        }
        if (error != nullptr) {
            *error = "script not found: " + path.string();
        }
        return 0;
    }

    const std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    if (OCG_LoadScript(duel, bytes.data(), static_cast<std::uint32_t>(bytes.size()), name) == 0) {
        if (error != nullptr) {
            *error = "OCG_LoadScript rejected: " + requested;
        }
        return 0;
    }
    ++successful_loads_;
    return 1;
}

int script_reader_callback(void* payload, OCG_Duel duel, const char* name) {
    auto* context = static_cast<std::pair<ScriptStore*, std::string*>*>(payload);
    ++context->first->reader_requests_;
    return context->first->load(duel, name, context->second);
}

void log_callback(void* payload, const char* message, int type) {
    auto* output = static_cast<std::string*>(payload);
    if (output != nullptr && message != nullptr) {
        *output += "[" + std::to_string(type) + "] " + message + "\n";
    }
}

}  // namespace ygo::core::detail
