#include "script_reader.hpp"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "lauxlib.h"
#include "lua.h"
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

std::string strip_dot_slash(const char* name) {
    std::string requested(name);
    while (requested.rfind("./", 0) == 0) {
        requested.erase(0, 2);
    }
    return requested;
}

int dump_writer(lua_State*, const void* data, std::size_t size, void* output) {
    static_cast<std::string*>(output)->append(static_cast<const char*>(data), size);
    return 0;
}

}  // namespace

std::shared_ptr<const ScriptCache> ScriptCache::shared(const std::filesystem::path& root) {
    // Kept for the whole process (the card scripts do not change while it runs): a duel created
    // after the last one was destroyed must not compile everything again. Never destroyed either:
    // worker threads may still start duels while the process exits.
    static auto* mutex = new std::mutex;
    static auto* caches = new std::map<std::filesystem::path, std::shared_ptr<const ScriptCache>>;
    std::lock_guard lock(*mutex);
    auto& cache = (*caches)[root];
    if (!cache) {
        cache = std::make_shared<const ScriptCache>(root);
    }
    return cache;
}

const ScriptCache::Entry& ScriptCache::get(const std::string& name) const {
    {
        std::lock_guard lock(mutex_);
        if (const auto it = entries_.find(name); it != entries_.end()) {
            return it->second;
        }
    }
    auto entry = read(name);  // outside the lock: file I/O and compiling
    std::lock_guard lock(mutex_);
    return entries_.try_emplace(name, std::move(entry)).first->second;  // another thread may have won
}

ScriptCache::Entry ScriptCache::read(const std::string& name) const {
    Entry entry;
    const auto requested = strip_dot_slash(name.c_str());
    const bool bare = requested.find('/') == std::string::npos && requested.find('\\') == std::string::npos;
    auto path = root_ / requested;
    std::ifstream stream(path, std::ios::binary);
    if (!stream && bare) {
        // The pinned CardScripts tree keeps official card scripts and the
        // optional unofficial helper in canonical subdirectories, while the
        // core requests both card scripts and helpers by basename.
        path = root_ / "official" / requested;
        stream.clear();
        stream.open(path, std::ios::binary);
    }
    if (!stream && bare) {
        path = root_ / "unofficial" / requested;
        stream.clear();
        stream.open(path, std::ios::binary);
    }
    entry.path = path;
    if (!stream) {
        return entry;  // also cached: plain Normal Monsters ask for scripts that do not exist
    }
    entry.found = true;
    entry.bytes.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    // Compile once in a throwaway state and keep the bytecode, with debug info so error messages
    // keep their line numbers. The chunk name is the engine's script name, as when the engine
    // loads the source itself. A script that does not compile stays source, so each duel still
    // reports the syntax error as before.
    lua_State* state = luaL_newstate();
    if (state != nullptr) {
        if (luaL_loadbuffer(state, entry.bytes.data(), entry.bytes.size(), name.c_str()) == LUA_OK) {
            std::string chunk;
            if (lua_dump(state, dump_writer, &chunk, 0) == 0) {
                entry.bytes = std::move(chunk);
                entry.binary = true;
            }
        }
        lua_close(state);
    }
    return entry;
}

int ScriptStore::load(OCG_Duel duel, const char* name, std::string* error) {
    if (name == nullptr) {
        if (error != nullptr) {
            *error = "script callback received null name";
        }
        return 0;
    }
    const auto& entry = cache_->get(name);
    if (!entry.found) {
        // Normal monsters intentionally have no cXXXX.lua script. Required
        // effect/procedure cards must fail closed when their script is absent,
        // including requests that carry an official/ subdirectory prefix.
        const auto requested = strip_dot_slash(name);
        const auto code = card_code_from_script_name(requested);
        if (code.has_value()) {
            if (required_codes_.find(*code) != required_codes_.end() && error != nullptr) {
                *error = "required card script not found: " + requested;
            }
            return 0;
        }
        if (error != nullptr) {
            *error = "script not found: " + entry.path.string();
        }
        return 0;
    }
    if (OCG_LoadScript(duel, entry.bytes.data(), static_cast<std::uint32_t>(entry.bytes.size()), name) == 0) {
        if (error != nullptr) {
            *error = "OCG_LoadScript rejected: " + strip_dot_slash(name);
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
    constexpr std::size_t kMaxLog = 64 * 1024;  // a script error can repeat thousands of times
    auto* output = static_cast<std::string*>(payload);
    if (output != nullptr && message != nullptr && output->size() < kMaxLog) {
        *output += "[" + std::to_string(type) + "] " + message + "\n";
    }
}

}  // namespace ygo::core::detail
