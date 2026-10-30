#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ocgapi_types.h"


namespace ygo::core::detail {

// Every script file a duel asked for, read once per process and kept as Lua bytecode, so a new
// duel neither reads files nor parses Lua. Only the compiled chunks are shared: each duel still
// runs them in its own Lua state (scripts set globals such as s.global_check). Filled lazily
// under a mutex; entries are never changed or removed, so references to them stay valid.
class ScriptCache final {
public:
    struct Entry {
        bool found = false;
        bool binary = false;            // bytes is a lua_dump chunk; false: the source (it did not compile)
        std::string bytes;
        std::filesystem::path path;     // where it was read, or the last path tried
    };

    // One cache per script root, kept until the process exits.
    static std::shared_ptr<const ScriptCache> shared(const std::filesystem::path& root);

    explicit ScriptCache(std::filesystem::path root) : root_(std::move(root)) {}
    // `name` exactly as the engine requested it; it is also the chunk name (error messages).
    const Entry& get(const std::string& name) const;
    const std::filesystem::path& root() const noexcept { return root_; }

private:
    Entry read(const std::string& name) const;

    std::filesystem::path root_;
    mutable std::mutex mutex_;
    mutable std::unordered_map<std::string, Entry> entries_;
};

class ScriptStore final {
public:
    ScriptStore(const std::filesystem::path& root, const std::vector<std::uint32_t>& required_codes)
        : cache_(ScriptCache::shared(root)), required_codes_(required_codes.begin(), required_codes.end()) {}

    int load(OCG_Duel duel, const char* name, std::string* error);
    // For cards that have no script at all (plain Normal Monsters, tokens).
    void not_required(std::uint32_t code) { required_codes_.erase(code); }
    std::size_t reader_requests() const noexcept { return reader_requests_; }
    std::size_t successful_loads() const noexcept { return successful_loads_; }

private:
    friend int script_reader_callback(void* payload, OCG_Duel duel, const char* name);

    std::shared_ptr<const ScriptCache> cache_;  // shared by copies (cloned duels)
    std::unordered_set<std::uint32_t> required_codes_;
    std::size_t reader_requests_ = 0;
    std::size_t successful_loads_ = 0;
};

int script_reader_callback(void* payload, OCG_Duel duel, const char* name);
void log_callback(void* payload, const char* message, int type);

}  // namespace ygo::core::detail
