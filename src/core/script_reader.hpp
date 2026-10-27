#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

#include "ocgapi_types.h"


namespace ygo::core::detail {

class ScriptStore final {
public:
    ScriptStore(std::filesystem::path root, const std::vector<std::uint32_t>& required_codes
                )
        : root_(std::move(root)), required_codes_(required_codes.begin(), required_codes.end())
    {}

    int load(OCG_Duel duel, const char* name, std::string* error);
    std::size_t reader_requests() const noexcept { return reader_requests_; }
    std::size_t successful_loads() const noexcept { return successful_loads_; }

private:
    friend int script_reader_callback(void* payload, OCG_Duel duel, const char* name);

    std::filesystem::path root_;
    std::unordered_set<std::uint32_t> required_codes_;
    std::size_t reader_requests_ = 0;
    std::size_t successful_loads_ = 0;
};

int script_reader_callback(void* payload, OCG_Duel duel, const char* name);
void log_callback(void* payload, const char* message, int type);

}  // namespace ygo::core::detail
