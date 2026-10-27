#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ygo::util {

std::string sha256_bytes(const std::vector<std::uint8_t>& bytes);
std::string sha256_string(std::string_view value);

}  // namespace ygo::util
