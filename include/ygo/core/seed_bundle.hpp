#pragma once

#include <array>
#include <cstdint>

namespace ygo::core {

struct SeedBundle {
    std::array<std::uint64_t, 4> words{};

    bool operator==(const SeedBundle& other) const noexcept { return words == other.words; }
    bool operator!=(const SeedBundle& other) const noexcept { return !(*this == other); }
};

inline SeedBundle derive_seed_bundle(const std::uint64_t root_seed) noexcept {
    return {{root_seed,
             root_seed ^ 0x9e3779b97f4a7c15ULL,
             root_seed + 0x6a09e667f3bcc909ULL,
             (root_seed << 1) ^ 0xbb67ae8584caa73bULL}};
}

}  // namespace ygo::core
