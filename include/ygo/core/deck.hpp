#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ygo::core {

struct Deck {
    std::vector<std::uint32_t> main_deck;
    std::vector<std::uint32_t> extra_deck;
    std::string sha256;
};

// Reads a .ydk file (#main / #extra; the !side section is ignored).
Deck load_deck(const std::filesystem::path& path);

// Sorted, de-duplicated passcodes of every card in the given decks.
std::vector<std::uint32_t> card_codes(const Deck& deck_a, const Deck& deck_b);

}  // namespace ygo::core
