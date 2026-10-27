#include "ygo/core/deck.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include "ygo/util/sha256.hpp"

namespace ygo::core {
namespace {

std::string trim(const std::string& line) {
    const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    const auto first = std::find_if_not(line.begin(), line.end(), is_space);
    const auto last = std::find_if_not(line.rbegin(), line.rend(), is_space).base();
    return first < last ? std::string(first, last) : std::string();
}

}  // namespace

Deck load_deck(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open deck: " + path.string());
    }
    const std::string contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    enum class Section { None, Main, Extra, Side };
    Section section = Section::None;
    Deck deck;
    std::istringstream lines(contents);
    std::string raw;
    while (std::getline(lines, raw)) {
        const auto line = trim(raw);
        if (line.empty()) {
            continue;
        }
        if (line == "#main") {
            section = Section::Main;
            continue;
        }
        if (line == "#extra") {
            section = Section::Extra;
            continue;
        }
        if (line == "!side") {
            section = Section::Side;
            continue;
        }
        if (line.front() == '#') {
            continue;
        }
        if (section == Section::None) {
            throw std::runtime_error("deck card appears before a section in " + path.string() + ": " + line);
        }
        if (section == Section::Side) {
            throw std::runtime_error("side deck must be empty: " + path.string());
        }
        std::uint32_t code = 0;
        std::istringstream parser(line);
        parser >> code;
        if (!parser || !parser.eof()) {
            throw std::runtime_error("invalid card passcode in " + path.string() + ": " + line);
        }
        (section == Section::Extra ? deck.extra_deck : deck.main_deck).push_back(code);
    }
    if (deck.main_deck.size() < 40) {
        throw std::runtime_error("main deck has fewer than 40 cards: " + path.string());
    }
    deck.sha256 = util::sha256_string(contents);
    return deck;
}

std::vector<std::uint32_t> card_codes(const Deck& deck_a, const Deck& deck_b) {
    std::vector<std::uint32_t> codes;
    for (const auto* deck : {&deck_a, &deck_b}) {
        codes.insert(codes.end(), deck->main_deck.begin(), deck->main_deck.end());
        codes.insert(codes.end(), deck->extra_deck.begin(), deck->extra_deck.end());
    }
    std::sort(codes.begin(), codes.end());
    codes.erase(std::unique(codes.begin(), codes.end()), codes.end());
    return codes;
}

}  // namespace ygo::core
