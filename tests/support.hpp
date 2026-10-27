#pragma once

#include <cstdint>

#include "ygo/core/deck.hpp"
#include "ygo/duel/duel.hpp"

namespace test {

inline const ygo::core::Deck& swordsoul() {
    static const auto deck = ygo::core::load_deck(OCGFORGE_SOURCE_DIR "/decks/swordsoul_tenyi.ydk");
    return deck;
}

inline const ygo::core::Deck& salamangreat() {
    static const auto deck = ygo::core::load_deck(OCGFORGE_SOURCE_DIR "/decks/salamangreat.ydk");
    return deck;
}

inline ygo::duel::DuelConfig duel_config(std::uint64_t seed) {
    ygo::duel::DuelConfig config;
    config.card_scripts_root = OCGFORGE_CACHE_DIR "/cardscripts";
    config.card_data_tsv = OCGFORGE_CACHE_DIR "/card_data.tsv";
    config.deck0 = swordsoul();
    config.deck1 = salamangreat();
    config.seed = seed;
    config.starting_player = static_cast<std::uint8_t>(seed % 2);
    return config;
}

}  // namespace test
