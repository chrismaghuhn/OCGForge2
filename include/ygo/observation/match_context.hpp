#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ygo::observation {

struct MatchKnowledgeConfig {
    bool own_decklist_known = true;
    bool opponent_decklist_known = false;
};

struct StaticDeckContext {
    bool known = false;
    std::vector<std::uint32_t> main_deck;
    std::vector<std::uint32_t> extra_deck;
};

struct MatchContext {
    std::uint8_t perspective_player = 0;
    std::uint64_t duel_flags = 0;
    MatchKnowledgeConfig knowledge;
    StaticDeckContext own_deck;
    StaticDeckContext opponent_deck;
};

}  // namespace ygo::observation
