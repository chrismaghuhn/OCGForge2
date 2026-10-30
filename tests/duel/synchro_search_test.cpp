// Card script patch 0001 remembers Synchro.Condition's material search per state. Checked on
// boards built by a puzzle script: the answers are the ones the rules give, and a duel with the
// memo switched off (Synchro.SearchKey returning nil) offers exactly the same choices.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "ocgapi_constants.h"
#include "support.hpp"

using ygo::duel::Duel;

namespace {

constexpr std::uint32_t kVisasAmritara = 821049;  // Level 8 Synchro: 1+ Tuners + 1 LIGHT non-Tuner
constexpr std::uint32_t kWaterSpirit = 487395;    // Level 1 Tuner
constexpr std::uint32_t kFlamvellGuard = 21615956;  // Level 1 Tuner
constexpr std::uint32_t kKeyMace = 1929294;       // Level 1 LIGHT
constexpr std::uint32_t kEldeen = 6367785;        // Level 3 LIGHT
constexpr std::uint32_t kTatsunoko = 55863245;    // Level 3 Tuner; once Synchro Summoned, hand Synchro
constexpr std::uint32_t kOrion = 2971090;         // Level 5 LIGHT
constexpr std::uint32_t kGradius = 10992251;      // Level 4 LIGHT

struct Board {
    std::vector<std::uint32_t> monsters;  // face-up in our Main Monster Zones, then Extra Monster Zones
    std::vector<std::uint32_t> hand;
    bool synchro_summoned = false;  // the monsters were Synchro Summoned (Tatsunoko's hand Synchro needs it)
};

// Player 0 in Main Phase 1 with Visas Amritara in the Extra Deck.
std::string puzzle(const Board& board, bool memo) {
    std::string script = memo ? "" : "Synchro.SearchKey=function() return nil end\n";
    script += "Debug.ReloadFieldBegin(DUEL_ATTACK_FIRST_TURN,5)\n";
    script += "Debug.SetPlayerInfo(0,8000,0,0)\nDebug.SetPlayerInfo(1,8000,0,0)\n";
    script += "Debug.AddCard(" + std::to_string(kVisasAmritara) + ",0,0,LOCATION_EXTRA,0,POS_FACEDOWN)\n";
    for (std::size_t i = 0; i < board.monsters.size(); ++i) {
        script += "local c" + std::to_string(i) + "=Debug.AddCard(" + std::to_string(board.monsters[i]) +
                  ",0,0,LOCATION_MZONE," + std::to_string(i) + ",POS_FACEUP_ATTACK" +
                  (board.synchro_summoned ? ",true" : "") + ")\n";
        if (board.synchro_summoned) {
            script += "Debug.PreSummon(c" + std::to_string(i) + ",SUMMON_TYPE_SYNCHRO,LOCATION_EXTRA)\n";
        }
    }
    for (std::size_t i = 0; i < board.hand.size(); ++i) {
        script += "Debug.AddCard(" + std::to_string(board.hand[i]) + ",0,0,LOCATION_HAND," + std::to_string(i) +
                  ",POS_FACEDOWN)\n";
    }
    for (std::uint32_t i = 0; i < 10; ++i) {  // something to draw
        script += "Debug.AddCard(" + std::to_string(kKeyMace) + ",0,0,LOCATION_DECK," + std::to_string(i) +
                  ",POS_FACEDOWN)\n";
        script += "Debug.AddCard(" + std::to_string(kKeyMace) + ",1,1,LOCATION_DECK," + std::to_string(i) +
                  ",POS_FACEDOWN)\n";
    }
    return script + "Debug.ReloadFieldEnd()\n";
}

ygo::duel::DuelConfig config(const Board& board, bool memo) {
    auto result = test::duel_config(1);
    result.starting_player = 0;
    result.fixture_script = puzzle(board, memo);
    return result;
}

std::vector<std::string> keys(const Duel& duel) {
    std::vector<std::string> result;
    for (const auto& candidate : duel.request().candidates) {
        result.push_back(candidate.semantic_key);
    }
    return result;
}

bool offers_visas(const Duel& duel) {
    for (const auto& candidate : duel.request().candidates) {
        if (candidate.source_card == kVisasAmritara && candidate.source_location == LOCATION_EXTRA) {
            return true;
        }
    }
    return false;
}

// Both duels from the board, played on with the same random answers: the same choices every time,
// the Synchro Summon's material selection (Synchro.Target) included.
bool same_play(const Board& board, std::uint64_t seed) {
    Duel with_memo(config(board, true));
    Duel without_memo(config(board, false));
    std::mt19937_64 rng(seed);
    for (int step = 0; step < 60 && !with_memo.finished(); ++step) {
        REQUIRE(!without_memo.finished());
        const auto offered = keys(with_memo);
        REQUIRE(offered == keys(without_memo));
        // Prefer the Synchro Summon when it is offered, so its material selection gets played.
        std::string key = offered[rng() % offered.size()];
        for (const auto& candidate : with_memo.request().candidates) {
            if (candidate.source_card == kVisasAmritara && candidate.source_location == LOCATION_EXTRA) {
                key = candidate.semantic_key;
            }
        }
        with_memo.act(key);
        without_memo.act(key);
    }
    return with_memo.finished() == without_memo.finished();
}

}  // namespace

TEST_CASE("Synchro check with many Tuners: no Level 8 from six Level 1 Tuners and a Level 1") {
    const Board board{{kWaterSpirit, kFlamvellGuard, kWaterSpirit, kFlamvellGuard, kWaterSpirit, kKeyMace}, {}};
    Duel duel(config(board, true));
    CHECK_FALSE(offers_visas(duel));
    CHECK(same_play(board, 1));
}

TEST_CASE("Synchro check with many Tuners: five Level 1 Tuners and a Level 3 make Level 8") {
    const Board board{{kWaterSpirit, kFlamvellGuard, kWaterSpirit, kFlamvellGuard, kWaterSpirit, kEldeen}, {}};
    Duel duel(config(board, true));
    CHECK(offers_visas(duel));
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        CAPTURE(seed);
        CHECK(same_play(board, seed));
    }
}

TEST_CASE("Synchro check with a hand Synchro Tuner gives the same choices with and without the memo") {
    // Tatsunoko (3) + a LIGHT monster in hand: Level 8 with Orion (5), not with Gradius (4).
    const Board fits{{kTatsunoko}, {kOrion, kGradius}, true};
    const Board misses{{kTatsunoko}, {kGradius, kGradius}, true};
    Duel duel(config(fits, true));
    CHECK(offers_visas(duel));
    Duel other(config(misses, true));
    CHECK_FALSE(offers_visas(other));
    CHECK(same_play(fits, 1));
    CHECK(same_play(misses, 1));
}
