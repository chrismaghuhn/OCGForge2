// ocgcore patch 0009: the end-of-phase reset walked the phase-reset effects with a saved
// iterator, while removing a grant also removes the effects it granted. With
// Pendulum.GrantAdditionalPendulumSummon both reset at the End Phase, so the saved node could be
// freed: a use-after-free in field::reset_phase (a segfault in long runs, an error under ASan).

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>

#include "support.hpp"

using ygo::duel::Duel;

namespace {

constexpr std::uint32_t kScaleSurveyor = 42382265;  // D/D Scale Surveyor, a Pendulum Monster
constexpr std::uint32_t kKeyMace = 1929294;

// Player 0 in Main Phase 1 with Pendulum cards in both players' Pendulum Zones, granted an extra
// Pendulum Summon until the End Phase (as Solfachord Happiness or D/D/D Sky King Zeus Ragnarok
// do), plus `filler` other effects that reset then: their number changes the set's layout and so
// the order the reset visits the grants and the effects they granted.
std::string puzzle(int filler) {
    std::string script = "Debug.ReloadFieldBegin(DUEL_ATTACK_FIRST_TURN,5)\n";
    script += "Debug.SetPlayerInfo(0,8000,0,0)\nDebug.SetPlayerInfo(1,8000,0,0)\n";
    for (int p = 0; p < 2; ++p) {
        const auto player = std::to_string(p);
        for (int seq = 0; seq < 2; ++seq) {
            script += "local pz" + player + std::to_string(seq) + "=Debug.AddCard(" + std::to_string(kScaleSurveyor) +
                      "," + player + "," + player + ",LOCATION_PZONE," + std::to_string(seq) + ",POS_FACEUP)\n";
        }
        for (int i = 0; i < 10; ++i) {  // something to draw
            script += "Debug.AddCard(" + std::to_string(kKeyMace) + "," + player + "," + player +
                      ",LOCATION_DECK," + std::to_string(i) + ",POS_FACEDOWN)\n";
        }
    }
    // Two grants from different cards (the second one's owner id differs, or it is refused).
    script += "Pendulum.GrantAdditionalPendulumSummon(pz00,nil,0,LOCATION_HAND|LOCATION_EXTRA,1163,1163,100)\n";
    script += "Pendulum.GrantAdditionalPendulumSummon(pz01,nil,0,LOCATION_HAND|LOCATION_EXTRA,1163,1163,101)\n";
    script += "for i=1," + std::to_string(filler) + " do\n"
              "  local e=Effect.CreateEffect(pz00)\n"
              "  e:SetType(EFFECT_TYPE_SINGLE)\n"
              "  e:SetCode(EFFECT_UPDATE_DEFENSE)\n"
              "  e:SetValue(i)\n"
              "  e:SetReset(RESET_PHASE|PHASE_END)\n"
              "  pz10:RegisterEffect(e)\n"
              "end\n";
    return script + "Debug.ReloadFieldEnd()\n";
}

ygo::duel::DuelConfig config(int filler) {
    auto result = test::duel_config(1);
    result.starting_player = 0;
    result.fixture_script = puzzle(filler);
    return result;
}

bool offers(const Duel& duel, const std::string& key) {
    for (const auto& candidate : duel.request().candidates) {
        if (candidate.semantic_key == key) {
            return true;
        }
    }
    return false;
}

std::uint32_t turn(const Duel& duel) {
    const auto* observation = duel.observation();
    return observation && observation->globals.turn_count ? *observation->globals.turn_count : 0;
}

}  // namespace

TEST_CASE("the End Phase resets granted extra Pendulum Summons together with their grants") {
    for (int filler = 0; filler <= 24; ++filler) {
        CAPTURE(filler);
        Duel duel(config(filler));
        // The duel starts in the Standby Phase, where the Pendulum cards offer chains: pass them.
        for (int step = 0; step < 10 && offers(duel, "chain.pass"); ++step) {
            duel.act("chain.pass");
        }
        REQUIRE(offers(duel, "idle.7.0"));  // to the End Phase
        duel.act("idle.7.0");
        // Pass (or take the first choice) until the opponent's turn: the End Phase reset ran.
        for (int step = 0; step < 40 && !duel.finished() && turn(duel) < 2; ++step) {
            duel.act(offers(duel, "chain.pass") ? std::string("chain.pass")
                                                : duel.request().candidates.front().semantic_key);
        }
        REQUIRE_FALSE(duel.finished());
        CHECK(turn(duel) == 2);
    }
}
