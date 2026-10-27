// Regression test for the "Hiita loop": the two card lists of
// MSG_SELECT_UNSELECT_CARD used to be labelled the wrong way round, so the bot
// unselected cards it meant to select and the engine asked the same question
// forever.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

#include "ocgapi_constants.h"
#include "ygo/protocol/message_decoder.hpp"

using namespace ygo::protocol;

namespace {

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void put_card(std::vector<std::uint8_t>& out, std::uint32_t code, std::uint8_t location, std::uint32_t sequence) {
    put_u32(out, code);
    out.push_back(0);  // controller
    out.push_back(location);
    put_u32(out, sequence);
    put_u32(out, POS_FACEUP_ATTACK);
}

// One card the player can still select, one card that is already selected.
std::vector<std::uint8_t> select_unselect_message() {
    std::vector<std::uint8_t> frame{MSG_SELECT_UNSELECT_CARD, 0, /*finishable*/ 1, /*cancelable*/ 0};
    put_u32(frame, 1);  // min
    put_u32(frame, 2);  // max
    put_u32(frame, 1);  // select list
    put_card(frame, 1111, LOCATION_MZONE, 0);
    put_u32(frame, 1);  // unselect list
    put_card(frame, 2222, LOCATION_MZONE, 1);

    std::vector<std::uint8_t> message;
    put_u32(message, static_cast<std::uint32_t>(frame.size()));
    std::copy(frame.begin(), frame.end(), std::back_inserter(message));
    return message;
}

}  // namespace

TEST_CASE("select-unselect: first list is Select, second list is Unselect") {
    const auto decoded = decode_messages(select_unselect_message());
    REQUIRE(decoded.decisions.size() == 1);
    const auto& candidates = decoded.decisions.front().candidates;
    REQUIRE(candidates.size() == 3);

    CHECK(candidates[0].source_card == 1111);
    CHECK(candidates[0].card_selection_operation == CardSelectionOperation::Select);

    CHECK(candidates[1].source_card == 2222);
    CHECK(candidates[1].card_selection_operation == CardSelectionOperation::Unselect);

    CHECK(candidates[2].action_kind == ActionKind::Finish);
    CHECK(candidates[2].card_selection_operation == CardSelectionOperation::None);
}

TEST_CASE("select-unselect: responses pick the card by its combined index") {
    const auto decoded = decode_messages(select_unselect_message());
    const auto& candidates = decoded.decisions.front().candidates;
    // Response is [count=1][index], index counts across both lists.
    CHECK(candidates[0].exact_response_bytes == std::vector<std::uint8_t>{1, 0, 0, 0, 0, 0, 0, 0});
    CHECK(candidates[1].exact_response_bytes == std::vector<std::uint8_t>{1, 0, 0, 0, 1, 0, 0, 0});
    CHECK(candidates[2].exact_response_bytes == std::vector<std::uint8_t>{0xff, 0xff, 0xff, 0xff});
}
