#include <catch2/catch_test_macros.hpp>

#include "support.hpp"

TEST_CASE("decks load with main and extra deck") {
    for (const auto* deck : {&test::swordsoul(), &test::salamangreat()}) {
        CHECK(deck->main_deck.size() >= 40);
        CHECK(deck->main_deck.size() <= 60);
        CHECK(deck->extra_deck.size() <= 15);
        CHECK(deck->sha256.size() == 64);
    }
}

TEST_CASE("card_codes is sorted and unique") {
    const auto codes = ygo::core::card_codes(test::swordsoul(), test::salamangreat());
    REQUIRE_FALSE(codes.empty());
    for (std::size_t i = 1; i < codes.size(); ++i) {
        CHECK(codes[i - 1] < codes[i]);
    }
}
