#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

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

TEST_CASE("the side deck is ignored") {
    const auto path = std::filesystem::temp_directory_path() / "ocgforge_side_deck_test.ydk";
    {
        std::ofstream out(path);
        out << "#created by test\n#main\n";
        for (const auto code : test::salamangreat().main_deck) {
            out << code << "\n";
        }
        out << "#extra\n";
        for (const auto code : test::salamangreat().extra_deck) {
            out << code << "\n";
        }
        out << "!side\n14558127\n14558127\n";
    }
    const auto deck = ygo::core::load_deck(path);
    std::filesystem::remove(path);
    CHECK(deck.main_deck == test::salamangreat().main_deck);
    CHECK(deck.extra_deck == test::salamangreat().extra_deck);
}
