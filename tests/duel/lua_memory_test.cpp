#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <memory>

#include "support.hpp"
#include "ygo/core/core_error.hpp"
#include "ygo/core/core_host.hpp"

using Catch::Matchers::ContainsSubstring;
using ygo::core::CoreHost;
using ygo::core::CoreHostConfig;
using ygo::duel::Duel;

namespace {

CoreHostConfig host_config(std::uint64_t limit) {
    CoreHostConfig config;
    config.card_scripts_root = OCGFORGE_CACHE_DIR "/cardscripts";
    config.card_data_tsv = OCGFORGE_CACHE_DIR "/card_data.tsv";
    config.seed = ygo::core::derive_seed_bundle(1);
    config.lua_memory_limit = limit;
    return config;
}

// About 16 MB of Lua table, kept alive by a global.
constexpr const char* kHog = "hog = {} for i = 1, 1000000 do hog[i] = i end";

}  // namespace

TEST_CASE("a tiny Lua memory budget lets the state be built, then fails the duel cleanly") {
    // The limit is armed only after OCG_CreateDuel built the Lua state, so creating the state
    // works; loading the first global script then runs out of budget and the host throws.
    try {
        CoreHost host(host_config(1));
        FAIL("a 1-byte budget should not fit constant.lua");
    } catch (const ygo::core::CoreError& error) {
        CHECK_THAT(error.what(), ContainsSubstring("Lua memory budget exceeded"));
    }
    // And a Duel (decks, start, first decision) built on it fails the same way.
    auto config = test::duel_config(3);
    config.lua_memory_limit = 1;
    CHECK_THROWS_WITH(Duel(config), ContainsSubstring("Lua memory budget exceeded"));
}

TEST_CASE("a script allocating past the Lua memory budget fails; without a budget it runs") {
    {
        CoreHost host(host_config(8u << 20));
        CHECK_THROWS_WITH(host.load_script_text("hog.lua", kHog), ContainsSubstring("Lua memory budget exceeded"));
        CHECK(host.lua_memory().peak <= (8u << 20));
    }
    CoreHost host(host_config(0));
    host.load_script_text("hog.lua", kHog);
    CHECK(host.lua_memory().used > (8u << 20));
}

TEST_CASE("Lua memory is reported, survives a clone and stays below the default budget") {
    Duel duel(test::duel_config(5));
    const auto start = duel.lua_memory();
    CHECK(start.used > 0);
    CHECK(start.peak >= start.used);
    CHECK(start.heap_extent >= start.used);
    CHECK(start.peak < ygo::core::CoreHostConfig::kDefaultLuaMemoryLimit);
    for (int i = 0; i < 50 && !duel.finished(); ++i) {
        duel.act(duel.request().candidates.front().semantic_key);
    }
    REQUIRE_FALSE(duel.finished());
    const auto copy = duel.clone();
    const auto original = duel.lua_memory();
    const auto cloned = copy->lua_memory();
    CHECK(cloned.used == original.used);  // the clone copies the whole Lua state, counters included
    CHECK(cloned.peak == original.peak);
    CHECK(cloned.heap_extent == original.heap_extent);
}
