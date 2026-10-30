#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "../../src/core/script_reader.hpp"
#include "support.hpp"

using ygo::core::detail::ScriptCache;

TEST_CASE("the script cache keeps each script once, as bytecode") {
    const auto cache = ScriptCache::shared(OCGFORGE_CACHE_DIR "/cardscripts");
    CHECK(cache == ScriptCache::shared(OCGFORGE_CACHE_DIR "/cardscripts"));

    const auto& utility = cache->get("utility.lua");
    REQUIRE(utility.found);
    CHECK(utility.binary);
    CHECK(utility.bytes.rfind("\x1bLua", 0) == 0);
    CHECK(&cache->get("utility.lua") == &utility);

    const auto& missing = cache->get("c1.lua");  // no such card script: remembered, not reread
    CHECK_FALSE(missing.found);
    CHECK(&cache->get("c1.lua") == &missing);

    // Threads asking for the same script all get the one entry.
    std::vector<const ScriptCache::Entry*> seen(8);
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < seen.size(); ++i) {
        threads.emplace_back([&, i] { seen[i] = &cache->get("proc_normal.lua"); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto* entry : seen) {
        CHECK(entry == seen.front());
    }
}

TEST_CASE("a script that does not compile stays source and still fails in the duel") {
    const auto root = std::filesystem::temp_directory_path() / "ocgforge_script_cache_test";
    std::filesystem::create_directories(root);
    {
        std::ofstream(root / "broken.lua") << "local x = = 1\n";
    }
    ScriptCache cache(root);
    const auto& broken = cache.get("broken.lua");
    CHECK(broken.found);
    CHECK_FALSE(broken.binary);
    CHECK(broken.bytes == "local x = = 1\n");
    std::filesystem::remove_all(root);
}

TEST_CASE("duels started from cached bytecode play like each other") {
    // Same seed, same actions, same decisions; the second duel takes every script from the cache.
    ygo::duel::Duel first(test::duel_config(11));
    ygo::duel::Duel second(test::duel_config(11));
    for (int step = 0; step < 200 && !first.finished(); ++step) {
        REQUIRE_FALSE(second.finished());
        REQUIRE(first.request().candidates.size() == second.request().candidates.size());
        const auto key = first.request().candidates[static_cast<std::size_t>(step) %
                                                     first.request().candidates.size()].semantic_key;
        REQUIRE(second.request().candidates[static_cast<std::size_t>(step) %
                                            second.request().candidates.size()].semantic_key == key);
        first.act(key);
        second.act(key);
    }
    CHECK(first.finished() == second.finished());
}
