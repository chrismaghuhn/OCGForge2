// ClientDuel plays whole duels when fed an engine's raw message stream, the way an EDOPro server
// would feed it, and its responses are accepted by that engine.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <vector>

#include "ocgapi_constants.h"
#include "support.hpp"
#include "ygo/client/client_duel.hpp"
#include "ygo/core/core_host.hpp"
#include "ygo/protocol/message_decoder.hpp"

namespace {

std::vector<std::vector<std::uint8_t>> split(const std::vector<std::uint8_t>& stream) {
    std::vector<std::vector<std::uint8_t>> messages;
    for (std::size_t offset = 0; offset + 4 <= stream.size();) {
        const auto size = static_cast<std::uint32_t>(stream[offset]) | stream[offset + 1] << 8 |
                          stream[offset + 2] << 16 | stream[offset + 3] << 24;
        offset += 4;
        messages.emplace_back(stream.begin() + static_cast<std::ptrdiff_t>(offset),
                              stream.begin() + static_cast<std::ptrdiff_t>(offset + size));
        offset += size;
    }
    return messages;
}

}  // namespace

TEST_CASE("ClientDuel answers every question of an engine's message stream") {
    int finished = 0;
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        const auto config = test::duel_config(seed);
        ygo::core::CoreHostConfig host_config;
        host_config.card_scripts_root = config.card_scripts_root;
        host_config.card_data_tsv = config.card_data_tsv;
        host_config.seed = ygo::core::derive_seed_bundle(seed);
        host_config.duel_flags = DUEL_MODE_MR5;
        host_config.starting_player = config.starting_player;
        host_config.required_script_codes = ygo::core::card_codes(config.deck0, config.deck1);
        ygo::core::CoreHost host(host_config);
        host.load_deck(0, config.deck0);
        host.load_deck(1, config.deck1);
        host.start_duel();

        ygo::client::ClientDuel clients[2] = {ygo::client::ClientDuel(0), ygo::client::ClientDuel(1)};
        std::mt19937_64 rng(seed);
        bool over = false;
        for (int decisions = 0; decisions < 3000 && !over;) {
            const auto stream = host.process().message;
            for (const auto& message : split(stream)) {
                for (auto& client : clients) {
                    if (!client.feed(message)) {
                        continue;
                    }
                    std::optional<std::vector<std::uint8_t>> response;
                    while (!response) {
                        const auto& candidates = client.request().candidates;
                        std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
                        response = client.act(candidates[pick(rng)].semantic_key);
                    }
                    host.submit_response(*response);
                    ++decisions;
                }
                over = over || clients[0].finished();
            }
        }
        CAPTURE(seed);
        if (over) {
            ++finished;
            CHECK(clients[0].winner() == clients[1].winner());
        }
        for (const auto& client : clients) {
            for (const auto& skipped : client.skipped()) {
                CAPTURE(skipped);
                CHECK(skipped.find("question for the other player") != std::string::npos);
            }
        }
    }
    CHECK(finished >= 15);
}

TEST_CASE("ClientDuel stops offering an answer given 8 times to the same question in the same state") {
    const auto config = test::duel_config(3);
    ygo::core::CoreHostConfig host_config;
    host_config.card_scripts_root = config.card_scripts_root;
    host_config.card_data_tsv = config.card_data_tsv;
    host_config.seed = ygo::core::derive_seed_bundle(3);
    host_config.duel_flags = DUEL_MODE_MR5;
    host_config.starting_player = 0;
    host_config.required_script_codes = ygo::core::card_codes(config.deck0, config.deck1);
    ygo::core::CoreHost host(host_config);
    host.load_deck(0, config.deck0);
    host.load_deck(1, config.deck1);
    host.start_duel();
    ygo::client::ClientDuel client(0, DUEL_MODE_MR5, config.card_data_tsv, config.deck0.main_deck);
    std::vector<std::uint8_t> idle;
    while (idle.empty()) {
        for (const auto& message : split(host.process().message)) {
            if (client.feed(message) && message[0] == MSG_SELECT_IDLECMD) {
                idle = message;
            } else if (client.pending()) {
                host.submit_response(*client.act(client.request().candidates.front().semantic_key));
            }
        }
    }
    const auto looped = client.request().candidates.front().semantic_key;
    const auto offered = client.request().candidates.size();
    for (int answer = 1; answer <= 8; ++answer) {
        (void)client.act(looped);  // the server would ask the same again, nothing changed
        REQUIRE(client.feed(idle));
    }
    CHECK(client.request().candidates.size() == offered - 1);
    for (const auto& candidate : client.request().candidates) {
        CHECK(candidate.semantic_key != looped);
    }
    CHECK(client.blocked_answers() == 1);
}
