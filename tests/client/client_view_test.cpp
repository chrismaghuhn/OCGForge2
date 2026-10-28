// A ClientDuel fed what an EDOPro server sends one player (the engine's messages plus the server's
// card data refreshes, emulated after gframe/generic_duel.cpp) builds the same model input as the
// training engine for every question of that player.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "ocgapi_constants.h"
#include "ocgapi_types.h"
#include "support.hpp"
#include "ygo/client/client_duel.hpp"
#include "ygo/core/core_host.hpp"
#include "ygo/encode/encoder.hpp"
#include "ygo/observation/decision_integration.hpp"
#include "ygo/observation/observation_builder.hpp"
#include "ygo/observation/observation_session.hpp"
#include "ygo/protocol/continuation.hpp"
#include "ygo/protocol/message_decoder.hpp"

namespace {

using Bytes = std::vector<std::uint8_t>;

std::vector<Bytes> split(const Bytes& stream) {
    std::vector<Bytes> messages;
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

std::uint32_t u32_at(const Bytes& b, std::size_t at) {
    return static_cast<std::uint32_t>(b[at]) | b[at + 1] << 8 | b[at + 2] << 16 | static_cast<std::uint32_t>(b[at + 3]) << 24;
}

// One card's query records (u16 size, u32 flag, payload ... QUERY_END) from `at`, with what the
// server leaves out for the other player (core_utils.cpp Query::GenerateBuffer for the public
// buffer): a card that is neither public nor face up loses its private properties.
std::size_t public_card(const Bytes& in, std::size_t at, Bytes& out) {
    constexpr std::uint32_t kPrivate = QUERY_CODE | QUERY_ALIAS | QUERY_TYPE | QUERY_LEVEL | QUERY_RANK |
                                       QUERY_ATTRIBUTE | QUERY_RACE | QUERY_ATTACK | QUERY_DEFENSE |
                                       QUERY_BASE_ATTACK | QUERY_BASE_DEFENSE | QUERY_STATUS | QUERY_LSCALE |
                                       QUERY_RSCALE | QUERY_LINK;
    if (in[at] == 0 && in[at + 1] == 0) {
        out.insert(out.end(), {0, 0});
        return at + 2;
    }
    bool visible = false;
    std::vector<std::pair<std::size_t, std::size_t>> records;
    for (std::size_t i = at;;) {
        const std::size_t size = in[i] | in[i + 1] << 8;
        const auto flag = u32_at(in, i + 2);
        records.emplace_back(i, size + 2);
        if (flag == QUERY_POSITION && (u32_at(in, i + 6) & POS_FACEUP)) visible = true;
        if (flag == QUERY_IS_PUBLIC && in[i + 6] != 0) visible = true;
        i += size + 2;
        if (flag == QUERY_END) {
            for (const auto& [start, length] : records) {
                if (visible || (u32_at(in, start + 2) & kPrivate) == 0) {
                    out.insert(out.end(), in.begin() + static_cast<std::ptrdiff_t>(start),
                               in.begin() + static_cast<std::ptrdiff_t>(start + length));
                }
            }
            return i;
        }
    }
}

// The refreshes generic_duel.cpp sends around each engine message (default query flags there),
// as player `me` receives them.
class ServerRefreshes {
public:
    ServerRefreshes(const ygo::core::CoreHost& host, std::uint8_t me) : host_(host), me_(me) {}

    std::vector<Bytes> before(const Bytes& message) const {
        std::vector<Bytes> out;
        switch (message[0]) {
        case MSG_SELECT_BATTLECMD:
        case MSG_SELECT_IDLECMD:
            for (std::uint8_t p = 0; p < 2; ++p) {
                out.push_back(location(p, LOCATION_MZONE, 0x3981fff));
                out.push_back(location(p, LOCATION_SZONE, 0x3f81fff));
                out.push_back(location(p, LOCATION_HAND, 0x3781fff));
            }
            break;
        case MSG_SELECT_CHAIN:
        case MSG_NEW_TURN:
            for (std::uint8_t p = 0; p < 2; ++p) {
                out.push_back(location(p, LOCATION_MZONE, 0x3981fff));
                out.push_back(location(p, LOCATION_SZONE, 0x3f81fff));
            }
            break;
        default:
            break;
        }
        return out;
    }

    std::vector<Bytes> after(const Bytes& message) const {
        std::vector<Bytes> out;
        const auto type = message[0];
        switch (type) {
        case MSG_SHUFFLE_HAND:
        case MSG_DRAW:
            out.push_back(location(message[1], LOCATION_HAND, 0x3781fff));
            break;
        case MSG_SHUFFLE_EXTRA:
            out.push_back(location(message[1], LOCATION_EXTRA, 0x381fff));
            break;
        case MSG_SWAP_GRAVE_DECK:
            out.push_back(location(message[1], LOCATION_GRAVE, 0x381fff));
            break;
        case MSG_SHUFFLE_SET_CARD:
            for (std::uint8_t p = 0; p < 2; ++p) {
                out.push_back(location(p, message[1], 0x3181fff));
            }
            break;
        case MSG_DAMAGE_STEP_START:
        case MSG_DAMAGE_STEP_END:
        case MSG_SUMMONED:
        case MSG_SPSUMMONED:
        case MSG_FLIPSUMMONED:
        case MSG_NEW_PHASE:
        case MSG_CHAINED:
        case MSG_CHAIN_SOLVED:
        case MSG_CHAIN_END:
            for (std::uint8_t p = 0; p < 2; ++p) {
                out.push_back(location(p, LOCATION_MZONE, 0x3981fff));
                if (type != MSG_DAMAGE_STEP_START && type != MSG_DAMAGE_STEP_END) {
                    out.push_back(location(p, LOCATION_SZONE, 0x3f81fff));
                }
                if (type == MSG_NEW_PHASE || type == MSG_CHAINED || type == MSG_CHAIN_END) {
                    out.push_back(location(p, LOCATION_HAND, 0x3781fff));
                }
            }
            break;
        case MSG_MOVE: {
            const auto previous_controller = message[5], previous_location = message[6];
            const auto controller = message[15], location_ = message[16];
            const auto sequence = message[17];
            if (location_ != 0 && (location_ & LOCATION_OVERLAY) == 0 &&
                (location_ != previous_location || controller != previous_controller)) {
                out.push_back(single(controller, location_, sequence));
            }
            break;
        }
        case MSG_POS_CHANGE:
            if ((message[8] & POS_FACEDOWN) && (message[9] & POS_FACEUP)) {
                out.push_back(single(message[5], message[6], message[7]));
            }
            break;
        case MSG_SWAP:
            out.push_back(single(message[5], message[6], message[7]));
            out.push_back(single(message[19], message[20], message[21]));
            break;
        default:
            break;
        }
        return out;
    }

    Bytes location(std::uint8_t player, std::uint32_t location, std::uint32_t flags) const {
        OCG_QueryInfo info{};
        info.flags = flags;
        info.con = player;
        info.loc = location;
        Bytes out = {MSG_UPDATE_DATA, player, static_cast<std::uint8_t>(location)};
        const auto data = host_.query_location(info);
        if (player == me_ || data.size() < 4) {
            out.insert(out.end(), data.begin(), data.end());
            return out;
        }
        Bytes cards;
        for (std::size_t at = 4; at < data.size();) {
            at = public_card(data, at, cards);
        }
        const auto size = static_cast<std::uint32_t>(cards.size());
        out.insert(out.end(), {static_cast<std::uint8_t>(size), static_cast<std::uint8_t>(size >> 8),
                               static_cast<std::uint8_t>(size >> 16), static_cast<std::uint8_t>(size >> 24)});
        out.insert(out.end(), cards.begin(), cards.end());
        return out;
    }

    Bytes single(std::uint8_t player, std::uint32_t location, std::uint32_t sequence) const {
        OCG_QueryInfo info{};
        info.flags = 0x3f81fff;
        info.con = player;
        info.loc = location;
        info.seq = sequence;
        Bytes out = {MSG_UPDATE_CARD, player, static_cast<std::uint8_t>(location), static_cast<std::uint8_t>(sequence)};
        const auto data = host_.query(info);
        if (player == me_ || data.empty()) {
            out.insert(out.end(), data.begin(), data.end());
        } else {
            public_card(data, 0, out);
        }
        return out;
    }

private:
    const ygo::core::CoreHost& host_;
    std::uint8_t me_;
};

Bytes edopro_start(const ygo::core::CoreHost& host, std::uint8_t player) {
    Bytes out = {MSG_START, player};
    for (const std::uint32_t lp : {8000u, 8000u}) {
        for (int shift = 0; shift < 32; shift += 8) out.push_back(static_cast<std::uint8_t>(lp >> shift));
    }
    for (std::uint8_t p = 0; p < 2; ++p) {
        for (const auto location : {LOCATION_DECK, LOCATION_EXTRA}) {
            const auto count = host.query_count(p, location);
            out.push_back(static_cast<std::uint8_t>(count));
            out.push_back(static_cast<std::uint8_t>(count >> 8));
        }
    }
    return out;
}

}  // namespace

TEST_CASE("ClientDuel builds the training engine's model input from what EDOPro sends") {
    std::map<std::string, int> mismatches;
    int compared = 0;
    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
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

        ygo::encode::CardIndex index;
        for (const auto code : ygo::core::card_codes(config.deck0, config.deck1)) {
            index.emplace(code, static_cast<std::int64_t>(index.size()) + 2);
        }
        const std::uint8_t me = static_cast<std::uint8_t>(seed % 2);
        ygo::client::ClientDuel client(me, DUEL_MODE_MR5, config.card_data_tsv,
                                       me == 0 ? config.deck0.main_deck : config.deck1.main_deck);
        ygo::observation::ObservationSession session(me, DUEL_MODE_MR5);
        ServerRefreshes server(host, me);
        client.feed(edopro_start(host, me));
        for (std::uint8_t p = 0; p < 2; ++p) {
            client.feed(server.location(p, LOCATION_EXTRA, 0x381fff));
        }
        REQUIRE(client.skipped().empty());
        std::mt19937_64 rng(seed);
        std::uint64_t step = 0;
        bool over = false;
        for (int decisions = 0; decisions < 400 && !over;) {
            const auto stream = host.process().message;
            for (const auto& message : split(stream)) {
                Bytes framed = {static_cast<std::uint8_t>(message.size()), static_cast<std::uint8_t>(message.size() >> 8),
                                static_cast<std::uint8_t>(message.size() >> 16), static_cast<std::uint8_t>(message.size() >> 24)};
                framed.insert(framed.end(), message.begin(), message.end());
                session.ingest(framed, step++);
                for (const auto& refresh : server.before(message)) {
                    client.feed(refresh);
                }
                const bool asked = client.feed(message);
                for (const auto& refresh : server.after(message)) {
                    client.feed(refresh);
                }
                over = over || message[0] == MSG_WIN;
                const auto decoded = ygo::protocol::decode_messages(framed);
                if (!decoded.interactive) {
                    continue;
                }
                const auto& request = decoded.decisions.front();
                std::uniform_int_distribution<std::size_t> pick(0, request.candidates.size() - 1);
                if (!asked) {  // the other player: answer for them directly
                    const auto& candidate = request.candidates[pick(rng)];
                    if (!request.continuation) {
                        host.submit_response(candidate.exact_response_bytes);
                    } else {
                        auto r = request;
                        auto t = ygo::protocol::apply_continuation_action(r, candidate.semantic_key);
                        while (!t.terminal) {
                            std::uniform_int_distribution<std::size_t> next(0, t.request.candidates.size() - 1);
                            r = t.request;
                            t = ygo::protocol::apply_continuation_action(r, r.candidates[next(rng)].semantic_key);
                        }
                        host.submit_response(t.engine_response);
                    }
                    ++decisions;
                    continue;
                }
                std::optional<Bytes> response;
                while (!response) {
                    REQUIRE(client.observation() != nullptr);
                    // The engine's own observation of the same question.
                    ygo::observation::ObservationBuildConfig build;
                    build.visible_events = session.visible_events();
                    build.event_summary = session.summary();
                    build.detail = ygo::observation::ObservationDetail::ModelInput;
                    build.knowledge.own_decklist_known = true;
                    build.own_deck.known = true;
                    build.own_deck.main_deck = me == 0 ? config.deck0.main_deck : config.deck1.main_deck;
                    auto engine = ygo::observation::build_player_observation(host, me, build);
                    ygo::observation::attach_decision_context(engine, client.request());
                    const auto want = ygo::encode::encode(engine, client.request(), index);
                    const auto got = ygo::encode::encode(*client.observation(), client.request(), index);
                    ++compared;
                    if (want.token_card != got.token_card) ++mismatches["token_card"];
                    if (want.token_zone != got.token_zone) ++mismatches["token_zone"];
                    if (want.token_numeric != got.token_numeric) {
                        ++mismatches["token_numeric"];
                        const auto& names = ygo::encode::token_numeric_names();
                        for (std::size_t i = 0; i < want.token_numeric.size() && i < got.token_numeric.size(); ++i) {
                            if (want.token_numeric[i] != got.token_numeric[i]) {
                                ++mismatches["  column " + names[i % names.size()]];
                            }
                        }
                    }
                    if (want.global_numeric != got.global_numeric) ++mismatches["global_numeric"];
                    if (want.candidate_source != got.candidate_source) ++mismatches["candidate_source"];
                    if (want.candidate_numeric != got.candidate_numeric) ++mismatches["candidate_numeric"];
                    const auto& candidates = client.request().candidates;
                    response = client.act(candidates[pick(rng) % candidates.size()].semantic_key);
                }
                host.submit_response(*response);
                ++decisions;
            }
        }
    }
    for (const auto& [field, count] : mismatches) {
        std::printf("mismatch %s: %d of %d\n", field.c_str(), count, compared);
    }
    CHECK(compared > 100);
    CHECK(mismatches.empty());
}
