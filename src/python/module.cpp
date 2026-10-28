// Python binding: ocgforge._core. Exposes VectorEnv and the encoder's feature
// names (for the Python/C++ consistency test).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <string>
#include <vector>

#include "ygo/client/client_duel.hpp"
#include "ygo/core/deck.hpp"
#include "ygo/duel/duel.hpp"
#include "ygo/encode/vector_env.hpp"
#include "ygo/protocol/message_decoder.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using ygo::encode::EncodedDecision;
using ygo::encode::VectorEnv;

namespace {

// Copies values into a numpy array that owns its memory.
template <typename T>
nb::object to_numpy(const std::vector<T>& values, std::initializer_list<std::size_t> shape) {
    auto* data = new std::vector<T>(values);
    nb::capsule owner(data, [](void* pointer) noexcept { delete static_cast<std::vector<T>*>(pointer); });
    return nb::cast(nb::ndarray<nb::numpy, T>(data->data(), shape.size(), shape.begin(), owner));
}

// A batch of decisions padded to common widths, in the layout features.collate produces
// (pad values: token_side/event_side 2, indices -1, else 0), plus per-row counts.
nb::dict to_padded(const std::vector<const EncodedDecision*>& rows) {
    namespace E = ygo::encode;
    const auto b = rows.size();
    std::size_t t = 1, c = 0, e = 0;
    for (const auto* d : rows) {
        t = std::max(t, d->token_card.size());
        c = std::max(c, d->candidate_kind.size());
        e = std::max(e, d->event_kind.size());
    }
    const auto tn = E::token_numeric_names().size();
    const auto cn = E::candidate_numeric_names().size();
    const auto en = E::event_numeric_names().size();
    const auto gn = E::global_numeric_names().size();
    // Copies row r's `values` (width w per item) into a [b, rows_max, w] buffer.
    auto pack = [&](auto member, std::size_t rows_max, std::size_t w, auto pad) {
        using T = typename std::decay_t<decltype(rows[0]->*member)>::value_type;
        std::vector<T> out(b * rows_max * w, static_cast<T>(pad));
        for (std::size_t r = 0; r < b; ++r) {
            const auto& values = rows[r]->*member;
            std::copy(values.begin(), values.end(), out.begin() + static_cast<std::ptrdiff_t>(r * rows_max * w));
        }
        return out;
    };
    nb::dict out;
    out["token_card"] = to_numpy(pack(&EncodedDecision::token_card, t, 1, 0), {b, t});
    out["token_zone"] = to_numpy(pack(&EncodedDecision::token_zone, t, 1, 0), {b, t});
    out["token_side"] = to_numpy(pack(&EncodedDecision::token_side, t, 1, 2), {b, t});
    out["token_numeric"] = to_numpy(pack(&EncodedDecision::token_numeric, t, tn, 0.0f), {b, t, tn});
    out["candidate_kind"] = to_numpy(pack(&EncodedDecision::candidate_kind, c, 1, 0), {b, c});
    out["candidate_source"] = to_numpy(pack(&EncodedDecision::candidate_source, c, 1, -1), {b, c});
    out["candidate_card"] = to_numpy(pack(&EncodedDecision::candidate_card, c, 1, 0), {b, c});
    out["candidate_zone"] = to_numpy(pack(&EncodedDecision::candidate_zone, c, 1, 0), {b, c});
    out["candidate_targets"] = to_numpy(pack(&EncodedDecision::candidate_targets, c, E::kMaxTargets, -1),
                                        {b, c, E::kMaxTargets});
    out["candidate_numeric"] = to_numpy(pack(&EncodedDecision::candidate_numeric, c, cn, 0.0f), {b, c, cn});
    out["event_kind"] = to_numpy(pack(&EncodedDecision::event_kind, e, 1, 0), {b, e});
    out["event_card"] = to_numpy(pack(&EncodedDecision::event_card, e, 1, 0), {b, e});
    out["event_side"] = to_numpy(pack(&EncodedDecision::event_side, e, 1, 2), {b, e});
    out["event_from"] = to_numpy(pack(&EncodedDecision::event_from, e, 1, 0), {b, e});
    out["event_to"] = to_numpy(pack(&EncodedDecision::event_to, e, 1, 0), {b, e});
    out["event_token"] = to_numpy(pack(&EncodedDecision::event_token, e, 1, -1), {b, e});
    out["event_numeric"] = to_numpy(pack(&EncodedDecision::event_numeric, e, en, 0.0f), {b, e, en});
    out["global_numeric"] = to_numpy(pack(&EncodedDecision::global_numeric, 1, gn, 0.0f), {b, gn});
    std::vector<std::int64_t> phase(b), decision(b), player(b), tokens(b), candidates(b), events(b);
    for (std::size_t r = 0; r < b; ++r) {
        phase[r] = rows[r]->phase;
        decision[r] = rows[r]->decision;
        player[r] = rows[r]->player;
        tokens[r] = static_cast<std::int64_t>(rows[r]->token_card.size());
        candidates[r] = static_cast<std::int64_t>(rows[r]->candidate_kind.size());
        events[r] = static_cast<std::int64_t>(rows[r]->event_kind.size());
    }
    out["phase"] = to_numpy(phase, {b});
    out["decision"] = to_numpy(decision, {b});
    out["player"] = to_numpy(player, {b});
    out["token_count"] = to_numpy(tokens, {b});
    out["candidate_count"] = to_numpy(candidates, {b});
    out["event_count"] = to_numpy(events, {b});
    return out;
}

nb::dict to_dict(const EncodedDecision& d) {
    const auto tokens = d.token_card.size();
    const auto candidates = d.candidate_kind.size();
    nb::dict out;
    out["player"] = d.player;
    out["token_card"] = to_numpy(d.token_card, {tokens});
    out["token_zone"] = to_numpy(d.token_zone, {tokens});
    out["token_side"] = to_numpy(d.token_side, {tokens});
    out["token_numeric"] = to_numpy(d.token_numeric, {tokens, ygo::encode::token_numeric_names().size()});
    out["global_numeric"] = to_numpy(d.global_numeric, {d.global_numeric.size()});
    out["phase"] = d.phase;
    out["decision"] = d.decision;
    out["candidate_kind"] = to_numpy(d.candidate_kind, {candidates});
    out["candidate_source"] = to_numpy(d.candidate_source, {candidates});
    out["candidate_card"] = to_numpy(d.candidate_card, {candidates});
    out["candidate_zone"] = to_numpy(d.candidate_zone, {candidates});
    out["candidate_targets"] = to_numpy(d.candidate_targets, {candidates, ygo::encode::kMaxTargets});
    out["candidate_numeric"] = to_numpy(d.candidate_numeric, {candidates, ygo::encode::candidate_numeric_names().size()});
    const auto events = d.event_kind.size();
    out["event_kind"] = to_numpy(d.event_kind, {events});
    out["event_card"] = to_numpy(d.event_card, {events});
    out["event_side"] = to_numpy(d.event_side, {events});
    out["event_from"] = to_numpy(d.event_from, {events});
    out["event_to"] = to_numpy(d.event_to, {events});
    out["event_token"] = to_numpy(d.event_token, {events});
    out["event_numeric"] = to_numpy(d.event_numeric, {events, ygo::encode::event_numeric_names().size()});
    return out;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "OCGForge duel environment (C++)";

    nb::class_<VectorEnv>(m, "VectorEnv",
                          "num_envs duels that always have a pending decision; finished duels restart.")
        .def("__init__",
             [](VectorEnv* self, const std::string& card_scripts_root, const std::string& card_data_tsv,
                const std::vector<std::string>& decks, const ygo::encode::CardIndex& card_index, std::size_t num_envs,
                std::uint64_t seed, std::size_t threads, std::uint64_t max_decisions) {
                 ygo::encode::VectorEnvConfig config;
                 config.card_scripts_root = card_scripts_root;
                 config.card_data_tsv = card_data_tsv;
                 for (const auto& path : decks) {
                     config.decks.push_back(ygo::core::load_deck(path));
                 }
                 config.num_envs = num_envs;
                 config.seed = seed;
                 config.threads = threads;
                 config.max_decisions = max_decisions;
                 nb::gil_scoped_release release;
                 new (self) VectorEnv(std::move(config), card_index);
             },
             "card_scripts_root"_a, "card_data_tsv"_a, "decks"_a, "card_index"_a, "num_envs"_a, "seed"_a = 0,
             "threads"_a = 0, "max_decisions"_a = 5000)
        .def("__len__", &VectorEnv::size)
        .def("observe",
             [](const VectorEnv& env) {
                 nb::list out;
                 for (std::size_t i = 0; i < env.size(); ++i) {
                     auto decision = to_dict(env.decision(i));
                     decision["duel_decision"] = env.duel_decision(i);
                     out.append(decision);
                 }
                 return out;
             },
             "The pending decision of every env, as features.py dicts (plus 'player' and 'duel_decision', the "
             "number of earlier decisions in that duel).")
        .def("step",
             [](VectorEnv& env, const std::vector<std::int64_t>& actions) {
                 ygo::encode::StepResult result;
                 {
                     nb::gil_scoped_release release;
                     result = env.step(actions);
                 }
                 return nb::make_tuple(result.done, result.winner, result.errors);
             },
             "actions"_a,
             "Play candidate actions[i] in env i. Returns (done, winner, errors) per env; winner 2 = draw, "
             "255 = the duel failed (errors[i] says why).")
        .def("recv",
             [](VectorEnv& env, std::size_t min_count) {
                 std::vector<ygo::encode::Ready> ready;
                 {
                     nb::gil_scoped_release release;
                     ready = env.recv(min_count);
                 }
                 nb::list ids, decisions, done, winner, errors;
                 for (const auto& r : ready) {
                     auto decision = to_dict(env.decision(r.env));
                     decision["duel_decision"] = env.duel_decision(r.env);
                     ids.append(r.env);
                     decisions.append(decision);
                     done.append(r.done != 0);
                     winner.append(r.winner);
                     errors.append(r.error);
                 }
                 return nb::make_tuple(ids, decisions, done, winner, errors);
             },
             "min_count"_a,
             "Wait until at least min_count envs (or all that still can) have a pending decision and hand them "
             "out: (envs, decisions, done, winner, errors). done/winner/errors describe the move that led there; "
             "a done env already holds a new duel. Do not mix with step.")
        .def("recv_batch",
             [](VectorEnv& env, std::size_t min_count) {
                 std::vector<ygo::encode::Ready> ready;
                 {
                     nb::gil_scoped_release release;
                     ready = env.recv(min_count);
                 }
                 std::vector<const EncodedDecision*> rows;
                 std::vector<std::int64_t> ids, duel_decision;
                 std::vector<std::uint8_t> done, winner;
                 nb::list errors;
                 for (const auto& r : ready) {
                     rows.push_back(&env.decision(r.env));
                     ids.push_back(static_cast<std::int64_t>(r.env));
                     duel_decision.push_back(static_cast<std::int64_t>(env.duel_decision(r.env)));
                     done.push_back(r.done);
                     winner.push_back(r.winner);
                     errors.append(r.error);
                 }
                 auto batch = to_padded(rows);
                 batch["duel_decision"] = to_numpy(duel_decision, {rows.size()});
                 return nb::make_tuple(to_numpy(ids, {rows.size()}), batch, to_numpy(done, {rows.size()}),
                                       to_numpy(winner, {rows.size()}), errors);
             },
             "min_count"_a,
             "Like recv, but the decisions come as one padded batch of numpy arrays (features.collate layout "
             "plus player, duel_decision and *_count per row): (envs, batch, done, winner, errors).")
        .def("send", &VectorEnv::send, "envs"_a, "actions"_a,
             "Play actions[i] in envs[i] on the worker threads; returns at once.")
        .def("decks", &VectorEnv::decks, "env"_a, "Deck indices of player 0 and 1 in the env's current duel.")
        .def("info",
             [](const VectorEnv& env, std::size_t i) {
                 const auto info = env.info(i);
                 nb::dict out;
                 out["busy"] = info.busy;
                 out["seed"] = info.seed;
                 out["starting_player"] = info.starting_player;
                 out["decks"] = nb::make_tuple(info.decks.first, info.decks.second);
                 out["actions"] = info.actions;
                 return out;
             },
             "env"_a, "For diagnosing hangs: seed, starting player, decks and actions of the env's duel; "
             "safe to call from another thread while step runs.")
        .def_prop_ro("duels_finished", &VectorEnv::duels_finished)
        .def_prop_ro("duel_errors", &VectorEnv::duel_errors)
        .def_prop_ro("blocked_answers", &VectorEnv::blocked_answers,
                     "Answers removed by loop blocking, summed over finished duels.")
        .def_prop_ro("loop_losses", &VectorEnv::loop_losses,
                     "Duels lost by a player who kept looping although answers were blocked.")
        .def_prop_ro("start_failures", &VectorEnv::start_failures,
                     "Duels that failed to start and were replaced; nonzero usually means a deck problem.")
        .def_prop_ro("last_start_failure", &VectorEnv::last_start_failure);

    using ygo::duel::Duel;
    nb::class_<Duel>(m, "Duel",
                     "One duel on our engine, as VectorEnv plays it: same seed, decks, starting player and answers "
                     "give the same duel, so search can try a line by replaying a duel's actions.")
        .def(
            "__init__",
            [](Duel* self, const std::string& card_scripts_root, const std::string& card_data_tsv,
               const std::string& deck0, const std::string& deck1, std::uint64_t seed, std::uint8_t starting_player,
               std::uint64_t max_decisions) {
                ygo::duel::DuelConfig config;
                config.card_scripts_root = card_scripts_root;
                config.card_data_tsv = card_data_tsv;
                config.deck0 = ygo::core::load_deck(deck0);
                config.deck1 = ygo::core::load_deck(deck1);
                config.seed = seed;
                config.starting_player = starting_player;
                config.max_decisions = max_decisions;
                new (self) Duel(std::move(config));
            },
            "card_scripts_root"_a, "card_data_tsv"_a, "deck0"_a, "deck1"_a, "seed"_a, "starting_player"_a,
            "max_decisions"_a = 5000)
        .def_static(
            "world",
            [](const std::string& card_scripts_root, const std::string& card_data_tsv, const std::string& deck0,
               const std::string& deck1, const std::string& script, std::uint64_t seed, std::uint8_t starting_player) {
                ygo::duel::DuelConfig config;
                config.card_scripts_root = card_scripts_root;
                config.card_data_tsv = card_data_tsv;
                config.deck0 = ygo::core::load_deck(deck0);
                config.deck1 = ygo::core::load_deck(deck1);
                config.fixture_script = script;
                config.seed = seed;
                config.starting_player = starting_player;
                return std::make_unique<Duel>(std::move(config));
            },
            "card_scripts_root"_a, "card_data_tsv"_a, "deck0"_a, "deck1"_a, "script"_a, "seed"_a, "starting_player"_a,
            "A duel whose field a puzzle script builds (search worlds); the decks give the decklists.")
        .def("snapshot",
             [](const Duel& d) {
                 const auto snapshot = d.snapshot();
                 nb::list cards;
                 for (const auto& c : snapshot.cards) {
                     nb::dict card;
                     card["controller"] = c.controller;
                     card["owner"] = c.owner;
                     card["location"] = c.location;
                     card["sequence"] = c.sequence;
                     card["code"] = c.code;
                     card["position"] = c.position;
                     card["public"] = c.is_public;
                     card["materials"] = c.materials;
                     card["counters"] = c.counters;
                     cards.append(card);
                 }
                 nb::dict out;
                 out["life_points"] = nb::make_tuple(snapshot.life_points[0], snapshot.life_points[1]);
                 out["cards"] = cards;
                 return out;
             },
             "Privileged: every card, hidden ones included (for building search worlds after hiding).")
        .def("act", &Duel::act, "semantic_key"_a, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("finished", &Duel::finished)
        .def_prop_ro("winner", [](const Duel& d) -> std::optional<std::uint8_t> {
            return d.result() ? std::optional(d.result()->winner) : std::nullopt;
        })
        .def_prop_ro("decision_count", &Duel::decision_count)
        .def_prop_ro("player", [](const Duel& d) { return d.request().player; })
        .def("keys", [](const Duel& d) {
            std::vector<std::string> keys;
            for (const auto& candidate : d.request().candidates) keys.push_back(candidate.semantic_key);
            return keys;
        })
        .def_prop_ro("turn", [](const Duel& d) -> std::uint32_t {
            const auto* o = d.observation();
            return o && o->globals.turn_count ? *o->globals.turn_count : 0;
        })
        .def_prop_ro("phase", [](const Duel& d) -> std::uint32_t {
            const auto* o = d.observation();
            return o && o->globals.phase ? *o->globals.phase : 0;
        }, "PHASE_* bit of the pending decision (0 if unknown).")
        .def_prop_ro("turn_player", [](const Duel& d) -> std::uint8_t {
            const auto* o = d.observation();
            return o && o->globals.turn_player ? *o->globals.turn_player : 255;
        })
        .def("encode", [](const Duel& d, const ygo::encode::CardIndex& card_index) {
            if (d.observation() == nullptr) {
                throw std::logic_error("Duel.encode needs a pending decision (the duel is over)");
            }
            return to_dict(ygo::encode::encode(*d.observation(), d.request(), card_index));
        }, "card_index"_a, "The pending decision as model input (one row, like features.row).");

    using ygo::client::ClientDuel;
    auto to_vector = [](const nb::bytes& bytes) {
        const auto* data = static_cast<const std::uint8_t*>(bytes.data());
        return std::vector<std::uint8_t>(data, data + bytes.size());
    };
    nb::class_<ClientDuel>(m, "ClientDuel",
                           "One player's side of a duel run by an EDOPro server: game messages in, responses out.")
        .def(
            "__init__",
            [](ClientDuel* self, std::uint8_t player, std::uint64_t duel_flags, const std::string& card_data_tsv,
               std::vector<std::uint32_t> main_deck) {
                new (self) ClientDuel(player, duel_flags, card_data_tsv, std::move(main_deck));
            },
            "player"_a, "duel_flags"_a = 0, "card_data_tsv"_a = "", "main_deck"_a = std::vector<std::uint32_t>{},
            "duel_flags 0 = Master Rule 2020; without card_data_tsv no observations are built.")
        .def("feed", [to_vector](ClientDuel& self, const nb::bytes& message) { return self.feed(to_vector(message)); },
             "message"_a, "One STOC_GAME_MSG payload; True when it asks this player for a decision.")
        .def_prop_ro("pending", &ClientDuel::pending)
        .def("request",
             [](const ClientDuel& self) {
                 const auto& request = self.request();
                 nb::dict out;
                 out["kind"] = ygo::protocol::decision_kind_name(request.kind);
                 out["player"] = request.player;
                 std::vector<std::string> keys;
                 for (const auto& candidate : request.candidates) {
                     keys.push_back(candidate.semantic_key);
                 }
                 out["candidates"] = keys;
                 return out;
             })
        .def(
            "encode",
            [](const ClientDuel& self, const ygo::encode::CardIndex& card_index) {
                if (self.observation() == nullptr) {
                    throw std::logic_error("ClientDuel.encode needs a pending question and card data");
                }
                return to_dict(ygo::encode::encode(*self.observation(), self.request(), card_index));
            },
            "card_index"_a, "The pending question as model input (one row, like features.row).")
        .def("act",
             [](ClientDuel& self, const std::string& key) -> std::optional<nb::bytes> {
                 auto response = self.act(key);
                 if (!response) {
                     return std::nullopt;
                 }
                 return nb::bytes(reinterpret_cast<const char*>(response->data()), response->size());
             },
             "semantic_key"_a, "The response bytes, or None when a multi-step selection continues.")
        .def_prop_ro("finished", &ClientDuel::finished)
        .def_prop_ro("winner", &ClientDuel::winner)
        .def_prop_ro("messages", &ClientDuel::messages)
        .def_prop_ro("blocked_answers", &ClientDuel::blocked_answers)
        .def_prop_ro("skipped", &ClientDuel::skipped);

    m.def("decision_names", &ygo::encode::decision_names);
    m.def("zone_names", &ygo::encode::zone_names);
    m.def("phase_names", &ygo::encode::phase_names);
    m.def("candidate_kind_names", &ygo::encode::candidate_kind_names);
    m.def("token_numeric_names", &ygo::encode::token_numeric_names);
    m.def("global_numeric_names", &ygo::encode::global_numeric_names);
    m.def("candidate_numeric_names", &ygo::encode::candidate_numeric_names);
    m.def("event_kind_names", &ygo::encode::event_kind_names);
    m.def("event_numeric_names", &ygo::encode::event_numeric_names);
    m.attr("MAX_TARGETS") = ygo::encode::kMaxTargets;
    m.attr("MAX_EVENTS") = ygo::encode::kMaxEvents;
}
