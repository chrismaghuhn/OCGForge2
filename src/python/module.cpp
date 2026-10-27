// Python binding: ocgforge._core. Exposes VectorEnv and the encoder's feature
// names (for the Python/C++ consistency test).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <vector>

#include "ygo/encode/vector_env.hpp"

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
        .def_prop_ro("start_failures", &VectorEnv::start_failures,
                     "Duels that failed to start and were replaced; nonzero usually means a deck problem.")
        .def_prop_ro("last_start_failure", &VectorEnv::last_start_failure);

    m.def("decision_names", &ygo::encode::decision_names);
    m.def("zone_names", &ygo::encode::zone_names);
    m.def("phase_names", &ygo::encode::phase_names);
    m.def("candidate_kind_names", &ygo::encode::candidate_kind_names);
    m.def("token_numeric_names", &ygo::encode::token_numeric_names);
    m.def("global_numeric_names", &ygo::encode::global_numeric_names);
    m.def("candidate_numeric_names", &ygo::encode::candidate_numeric_names);
    m.attr("MAX_TARGETS") = ygo::encode::kMaxTargets;
}
