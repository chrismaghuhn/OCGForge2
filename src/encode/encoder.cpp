#include "ygo/encode/encoder.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

#include "ocgapi_constants.h"
#include "ygo/observation/decision_integration.hpp"
#include "zone_projection.hpp"

namespace ygo::encode {
namespace {

using observation::ObservedCard;
using observation::PlayerObservation;
using observation::SemanticZone;
using protocol::ActionCandidate;
using protocol::ActionKind;
using protocol::DecisionRequest;

constexpr std::int64_t kUnknownCard = 1;

const std::vector<std::string> kDecisions = {
    "<pad>", "idle_command", "battle_command", "chain", "option", "card_selection", "tribute", "sum",
    "place", "counter", "ordering", "announcement", "unselect_card", "position", "yes_no"};
const std::vector<std::string> kZones = {
    "<pad>", "unknown", "main_deck", "hand", "monster_zone", "spell_trap_zone", "graveyard", "banished",
    "extra_deck", "field_zone", "pendulum_zone", "overlay", "chain"};
const std::vector<std::string> kPhases = {
    "<pad>", "draw", "standby", "main1", "battle_start", "battle_step", "damage", "damage_calculation",
    "battle", "main2", "end"};
const std::vector<std::string> kCandidateKinds = {
    "<pad>",
    "summon", "special_summon", "reposition", "set_monster", "set_spell_trap", "activate",
    "to_battle_phase", "to_end_phase", "shuffle_hand",
    "attack", "to_main_phase_2",
    "chain_activate", "chain_pass",
    "option", "select_card", "unselect_card", "announce", "place", "position", "yes", "no",
    "pick", "finish", "cancel", "assign_amount"};
const std::vector<std::string> kTokenNumeric = {
    "faceUp", "attackPosition", "sequence", "atk", "def", "level", "overlayCount", "counters",
    "disabled", "selected", "chainIndex"};
const std::vector<std::string> kGlobalNumeric = {
    "self_lp", "self_hand", "self_deck", "self_extra", "self_graveyard", "self_banished", "self_monsters",
    "self_spellTraps",
    "opp_lp", "opp_hand", "opp_deck", "opp_extra", "opp_graveyard", "opp_banished", "opp_monsters",
    "opp_spellTraps",
    "turn", "isOwnTurn", "chainSize", "selectMin", "selectMax", "selectedCount"};
// Public events worth remembering (see remembered_kind); phase changes, shuffles,
// chain bookkeeping and own draws carry nothing a later decision needs.
const std::vector<std::string> kEventKinds = {
    "<pad>", "turn_started", "card_moved", "card_revealed", "summoned", "set", "life_points_changed",
    "chain_activated", "card_destroyed", "card_banished", "card_returned", "position_changed", "counter_changed",
    "equipped", "targeted"};
const std::vector<std::string> kEventNumeric = {
    "turnsAgo",  // turns started since the event / 8, capped at 1
    "recency",   // 0 = newest encoded event, 1 = oldest possible
    "amount",    // life point change / 8000 (life point events only)
    "count"};    // the event's count / 20, capped at 1

const std::vector<std::string> kCandidateNumeric = {
    "hasSource", "targetCount", "effectIndex", "value", "faceUp", "attackPosition", "zone", "opponentSide",
    "variant"};

std::int64_t kind_index(const std::string& name) {
    const auto it = std::find(kCandidateKinds.begin(), kCandidateKinds.end(), name);
    if (it == kCandidateKinds.end()) {
        throw std::logic_error("unknown candidate kind " + name);
    }
    return it - kCandidateKinds.begin();
}

std::int64_t candidate_kind(const ActionCandidate& candidate) {
    static const char* const kIdle[] = {"summon", "special_summon", "reposition", "set_monster", "set_spell_trap",
                                        "activate", "to_battle_phase", "to_end_phase", "shuffle_hand"};
    static const char* const kBattle[] = {"activate", "attack", "to_main_phase_2", "to_end_phase"};
    switch (candidate.action_kind) {
    case ActionKind::IdleCommand:
        if (candidate.phase < std::size(kIdle)) {
            return kind_index(kIdle[candidate.phase]);
        }
        break;
    case ActionKind::BattleCommand:
        if (candidate.phase < std::size(kBattle)) {
            return kind_index(kBattle[candidate.phase]);
        }
        break;
    case ActionKind::Chain:
        return kind_index(candidate.semantic_key == "chain.pass" ? "chain_pass" : "chain_activate");
    case ActionKind::Option:
        return kind_index("option");
    case ActionKind::CardSelection:
        return kind_index(candidate.card_selection_operation == protocol::CardSelectionOperation::Unselect
                              ? "unselect_card"
                              : "select_card");
    case ActionKind::Announcement:
        return kind_index("announce");
    case ActionKind::Place:
        return kind_index("place");
    case ActionKind::Position:
        return kind_index("position");
    case ActionKind::YesNo:
        return kind_index(candidate.semantic_key == "yes_no.yes" ? "yes" : "no");
    case ActionKind::Pick:
        return kind_index("pick");
    case ActionKind::Finish:
        return kind_index("finish");
    case ActionKind::Cancel:
        return kind_index("cancel");
    case ActionKind::AssignAmount:
        return kind_index("assign_amount");
    }
    throw std::invalid_argument("cannot encode candidate " + candidate.semantic_key);
}

// Index into kEventKinds, or -1 for events not worth a memory slot: besides the kinds
// left out of kEventKinds, the empty chain marker (MSG_CHAINED, no card, no player) and
// the deciding player's own draws/hand reveals (its hand is visible anyway).
std::int64_t remembered_kind(const observation::VisibleGameEvent& event, std::uint8_t self) {
    using observation::VisibleEventKind;
    switch (event.kind) {
    case VisibleEventKind::TurnStarted: return 1;
    case VisibleEventKind::CardMoved: return 2;
    case VisibleEventKind::CardRevealed:
        return event.player == self && event.to_zone == SemanticZone::Hand ? -1 : 3;
    case VisibleEventKind::Summoned: return event.player ? 4 : -1;  // the engine also sends an empty "summoned" marker
    case VisibleEventKind::Set: return 5;
    case VisibleEventKind::LifePointsChanged: return 6;
    case VisibleEventKind::ChainActivated: return event.player || event.public_passcode ? 7 : -1;
    case VisibleEventKind::CardDestroyed: return 8;
    case VisibleEventKind::CardBanished: return 9;
    case VisibleEventKind::CardReturned: return 10;
    case VisibleEventKind::PositionChanged: return 11;
    case VisibleEventKind::CounterChanged: return 12;
    case VisibleEventKind::Equipped: return 13;
    case VisibleEventKind::Targeted: return 14;
    default: return -1;
    }
}

std::int64_t zone_index(SemanticZone zone) {
    // SemanticZone order shifted by one for padding.
    return static_cast<std::int64_t>(zone) + 1;
}

bool on_field_zone(SemanticZone zone) {
    return zone == SemanticZone::MonsterZone || zone == SemanticZone::SpellTrapZone ||
           zone == SemanticZone::FieldZone || zone == SemanticZone::PendulumRelevant;
}

std::int64_t phase_index(const std::optional<std::uint32_t>& phase) {
    // PHASE_* bits in order: index = log2(bit) + 1.
    if (!phase || *phase == 0 || !std::has_single_bit(*phase) || *phase > PHASE_END) {
        return 0;
    }
    return std::countr_zero(*phase) + 1;
}

// For an effect activation the engine sends a description (code << 20 | n)
// naming the card's n-th effect string; anything else is a system string.
// Options carry the same kind of description, naming the card whose effect offers them.
float effect_index(const ActionCandidate& candidate) {
    if (!candidate.choice_value) {
        return 0.0f;
    }
    const auto description = *candidate.choice_value;
    const auto card = description >> 20;
    const bool own_effect = candidate.source_card != 0 && card == candidate.source_card;
    if (!own_effect && !(candidate.action_kind == ActionKind::Option && card != 0)) {
        return 0.0f;
    }
    return static_cast<float>((description & 0xFFFFF) + 1) / 16.0f;
}

float scaled(std::optional<std::int32_t> value, float scale) {
    return value ? static_cast<float>(std::max(*value, 0)) / scale : 0.0f;
}

std::uint32_t zone_count(const PlayerObservation& observation, std::uint8_t player,
                         std::initializer_list<SemanticZone> kinds) {
    std::uint32_t total = 0;
    for (const auto& zone : observation.zones) {
        if (zone.player == player && std::find(kinds.begin(), kinds.end(), zone.kind) != kinds.end()) {
            total += zone.total_count;
        }
    }
    return total;
}

}  // namespace

EncodedDecision encode(const PlayerObservation& observation, const DecisionRequest& request,
                       const CardIndex& card_index) {
    if (request.kind == protocol::DecisionRequestKind::Unsupported) {
        throw std::invalid_argument("cannot encode an unsupported decision");
    }
    if (request.candidates.empty()) {
        throw std::invalid_argument("cannot encode a decision without candidates");
    }
    const auto self = observation.perspective_player;
    EncodedDecision out;
    out.player = self;
    out.decision = static_cast<std::int64_t>(request.kind) + 1;
    out.phase = phase_index(observation.globals.phase);

    // Tokens: every entity of the observation, in observation order.
    std::map<std::string, std::int64_t> token_of;
    std::map<std::pair<std::uint8_t, std::uint32_t>, std::uint32_t> overlay_count;  // (controller, parent seq)
    for (const auto& card : observation.entities) {
        if (card.zone == SemanticZone::Overlay && card.controller && card.sequence) {
            ++overlay_count[{*card.controller, *card.sequence}];
        }
    }
    std::map<std::string, float> chain_index;
    for (const auto& link : observation.chain.links) {
        if (link.source) {
            chain_index[link.source->value] = static_cast<float>(link.index + 1) / 10.0f;
        }
    }
    const auto columns = kTokenNumeric.size();
    for (const auto& card : observation.entities) {
        token_of[card.locator.value] = static_cast<std::int64_t>(out.token_card.size());
        std::int64_t index = kUnknownCard;
        if (card.identity_known && card.passcode) {
            const auto it = card_index.find(*card.passcode);
            index = it == card_index.end() ? kUnknownCard : it->second;
        }
        out.token_card.push_back(index);
        out.token_zone.push_back(zone_index(card.zone));
        out.token_side.push_back(card.controller.value_or(self) == self ? 0 : 1);

        const auto& current = card.current;
        std::uint32_t counters = 0;
        std::optional<std::int32_t> level;
        bool disabled = false;
        if (current) {
            for (const auto& counter : current->counters) {
                counters += counter.count;
            }
            // Links report level 0 next to their rating, so look at rating and rank first.
            if (current->link_rating && *current->link_rating > 0) {
                level = static_cast<std::int32_t>(*current->link_rating);
            } else if (current->rank && *current->rank > 0) {
                level = static_cast<std::int32_t>(*current->rank);
            } else if (current->level) {
                level = static_cast<std::int32_t>(*current->level);
            }
            disabled = current->status_flags && (*current->status_flags & STATUS_DISABLED) != 0;
        }
        const bool on_field = card.zone == SemanticZone::MonsterZone;
        const auto overlays = card.controller && card.sequence && on_field
                                  ? overlay_count[{*card.controller, *card.sequence}]
                                  : 0;
        const auto chain = chain_index.find(card.locator.value);
        const float row[] = {
            card.face_up ? 1.0f : 0.0f,
            card.position == observation::Position::FaceUpAttack ||
                    card.position == observation::Position::FaceDownAttack
                ? 1.0f
                : 0.0f,
            on_field_zone(card.zone) ? static_cast<float>(card.sequence.value_or(0)) / 7.0f : 0.0f,
            current ? scaled(current->attack, 5000.0f) : 0.0f,
            current ? scaled(current->defense, 5000.0f) : 0.0f,
            scaled(level, 12.0f),
            static_cast<float>(overlays) / 5.0f,
            static_cast<float>(counters) / 10.0f,
            disabled ? 1.0f : 0.0f,
            0.0f,  // selected, filled in below
            chain == chain_index.end() ? 0.0f : chain->second,
        };
        static_assert(std::size(row) == 11);
        out.token_numeric.insert(out.token_numeric.end(), std::begin(row), std::end(row));
    }

    const auto token_for = [&](std::uint32_t code, std::uint8_t controller, std::uint32_t location,
                               std::uint32_t sequence, std::uint32_t position) -> std::int64_t {
        if (location == 0 || location == LOCATION_DECK) {
            return -1;
        }
        std::optional<std::uint32_t> overlay;
        if ((location & LOCATION_OVERLAY) != 0) {
            overlay = position;
        }
        const auto locator = observation::find_card(observation, code, controller, location, sequence, overlay);
        if (!locator) {
            return -1;
        }
        const auto it = token_of.find(locator->value);
        return it == token_of.end() ? -1 : it->second;
    };
    const auto mark_selected = [&](std::int64_t token) {
        if (token >= 0) {
            out.token_numeric[static_cast<std::size_t>(token) * columns + 9] = 1.0f;
        }
    };

    // Multi-card selections: what is already picked, and the bounds.
    float select_min = 0.0f;
    float select_max = 0.0f;
    float selected_count = 0.0f;
    if (request.continuation) {
        const auto& continuation = *request.continuation;
        select_min = static_cast<float>(continuation.min_count) / 10.0f;
        select_max = static_cast<float>(continuation.max_count) / 10.0f;
        selected_count = static_cast<float>(continuation.selected_indices.size()) / 10.0f;
        for (const auto index : continuation.selected_indices) {
            const auto item = std::find_if(continuation.items.begin(), continuation.items.end(),
                                           [&](const auto& candidate_item) { return candidate_item.source_index == index; });
            if (item != continuation.items.end()) {
                mark_selected(token_for(item->card.code, item->card.controller, item->card.location,
                                        item->card.sequence, item->card.position));
            }
        }
    }

    // The engine lists some choices several times with identical card data (one
    // extra deck monster per summon procedure); `variant` numbers such repeats.
    std::map<std::tuple<std::int64_t, std::uint32_t, std::uint8_t, std::uint32_t, std::uint32_t>, int> repeats;
    for (const auto& candidate : request.candidates) {
        const auto kind = candidate_kind(candidate);
        out.candidate_kind.push_back(kind);
        const auto source = candidate.source_card == 0
                                ? -1
                                : token_for(candidate.source_card, candidate.source_controller,
                                            candidate.source_location, candidate.source_sequence,
                                            candidate.source_position);
        out.candidate_source.push_back(source);

        // A card the observation does not show (e.g. in the deck during a search)
        // still needs an identity, or the choices would look the same. Only cards
        // the deciding player may know: its own, or ones just shown to it.
        std::int64_t own_card = 0;
        std::uint32_t named_card = 0;
        if (source < 0 && candidate.source_card != 0 &&
            (candidate.source_controller == self ||
             observation::revealed_this_turn(observation, candidate.source_card, candidate.source_controller,
                                             candidate.source_location))) {
            named_card = candidate.source_card;
        } else if (candidate.action_kind == ActionKind::Option && candidate.choice_value) {
            named_card = static_cast<std::uint32_t>(*candidate.choice_value >> 20);  // the card offering the option
        }
        if (named_card != 0) {
            const auto it = card_index.find(named_card);
            own_card = it == card_index.end() ? kUnknownCard : it->second;
        }
        out.candidate_card.push_back(own_card);
        const bool has_place = candidate.source_location != 0;
        out.candidate_zone.push_back(
            has_place ? zone_index(observation::detail::project_zone(candidate.source_location, candidate.source_controller,
                                                                    candidate.source_sequence,
                                                                    observation.match_context.duel_flags)
                                       .zone)
                      : 0);
        const auto variant = repeats[{kind, candidate.source_card, candidate.source_controller,
                                      candidate.source_location, candidate.source_sequence}]++;
        if (candidate.card_selection_operation == protocol::CardSelectionOperation::Unselect) {
            mark_selected(source);
            selected_count += 0.1f;
        }
        std::int64_t targets[kMaxTargets] = {-1, -1, -1, -1};
        if (candidate.target_card != 0) {
            targets[0] = token_for(candidate.target_card, candidate.target_controller, candidate.target_location,
                                   candidate.target_sequence, 0);
        }
        out.candidate_targets.insert(out.candidate_targets.end(), std::begin(targets), std::end(targets));

        float value = 0.0f;
        if (candidate.amount >= 0) {
            value = static_cast<float>(candidate.amount) / 10.0f;
        } else if (candidate.action_kind == ActionKind::Announcement && candidate.choice_value) {
            value = static_cast<float>(std::min<std::uint64_t>(*candidate.choice_value, 24)) / 12.0f;  // a number
        } else if (candidate.action_kind == ActionKind::Option ||
                   (candidate.source_card == 0 && !has_place && candidate.choice_index)) {
            value = static_cast<float>(std::min<std::uint32_t>(*candidate.choice_index, 32)) / 16.0f;
        }
        const bool position_choice = candidate.action_kind == ActionKind::Position;
        const float row[] = {
            source >= 0 ? 1.0f : 0.0f,
            targets[0] >= 0 ? 0.25f : 0.0f,
            effect_index(candidate),
            value,
            position_choice && (candidate.position & POS_FACEUP) != 0 ? 1.0f : 0.0f,
            position_choice && (candidate.position & POS_ATTACK) != 0 ? 1.0f : 0.0f,
            has_place ? static_cast<float>(std::min<std::uint32_t>(candidate.source_sequence, 7)) / 7.0f : 0.0f,
            has_place && candidate.source_controller != self ? 1.0f : 0.0f,
            static_cast<float>(std::min(variant, 4)) / 4.0f,
        };
        static_assert(std::size(row) == 9);
        out.candidate_numeric.insert(out.candidate_numeric.end(), std::begin(row), std::end(row));
    }

    // Memory: the most recent events worth remembering, oldest first. Everything comes
    // from the deciding player's event stream, which only names cards it could see.
    //
    // event_token links an event to the card token at the place it happened, but only
    // while that is provably still the same card: walking from the newest event back,
    // a field slot becomes unusable as soon as a later card arrives there, a card leaves
    // its zone (we don't know which slot), or set cards are shuffled. Hand, graveyard
    // and banished positions renumber constantly and are never linked.
    {
        struct Picked {
            std::size_t index;
            std::uint32_t turns_ago;
            std::int64_t token;
        };
        std::vector<Picked> picked;
        std::set<std::tuple<std::uint8_t, SemanticZone, std::string>> moved_slots;  // (player, zone, sequence)
        std::set<std::pair<std::uint8_t, SemanticZone>> moved_zones;
        std::uint32_t turns = 0;
        const auto& events = observation.visible_events;
        const auto sequence_of = [](const observation::ObservationLocator& locator) {
            return locator.value.substr(locator.value.rfind(':') + 1);
        };
        for (auto it = events.rbegin(); it != events.rend(); ++it) {
            const auto& event = *it;
            if (picked.size() < kMaxEvents && remembered_kind(event, self) >= 0) {
                std::int64_t token = -1;
                if (event.entity && event.player && event.to_zone && on_field_zone(*event.to_zone) &&
                    moved_zones.count({*event.player, *event.to_zone}) == 0 &&
                    moved_slots.count({*event.player, *event.to_zone, sequence_of(*event.entity)}) == 0) {
                    const auto found = token_of.find(event.entity->value);
                    token = found == token_of.end() ? -1 : found->second;
                }
                picked.push_back({static_cast<std::size_t>(events.rend() - it - 1), turns, token});
            }
            // What this event changes for the older events still to come.
            // Moves come as CardMoved or, for some, as their own kind (returned, destroyed, banished).
            const bool move = event.kind == observation::VisibleEventKind::CardMoved ||
                              event.kind == observation::VisibleEventKind::CardReturned ||
                              event.kind == observation::VisibleEventKind::CardDestroyed ||
                              event.kind == observation::VisibleEventKind::CardBanished;
            if (move) {
                if (event.entity && event.player && event.to_zone) {
                    moved_slots.insert({*event.player, *event.to_zone, sequence_of(*event.entity)});
                }
                if (event.from_zone && on_field_zone(*event.from_zone)) {
                    moved_zones.insert({0, *event.from_zone});  // either player: control can change
                    moved_zones.insert({1, *event.from_zone});
                }
            } else if (event.kind == observation::VisibleEventKind::Shuffle && !event.player) {
                for (const std::uint8_t player : {0, 1}) {  // set cards shuffled: no field slot is certain
                    for (const auto zone : {SemanticZone::MonsterZone, SemanticZone::SpellTrapZone,
                                            SemanticZone::FieldZone, SemanticZone::PendulumRelevant}) {
                        moved_zones.insert({player, zone});
                    }
                }
            } else if (event.kind == observation::VisibleEventKind::TurnStarted) {
                ++turns;
            }
        }
        for (std::size_t n = picked.size(); n-- > 0;) {
            const auto& event = events[picked[n].index];
            out.event_kind.push_back(remembered_kind(event, self));
            std::int64_t card = 0;
            if (event.public_passcode && *event.public_passcode != 0) {
                const auto it = card_index.find(*event.public_passcode);
                card = it == card_index.end() ? kUnknownCard : it->second;
            }
            out.event_card.push_back(card);
            // A removed token has no destination, so no owner either.
            const bool has_side = event.player && !(event.from_zone && !event.to_zone);
            out.event_side.push_back(!has_side ? 2 : (*event.player == self ? 0 : 1));
            out.event_from.push_back(event.from_zone ? zone_index(*event.from_zone) : 0);
            out.event_to.push_back(event.to_zone ? zone_index(*event.to_zone) : 0);
            out.event_token.push_back(picked[n].token);
            const bool life_points = event.kind == observation::VisibleEventKind::LifePointsChanged;
            const float row[] = {
                std::min(static_cast<float>(picked[n].turns_ago) / 8.0f, 1.0f),
                static_cast<float>(n) / static_cast<float>(kMaxEvents),
                life_points && event.amount ? std::clamp(static_cast<float>(*event.amount) / 8000.0f, -1.0f, 1.0f)
                                            : 0.0f,
                static_cast<float>(std::min<std::uint32_t>(event.count.value_or(0), 20)) / 20.0f,
            };
            static_assert(std::size(row) == 4);
            out.event_numeric.insert(out.event_numeric.end(), std::begin(row), std::end(row));
        }
    }

    // Globals, from the deciding player's point of view.
    const auto opponent = static_cast<std::uint8_t>(1 - self);
    const auto& lp = observation.globals.life_points;
    for (const auto player : {self, opponent}) {
        const float life = player < lp.size() ? static_cast<float>(lp[player]) / 8000.0f : 0.0f;
        out.global_numeric.insert(out.global_numeric.end(), {
            life,
            static_cast<float>(zone_count(observation, player, {SemanticZone::Hand})) / 10.0f,
            static_cast<float>(zone_count(observation, player, {SemanticZone::MainDeck})) / 40.0f,
            static_cast<float>(zone_count(observation, player, {SemanticZone::ExtraDeck})) / 15.0f,
            static_cast<float>(zone_count(observation, player, {SemanticZone::Graveyard})) / 30.0f,
            static_cast<float>(zone_count(observation, player, {SemanticZone::Banished})) / 30.0f,
            static_cast<float>(zone_count(observation, player, {SemanticZone::MonsterZone})) / 5.0f,
            static_cast<float>(zone_count(observation, player, {SemanticZone::SpellTrapZone, SemanticZone::FieldZone,
                                                                SemanticZone::PendulumRelevant})) / 5.0f,
        });
    }
    out.global_numeric.insert(out.global_numeric.end(), {
        static_cast<float>(observation.globals.turn_count.value_or(0)) / 20.0f,
        observation.globals.turn_player == self ? 1.0f : 0.0f,
        static_cast<float>(observation.globals.chain_length) / 10.0f,
        select_min,
        select_max,
        selected_count,
    });
    if (out.global_numeric.size() != kGlobalNumeric.size()) {
        throw std::logic_error("global feature count does not match global_numeric_names()");
    }
    return out;
}

const std::vector<std::string>& decision_names() { return kDecisions; }
const std::vector<std::string>& zone_names() { return kZones; }
const std::vector<std::string>& phase_names() { return kPhases; }
const std::vector<std::string>& candidate_kind_names() { return kCandidateKinds; }
const std::vector<std::string>& token_numeric_names() { return kTokenNumeric; }
const std::vector<std::string>& global_numeric_names() { return kGlobalNumeric; }
const std::vector<std::string>& candidate_numeric_names() { return kCandidateNumeric; }
const std::vector<std::string>& event_kind_names() { return kEventKinds; }
const std::vector<std::string>& event_numeric_names() { return kEventNumeric; }

}  // namespace ygo::encode
