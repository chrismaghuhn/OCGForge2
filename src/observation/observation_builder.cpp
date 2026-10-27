#include "ygo/observation/observation_builder.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "card_projection.hpp"
#include "common.h"
#include "ocgapi_types.h"
#include "query_decoder.hpp"
#include "ygo/core/core_host.hpp"
#include "ygo/observation/serialization.hpp"
#include "zone_projection.hpp"

namespace ygo::observation {
namespace {

constexpr std::uint32_t kCardQueryFlags =
    QUERY_CODE | QUERY_POSITION | QUERY_ALIAS | QUERY_TYPE | QUERY_LEVEL | QUERY_RANK |
    QUERY_ATTRIBUTE | QUERY_RACE | QUERY_ATTACK | QUERY_DEFENSE | QUERY_BASE_ATTACK |
    QUERY_BASE_DEFENSE | QUERY_REASON | QUERY_REASON_CARD | QUERY_EQUIP_CARD | QUERY_TARGET_CARD |
    QUERY_OVERLAY_CARD | QUERY_COUNTERS | QUERY_OWNER | QUERY_STATUS | QUERY_IS_PUBLIC |
    QUERY_LSCALE | QUERY_RSCALE | QUERY_LINK | QUERY_IS_HIDDEN | QUERY_COVER;

std::string player_prefix(std::uint8_t player) {
    return "p" + std::to_string(player);
}

bool raw_face_up(const detail::RawCardQuery& query) {
    return query.position.has_value() && ((*query.position & POS_FACEUP) != 0);
}

bool card_identity_visible(const detail::RawCardQuery& query, std::uint8_t perspective,
                           std::uint8_t owner, SemanticZone zone) {
    if (owner == perspective) {
        return zone != SemanticZone::MainDeck;
    }
    if (query.is_public.value_or(0) != 0 || raw_face_up(query)) {
        return true;
    }
    if (zone == SemanticZone::Graveyard || zone == SemanticZone::Banished) {
        return query.is_public.value_or(0) != 0;
    }
    return false;
}

bool overlay_material_identity_visible(const detail::RawCardQuery& material,
                                       bool parent_identity_visible) {
    return parent_identity_visible && material.code.has_value();
}

bool keep_entity(const detail::RawCardQuery& query, std::uint8_t perspective,
                 std::uint8_t owner, SemanticZone zone) {
    if (zone == SemanticZone::MainDeck) {
        return false;
    }
    if (zone == SemanticZone::Hand || zone == SemanticZone::ExtraDeck) {
        return card_identity_visible(query, perspective, owner, zone);
    }
    return true;
}


std::string zone_token(SemanticZone zone) {
    return semantic_zone_name(zone);
}

using EntityKey = std::tuple<std::uint8_t, SemanticZone, std::uint32_t, std::uint32_t>;
constexpr std::uint32_t kNoOverlaySequence = std::numeric_limits<std::uint32_t>::max();

EntityKey entity_key(std::uint8_t controller, SemanticZone zone, std::uint32_t sequence,
                     std::uint32_t overlay_sequence = kNoOverlaySequence) {
    return {controller, zone, sequence, overlay_sequence};
}

detail::RawCardQuery query_card_impl(const ygo::core::CoreHost& host, std::uint8_t controller,
                                     std::uint32_t location, std::uint32_t sequence,
                                     std::optional<std::uint32_t> overlay_sequence
                                     ) {
    OCG_QueryInfo info{};
    info.flags = kCardQueryFlags;
    info.con = controller;
    info.loc = location;
    info.seq = sequence;
    if (overlay_sequence.has_value()) {
        info.loc |= LOCATION_OVERLAY;
        info.overlay_seq = *overlay_sequence;
    }

    const auto bytes = host.query(info);
    return bytes.empty() ? detail::RawCardQuery{} : detail::decode_card_query(bytes);
}

detail::RawCardQuery query_card(const ygo::core::CoreHost& host, std::uint8_t controller,
                                std::uint32_t location, std::uint32_t sequence,
                                std::optional<std::uint32_t> overlay_sequence = std::nullopt) {
    return query_card_impl(host, controller, location, sequence, overlay_sequence
    );
}


std::string locator_for(std::uint8_t controller, SemanticZone zone, std::uint32_t sequence,
                        const detail::RawCardQuery& query, bool sequence_visible,
                        std::map<std::pair<std::uint8_t, std::uint32_t>, std::uint32_t>& ordinals) {
    const auto prefix = player_prefix(controller) + ":" + zone_token(zone);
    if (sequence_visible) {
        return prefix + ":" + std::to_string(sequence);
    }
    if (query.code.has_value()) {
        const auto key = std::make_pair(controller, *query.code);
        const auto ordinal = ordinals[key]++;
        return prefix + ":public:" + std::to_string(*query.code) + ":" + std::to_string(ordinal);
    }
    return prefix + ":unknown";
}

bool sequence_is_visible(SemanticZone zone, std::uint8_t owner, std::uint8_t perspective) {
    if (zone == SemanticZone::MonsterZone || zone == SemanticZone::SpellTrapZone ||
        zone == SemanticZone::FieldZone || zone == SemanticZone::PendulumRelevant) {
        return true;
    }
    if (zone == SemanticZone::Hand) {
        return owner == perspective;
    }
    if (zone == SemanticZone::Graveyard || zone == SemanticZone::Banished) {
        return true;
    }
    return false;
}

void add_zone_counts(PlayerObservation& observation, const ygo::core::CoreHost& host,
                     const detail::RawFieldSnapshot& field, std::uint8_t perspective
                     ) {
    for (std::uint8_t player = 0; player < 2; ++player) {
        const auto& source = field.players[player];
        const auto add = [&observation, player](SemanticZone kind, std::uint32_t total,
                                                std::uint32_t public_count, bool order) {
            ObservedZone zone;
            zone.player = player;
            zone.kind = kind;
            zone.total_count = total;
            zone.public_identity_count = public_count;
            zone.hidden_count = total >= public_count ? total - public_count : 0;
            zone.player_observable_order = order;
            observation.zones.push_back(zone);
        };
        add(SemanticZone::MainDeck, source.main_deck_count, 0, false);
        add(SemanticZone::Hand, source.hand_count,
            player == perspective ? source.hand_count : 0, player == perspective);

        std::uint32_t monster_count = 0;
        std::uint32_t monster_public = 0;
        std::uint32_t spell_count = 0;
        std::uint32_t spell_public = 0;
        std::uint32_t field_count = 0;
        std::uint32_t field_public = 0;
        std::uint32_t pendulum_count = 0;
        std::uint32_t pendulum_public = 0;
        std::uint32_t overlay_count = 0;
        std::uint32_t overlay_public = 0;
        for (std::uint32_t sequence = 0; sequence < source.monster_slots.size(); ++sequence) {
            const auto& slot = source.monster_slots[sequence];
            if (slot.occupied) {
                ++monster_count;
                if (slot.position & POS_FACEUP) {
                    ++monster_public;
                }
                overlay_count += slot.overlay_count;
                if (slot.overlay_count != 0) {
                    const auto parent = query_card(host, player, LOCATION_MZONE, sequence
                    );
                    const auto owner = static_cast<std::uint8_t>(parent.owner.value_or(player));
                    const bool parent_identity_visible =
                        card_identity_visible(parent, perspective, owner, SemanticZone::MonsterZone);
                    if (parent_identity_visible) {
                        for (std::uint32_t overlay_sequence = 0;
                             overlay_sequence < slot.overlay_count; ++overlay_sequence) {
                            const auto material = query_card(host, player, LOCATION_MZONE, sequence,
                                                             overlay_sequence
                            );
                            if (overlay_material_identity_visible(material, parent_identity_visible)) {
                                ++overlay_public;
                            }
                        }
                    }
                }
            }
        }
        for (std::uint32_t sequence = 0; sequence < source.spell_trap_slots.size(); ++sequence) {
            const auto& slot = source.spell_trap_slots[sequence];
            if (!slot.occupied) {
                continue;
            }
            const auto projected = detail::project_zone(LOCATION_SZONE, player, sequence, field.duel_options).zone;
            if (projected == SemanticZone::FieldZone) {
                ++field_count;
                if (slot.position & POS_FACEUP) {
                    ++field_public;
                }
            } else if (projected == SemanticZone::PendulumRelevant) {
                ++pendulum_count;
                if (slot.position & POS_FACEUP) {
                    ++pendulum_public;
                }
            } else {
                ++spell_count;
                if (slot.position & POS_FACEUP) {
                    ++spell_public;
                }
            }
        }
        add(SemanticZone::MonsterZone, monster_count,
            player == perspective ? monster_count : monster_public, true);
        add(SemanticZone::SpellTrapZone, spell_count,
            player == perspective ? spell_count : spell_public, true);
        add(SemanticZone::FieldZone, field_count,
            player == perspective ? field_count : field_public, true);
        add(SemanticZone::PendulumRelevant, pendulum_count,
            player == perspective ? pendulum_count : pendulum_public, true);
        add(SemanticZone::Graveyard, source.graveyard_count, source.graveyard_count, true);
        add(SemanticZone::Banished, source.banished_count, source.banished_count, true);
        add(SemanticZone::ExtraDeck, source.extra_deck_count,
            player == perspective ? source.extra_deck_count : source.face_up_extra_deck_count, false);
        add(SemanticZone::Overlay, overlay_count, overlay_public, false);
    }
}

std::optional<ObservationLocator> add_card_entity(
    PlayerObservation& observation, const ygo::core::CoreHost& host,
    const detail::RawCardQuery& query, std::uint8_t owner, std::uint8_t controller,
    SemanticZone zone, std::uint32_t sequence, std::uint8_t perspective,
    std::map<std::pair<std::uint8_t, std::uint32_t>, std::uint32_t>& ordinals,
    std::map<EntityKey, ObservationLocator>& locator_index
    ) {
    const bool identity_visible = card_identity_visible(query, perspective, owner, zone);
    if (!keep_entity(query, perspective, owner, zone)) {
        return std::nullopt;
    }
    detail::CardProjectionInput input;
    input.query = query;
    input.owner = owner;
    input.controller = controller;
    input.zone = zone;
    input.sequence = sequence;
    input.locator = {locator_for(controller, zone, sequence, query,
                                 sequence_is_visible(zone, owner, perspective), ordinals)};
    input.identity_visible = identity_visible;
    input.current_features_visible = identity_visible;
    input.sequence_visible = sequence_is_visible(zone, owner, perspective);
    if (identity_visible && query.code.has_value()) {
        input.printed = host.static_card_data(*query.code);
    }
    auto projected = detail::project_card(input);
    locator_index[entity_key(controller, zone, sequence)] = projected.locator;
    observation.entities.push_back(projected);
    return projected.locator;
}

std::optional<ObservationLocator> find_locator(
    const std::map<EntityKey, ObservationLocator>& locator_index, const detail::RawLocation& location,
    std::uint64_t duel_flags
    ) {
    const auto zone = detail::project_zone(location.location, location.controller, location.sequence, duel_flags).zone;
    const auto overlay_sequence = zone == SemanticZone::Overlay ? location.position : kNoOverlaySequence;
    const auto it = locator_index.find(entity_key(location.controller, zone, location.sequence, overlay_sequence));
    if (it == locator_index.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<ObservationLocator> add_overlay_material(
    PlayerObservation& observation, const ygo::core::CoreHost& host, std::uint8_t controller,
    std::uint32_t parent_location, std::uint32_t parent_sequence, std::uint32_t overlay_sequence,
    bool parent_identity_visible, std::map<EntityKey, ObservationLocator>& locator_index
    ) {
    const auto query = query_card(host, controller, parent_location, parent_sequence, overlay_sequence
    );
    detail::CardProjectionInput input;
    input.query = query;
    input.owner = query.owner.value_or(controller);
    input.controller = controller;
    input.zone = SemanticZone::Overlay;
    input.sequence = parent_sequence;
    input.overlay_sequence = overlay_sequence;
    input.locator = {player_prefix(controller) + ":OVERLAY:" + std::to_string(parent_sequence) + ":" +
                     std::to_string(overlay_sequence)};
    input.identity_visible = overlay_material_identity_visible(query, parent_identity_visible);
    input.current_features_visible = input.identity_visible;
    input.sequence_visible = true;
    if (input.identity_visible && query.code.has_value()) {
        input.printed = host.static_card_data(*query.code);
    }
    auto projected = detail::project_card(input);
    locator_index[entity_key(controller, SemanticZone::Overlay, parent_sequence, overlay_sequence)] = projected.locator;
    observation.entities.push_back(projected);
    return projected.locator;
}

void project_event_globals(PlayerObservation& observation) {
    std::uint32_t observed_turns = 0;
    for (const auto& event : observation.visible_events) {
        switch (event.kind) {
        case VisibleEventKind::TurnStarted:
            ++observed_turns;
            if (event.player.has_value()) {
                observation.globals.turn_player = event.player;
            }
            break;
        case VisibleEventKind::PhaseChanged:
            if (event.phase.has_value()) {
                observation.globals.phase = event.phase;
            }
            break;
        case VisibleEventKind::Win:
            observation.globals.terminal = true;
            observation.globals.winner = event.winner;
            observation.globals.win_reason = event.win_reason;
            break;
        default:
            break;
        }
    }
    if (observed_turns != 0) {
        observation.globals.turn_count = observed_turns;
    }
}

}  // namespace

PlayerObservation build_player_observation(const ygo::core::CoreHost& host,
                                           std::uint8_t perspective_player,
                                           const ObservationBuildConfig& config) {
    if (perspective_player > 1) {
        throw std::invalid_argument("observation perspective must be player 0 or 1");
    }
    const auto field = detail::decode_field_query(host.query_field());
    PlayerObservation observation;
    observation.perspective_player = perspective_player;
    observation.decision_index = config.decision_index;
    observation.engine_step_index = config.engine_step_index;
    observation.visible_events = config.visible_events;
    if (config.decision_context.has_value()) {
        observation.decision_context = *config.decision_context;
        observation.globals.player_to_act = observation.decision_context.player;
    }
    project_event_globals(observation);
    observation.globals.duel_flags = field.duel_options;
    observation.globals.life_points = {field.players[0].life_points, field.players[1].life_points};
    observation.globals.chain_length = static_cast<std::uint32_t>(field.chain.size());

    observation.match_context.perspective_player = perspective_player;
    observation.match_context.duel_flags = field.duel_options;
    observation.match_context.knowledge = config.knowledge;
    observation.match_context.own_deck = config.own_deck;
    observation.match_context.opponent_deck = config.opponent_deck;
    if (!config.knowledge.own_decklist_known) {
        observation.match_context.own_deck = {};
    }
    if (!config.knowledge.opponent_decklist_known) {
        observation.match_context.opponent_deck = {};
    }

    add_zone_counts(observation, host, field, perspective_player
    );

    const std::array<std::uint32_t, 5> locations = {
        LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED};
    std::map<std::pair<std::uint8_t, std::uint32_t>, std::uint32_t> ordinals;
    std::map<EntityKey, ObservationLocator> locator_index;
    struct PendingRelationship {
        RelationshipKind kind;
        ObservationLocator source;
        detail::RawLocation target;
    };
    std::vector<PendingRelationship> pending_relationships;
    for (std::uint8_t player = 0; player < 2; ++player) {
        for (const auto location : locations) {
            OCG_QueryInfo info{};
            info.flags = kCardQueryFlags;
            info.con = player;
            info.loc = location;
            const auto decoded = detail::decode_location_query(host.query_location(info));
            for (std::uint32_t sequence = 0; sequence < decoded.entries.size(); ++sequence) {
                if (!decoded.entries[sequence].has_value()) {
                    continue;
                }
                const auto& query = *decoded.entries[sequence];
                const auto owner = static_cast<std::uint8_t>(query.owner.value_or(player));
                const auto projected_zone = detail::project_zone(location, player, sequence, field.duel_options).zone;
                const auto source = add_card_entity(observation, host, query, owner, player, projected_zone, sequence,
                                                    perspective_player, ordinals, locator_index
                );
                if (!source.has_value()) {
                    continue;
                }
                if (query.equip_card.has_value()) {
                    pending_relationships.push_back({RelationshipKind::Equip, *source, *query.equip_card});
                }
                for (const auto& target : query.targets) {
                    pending_relationships.push_back({RelationshipKind::Target, *source, target});
                }
                for (std::uint32_t overlay_sequence = 0;
                     overlay_sequence < query.overlay_codes.size(); ++overlay_sequence) {
                    const auto parent_identity_visible =
                        card_identity_visible(query, perspective_player, owner, projected_zone);
                    const auto material = add_overlay_material(observation, host, player, location, sequence,
                                                               overlay_sequence, parent_identity_visible,
                                                               locator_index
                    );
                    if (material.has_value()) {
                        pending_relationships.push_back({RelationshipKind::XyzMaterial, *material, detail::RawLocation{
                                                                                                          player,
                                                                                                          LOCATION_MZONE,
                                                                                                          sequence,
                                                                                                          0}});
                    }
                }
            }
        }

        OCG_QueryInfo extra_info{};
        extra_info.flags = kCardQueryFlags;
        extra_info.con = player;
        extra_info.loc = LOCATION_EXTRA;
        const auto extra = detail::decode_location_query(host.query_location(extra_info));
        for (std::uint32_t sequence = 0; sequence < extra.entries.size(); ++sequence) {
            if (!extra.entries[sequence].has_value()) {
                continue;
            }
            const auto& query = *extra.entries[sequence];
            const auto owner = static_cast<std::uint8_t>(query.owner.value_or(player));
            (void)add_card_entity(observation, host, query, owner, player, SemanticZone::ExtraDeck, sequence,
                                  perspective_player, ordinals, locator_index
            );
        }
    }

    {
        for (const auto& pending : pending_relationships) {
            const auto target = find_locator(locator_index, pending.target, field.duel_options
            );
            if (target.has_value()) {
                observation.relationships.push_back({pending.kind, pending.source, *target});
            }
        }
    }

    observation.chain.length = static_cast<std::uint32_t>(field.chain.size());
    for (std::uint32_t index = 0; index < field.chain.size(); ++index) {
        const auto& raw = field.chain[index];
        ChainLink link;
        link.index = index;
        link.activating_player = raw.triggering_player;
        link.activation_zone = detail::project_zone(raw.source.location, raw.source.controller,
                                                     raw.source.sequence, field.duel_options).zone;
        const auto source_it = std::find_if(
            observation.entities.begin(), observation.entities.end(), [&](const ObservedCard& card) {
                return card.controller.value_or(2) == raw.source.controller &&
                       card.zone == link.activation_zone && card.sequence.value_or(UINT32_MAX) == raw.source.sequence;
            });
        if (source_it != observation.entities.end() && source_it->identity_known) {
            link.source = source_it->locator;
            link.effect_description = raw.description;
            for (const auto& relationship : observation.relationships) {
                if (relationship.kind == RelationshipKind::Target &&
                    relationship.source == source_it->locator) {
                    link.targets.push_back(relationship.target);
                }
            }
        }
        observation.chain.links.push_back(link);
    }

    return observation;
}

}  // namespace ygo::observation
