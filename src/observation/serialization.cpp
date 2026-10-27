#include "ygo/observation/serialization.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ygo/util/sha256.hpp"

namespace ygo::observation {
namespace {

using SerializationClock = std::chrono::steady_clock;

constexpr std::uint8_t kCopyTopLevelZones = 0;
constexpr std::uint8_t kCopyTopLevelEntities = 1;
constexpr std::uint8_t kCopyTopLevelRelationships = 2;
constexpr std::uint8_t kCopyTopLevelVisibleEvents = 3;
constexpr std::uint8_t kCopyChainTargets = 4;
constexpr std::uint8_t kCopyEventTargets = 5;
constexpr std::uint8_t kCopyDecisionReferences = 6;
constexpr std::uint8_t kCopyLinkMarkers = 7;
constexpr std::uint8_t kCopyCounters = 8;
constexpr std::uint8_t kCopyOwnDeck = 9;
constexpr std::uint8_t kCopyOpponentDeck = 10;

constexpr std::uint8_t kSortZones = 0;
constexpr std::uint8_t kSortEntities = 1;
constexpr std::uint8_t kSortRelationships = 2;
constexpr std::uint8_t kSortVisibleEvents = 3;
constexpr std::uint8_t kSortChainTargets = 4;
constexpr std::uint8_t kSortEventTargets = 5;
constexpr std::uint8_t kSortDecisionReferences = 6;
constexpr std::uint8_t kSortLinkMarkers = 7;
constexpr std::uint8_t kSortCounters = 8;
constexpr std::uint8_t kSortDecks = 9;

constexpr std::uint8_t kNestedPrintedProperties = 0;
constexpr std::uint8_t kNestedCurrentProperties = 1;
constexpr std::uint8_t kNestedCounters = 2;
constexpr std::uint8_t kNestedLinkMarkers = 3;
constexpr std::uint8_t kNestedOwnDeck = 4;
constexpr std::uint8_t kNestedOpponentDeck = 5;
constexpr std::uint8_t kNestedOtherMatchContext = 6;

constexpr std::uint8_t kPrimitiveNumeric = 0;
constexpr std::uint8_t kPrimitiveBoolean = 1;
constexpr std::uint8_t kPrimitiveNull = 2;

using CanonicalOutput = std::ostringstream;

std::size_t output_offset(CanonicalOutput& output) {
    const auto position = output.tellp();
    return position < 0 ? 0 : static_cast<std::size_t>(position);
}

std::string output_extract(CanonicalOutput& output) {
    return output.str();
}

template <typename T>
std::uint64_t approximate_fixed_vector_bytes(const std::vector<T>& values) {
    return static_cast<std::uint64_t>(sizeof(T)) * static_cast<std::uint64_t>(values.size());
}


std::uint64_t approximate_locator_vector_bytes(const std::vector<ObservationLocator>& values) {
    std::uint64_t bytes = approximate_fixed_vector_bytes(values);
    for (const auto& value : values) {
        bytes += value.value.size();
    }
    return bytes;
}

class SerializationShapeProbe final {
public:
    SerializationShapeProbe() = default;

    template <typename Action>
    auto copy(std::uint8_t, std::uint64_t, std::uint64_t, Action&& action) -> decltype(action()) {
        return action();
    }

    template <typename Action>
    void sort(std::uint8_t, std::uint64_t, Action&& action) {
        action();
    }

    template <typename Action>
    std::string escape(const std::string&, Action&& action) {
        return action();
    }

    void set_observation_metadata(std::uint8_t, std::uint64_t, std::uint64_t, std::uint64_t) noexcept {}
    void record_section(std::uint8_t, std::size_t, std::size_t) noexcept {}
    void record_entity(std::uint64_t) noexcept {}
    void record_property(std::uint8_t, bool, std::uint64_t) noexcept {}
    void record_nested_bytes(std::uint8_t, std::uint64_t, std::uint64_t) noexcept {}
    void record_visible_event(const VisibleGameEvent&, std::uint64_t) noexcept {}
    void record_primitive(std::uint8_t) noexcept {}
    void record_final_extraction(std::uint64_t) noexcept {}
    void record_legacy_copy_event() noexcept {
    }
    void finish(std::uint64_t) noexcept {}

};

std::string json_escape_impl(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (character < 0x20) {
                const char* hex = "0123456789abcdef";
                out << "\\u00" << hex[character >> 4] << hex[character & 0xf];
            } else {
                out << static_cast<char>(character);
            }
            break;
        }
    }
    out << '"';
    return out.str();
}

std::string json_escape(const std::string& value, SerializationShapeProbe& shape) {
    return shape.escape(value, [&value]() { return json_escape_impl(value); });
}

std::string json_escape(const std::string& value) {
    return json_escape_impl(value);
}

template <typename T>
void write_optional_number(CanonicalOutput& out,
                           const std::optional<T>& value,
                           SerializationShapeProbe& shape) {
    if (value.has_value()) {
        shape.record_primitive(kPrimitiveNumeric);
        out << *value;
    } else {
        shape.record_primitive(kPrimitiveNull);
        out << "null";
    }
}

void write_optional_string(CanonicalOutput& out,
                           const std::optional<std::string>& value,
                           SerializationShapeProbe& shape) {
    if (value.has_value()) {
        out << json_escape(*value, shape);
    } else {
        shape.record_primitive(kPrimitiveNull);
        out << "null";
    }
}

void write_optional_locator(CanonicalOutput& out,
                            const std::optional<ObservationLocator>& value,
                            SerializationShapeProbe& shape) {
    if (value.has_value()) {
        out << json_escape(value->value, shape);
    } else {
        shape.record_primitive(kPrimitiveNull);
        out << "null";
    }
}

void write_optional_zone(CanonicalOutput& out,
                         const std::optional<SemanticZone>& value,
                         SerializationShapeProbe& shape) {
    if (value.has_value()) {
        out << json_escape(semantic_zone_name(*value), shape);
    } else {
        shape.record_primitive(kPrimitiveNull);
        out << "null";
    }
}

void write_optional_u8(CanonicalOutput& out,
                       const std::optional<std::uint8_t>& value,
                       SerializationShapeProbe& shape) {
    if (value.has_value()) {
        shape.record_primitive(kPrimitiveNumeric);
        out << static_cast<unsigned>(*value);
    } else {
        shape.record_primitive(kPrimitiveNull);
        out << "null";
    }
}

template <typename T, typename Writer>
void write_array(CanonicalOutput& out, const std::vector<T>& values, Writer writer) {
    out << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        writer(out, values[index]);
    }
    out << ']';
}

void write_counters(CanonicalOutput& out,
                    const std::vector<Counter>& input,
                    SerializationShapeProbe& shape) {
    auto values = shape.copy(kCopyCounters, input.size(), approximate_fixed_vector_bytes(input),
                             [&input]() { return input; });
    shape.sort(kSortCounters, values.size(), [&values]() {
        std::sort(values.begin(), values.end(), [](const Counter& left, const Counter& right) {
            return std::tie(left.type, left.count) < std::tie(right.type, right.count);
        });
    });
    const auto start = output_offset(out);
    write_array(out, values, [&shape](CanonicalOutput& stream, const Counter& counter) {
        shape.record_primitive(kPrimitiveNumeric);
        shape.record_primitive(kPrimitiveNumeric);
        stream << "{\"type\":" << counter.type << ",\"count\":" << counter.count << '}';
    });
    shape.record_nested_bytes(kNestedCounters, values.size(), output_offset(out) - start);
}

void write_properties(CanonicalOutput& out,
                      const std::optional<CardProperties>& properties,
                      const std::uint8_t property_kind,
                      SerializationShapeProbe& shape) {
    const auto start = output_offset(out);
    if (!properties.has_value()) {
        shape.record_property(property_kind, false, 4);
        out << "null";
        return;
    }
    const auto& value = *properties;
    auto markers = shape.copy(kCopyLinkMarkers, value.link_markers.size(),
                              approximate_fixed_vector_bytes(value.link_markers), [&value]() {
                                  return value.link_markers;
                              });
    shape.sort(kSortLinkMarkers, markers.size(), [&markers]() {
        std::sort(markers.begin(), markers.end(), [](LinkMarker left, LinkMarker right) {
            return static_cast<unsigned>(left) < static_cast<unsigned>(right);
        });
    });
    out << "{\"type\":";
    write_optional_number(out, value.type, shape);
    out << ",\"attribute\":";
    write_optional_number(out, value.attribute, shape);
    out << ",\"race\":";
    write_optional_number(out, value.race, shape);
    out << ",\"attack\":";
    write_optional_number(out, value.attack, shape);
    out << ",\"defense\":";
    write_optional_number(out, value.defense, shape);
    out << ",\"base_attack\":";
    write_optional_number(out, value.base_attack, shape);
    out << ",\"base_defense\":";
    write_optional_number(out, value.base_defense, shape);
    out << ",\"level\":";
    write_optional_number(out, value.level, shape);
    out << ",\"rank\":";
    write_optional_number(out, value.rank, shape);
    out << ",\"link_rating\":";
    write_optional_number(out, value.link_rating, shape);
    out << ",\"link_markers\":";
    const auto marker_start = output_offset(out);
    write_array(out, markers, [&shape](CanonicalOutput& stream, LinkMarker marker) {
        stream << json_escape(link_marker_name(marker), shape);
    });
    shape.record_nested_bytes(kNestedLinkMarkers, markers.size(), output_offset(out) - marker_start);
    out << ",\"left_scale\":";
    write_optional_number(out, value.left_scale, shape);
    out << ",\"right_scale\":";
    write_optional_number(out, value.right_scale, shape);
    out << ",\"status_flags\":";
    write_optional_number(out, value.status_flags, shape);
    out << ",\"counters\":";
    write_counters(out, value.counters, shape);
    out << '}';
    shape.record_property(property_kind, true, output_offset(out) - start);
}

void write_card(CanonicalOutput& out,
                const ObservedCard& card,
                SerializationShapeProbe& shape) {
    out << "{\"locator\":" << json_escape(card.locator.value, shape);
    shape.record_primitive(kPrimitiveBoolean);
    out << ",\"identity_known\":" << (card.identity_known ? "true" : "false")
        << ",\"passcode\":";
    write_optional_number(out, card.passcode, shape);
    out << ",\"owner\":";
    write_optional_u8(out, card.owner, shape);
    out << ",\"controller\":";
    write_optional_u8(out, card.controller, shape);
    out << ",\"zone\":" << json_escape(semantic_zone_name(card.zone), shape) << ",\"sequence\":";
    write_optional_number(out, card.sequence, shape);
    out << ",\"overlay_sequence\":";
    write_optional_number(out, card.overlay_sequence, shape);
    out << ",\"position\":" << json_escape(position_name(card.position), shape);
    shape.record_primitive(kPrimitiveBoolean);
    shape.record_primitive(kPrimitiveBoolean);
    out
        << ",\"face_up\":" << (card.face_up ? "true" : "false")
        << ",\"face_down\":" << (card.face_down ? "true" : "false")
        << ",\"printed\":";
    write_properties(out, card.printed, kNestedPrintedProperties, shape);
    out << ",\"current\":";
    write_properties(out, card.current, kNestedCurrentProperties, shape);
    out << '}';
}

void write_event(CanonicalOutput& out,
                 const VisibleGameEvent& event,
                 SerializationShapeProbe& shape) {
    auto targets = shape.copy(kCopyEventTargets, event.targets.size(),
                              approximate_locator_vector_bytes(event.targets), [&event]() {
                                  return event.targets;
                              });
    shape.sort(kSortEventTargets, targets.size(), [&targets]() {
        std::sort(targets.begin(), targets.end());
    });
    shape.record_primitive(kPrimitiveNumeric);
    shape.record_primitive(kPrimitiveNumeric);
    out << "{\"event_index\":" << event.event_index << ",\"engine_step_index\":"
        << event.engine_step_index << ",\"kind\":" << json_escape(visible_event_kind_name(event.kind), shape)
        << ",\"player\":";
    write_optional_u8(out, event.player, shape);
    out << ",\"entity\":";
    write_optional_locator(out, event.entity, shape);
    out << ",\"public_passcode\":";
    write_optional_number(out, event.public_passcode, shape);
    out << ",\"from_zone\":";
    write_optional_zone(out, event.from_zone, shape);
    out << ",\"to_zone\":";
    write_optional_zone(out, event.to_zone, shape);
    out << ",\"count\":";
    write_optional_number(out, event.count, shape);
    out << ",\"amount\":";
    write_optional_number(out, event.amount, shape);
    out << ",\"counter_type\":";
    write_optional_number(out, event.counter_type, shape);
    out << ",\"phase\":";
    write_optional_number(out, event.phase, shape);
    out << ",\"winner\":";
    write_optional_u8(out, event.winner, shape);
    out << ",\"win_reason\":";
    write_optional_u8(out, event.win_reason, shape);
    out << ",\"effect_description\":";
    write_optional_number(out, event.effect_description, shape);
    out << ",\"targets\":";
    write_array(out, targets, [&shape](CanonicalOutput& stream, const ObservationLocator& locator) {
        stream << json_escape(locator.value, shape);
    });
    out << '}';
}

void write_match_context(CanonicalOutput& out,
                         const MatchContext& context,
                         SerializationShapeProbe& shape) {
    const auto write_deck = [&out, &shape](const StaticDeckContext& deck,
                                           const std::uint8_t copy_kind,
                                           const std::uint8_t nested_kind) {
        const auto deck_start = output_offset(out);
        auto main = shape.copy(copy_kind, deck.main_deck.size(), approximate_fixed_vector_bytes(deck.main_deck),
                               [&deck]() { return deck.main_deck; });
        auto extra = shape.copy(copy_kind, deck.extra_deck.size(), approximate_fixed_vector_bytes(deck.extra_deck),
                                [&deck]() { return deck.extra_deck; });
        shape.sort(kSortDecks, main.size(), [&main]() { std::sort(main.begin(), main.end()); });
        shape.sort(kSortDecks, extra.size(), [&extra]() { std::sort(extra.begin(), extra.end()); });
        shape.record_primitive(kPrimitiveBoolean);
        out << "{\"known\":" << (deck.known ? "true" : "false") << ",\"main_deck\":";
        write_array(out, main, [&shape](CanonicalOutput& stream, std::uint32_t code) {
            shape.record_primitive(kPrimitiveNumeric);
            stream << code;
        });
        out << ",\"extra_deck\":";
        write_array(out, extra, [&shape](CanonicalOutput& stream, std::uint32_t code) {
            shape.record_primitive(kPrimitiveNumeric);
            stream << code;
        });
        out << '}';
        shape.record_nested_bytes(nested_kind, 0, output_offset(out) - deck_start);
    };
    const auto context_start = output_offset(out);
    shape.record_primitive(kPrimitiveNumeric);
    shape.record_primitive(kPrimitiveNumeric);
    shape.record_primitive(kPrimitiveBoolean);
    shape.record_primitive(kPrimitiveBoolean);
    out << "{\"perspective_player\":" << static_cast<unsigned>(context.perspective_player)
        << ",\"duel_flags\":" << context.duel_flags << ",\"knowledge\":{\"own_decklist_known\":"
        << (context.knowledge.own_decklist_known ? "true" : "false")
        << ",\"opponent_decklist_known\":"
        << (context.knowledge.opponent_decklist_known ? "true" : "false") << "},\"own_deck\":";
    const auto own_start = output_offset(out);
    write_deck(context.own_deck, kCopyOwnDeck, kNestedOwnDeck);
    const auto own_end = output_offset(out);
    out << ",\"opponent_deck\":";
    const auto opponent_start = output_offset(out);
    write_deck(context.opponent_deck, kCopyOpponentDeck, kNestedOpponentDeck);
    const auto opponent_end = output_offset(out);
    out << '}';
    const auto total = output_offset(out) - context_start;
    const auto deck_bytes = (own_end - own_start) + (opponent_end - opponent_start);
    shape.record_nested_bytes(kNestedOtherMatchContext, 0, total >= deck_bytes ? total - deck_bytes : 0);
}

std::string serialize_without_hash(
    const PlayerObservation& observation
) {
    SerializationShapeProbe shape;
    shape.set_observation_metadata(observation.perspective_player,
                                   observation.decision_index,
                                   observation.engine_step_index,
                                   observation.chain.length);
    const auto zones_copy_bytes = approximate_fixed_vector_bytes(observation.zones);
    const auto entities_copy_bytes = [&observation]() {
        std::uint64_t bytes = approximate_fixed_vector_bytes(observation.entities);
        return bytes;
    }();
    const auto relationships_copy_bytes = [&observation]() {
        std::uint64_t bytes = approximate_fixed_vector_bytes(observation.relationships);
        return bytes;
    }();
    const auto visible_events_copy_bytes = [&observation]() {
        std::uint64_t bytes = approximate_fixed_vector_bytes(observation.visible_events);
        return bytes;
    }();
    auto zones = shape.copy(kCopyTopLevelZones, observation.zones.size(),
                            zones_copy_bytes,
                            [&observation]() { return observation.zones; });
    shape.record_legacy_copy_event();
    shape.sort(kSortZones, zones.size(), [&zones]() {
        std::sort(zones.begin(), zones.end(), [](const ObservedZone& left, const ObservedZone& right) {
            return std::tie(left.player, left.kind, left.total_count, left.public_identity_count,
                            left.hidden_count) <
                   std::tie(right.player, right.kind, right.total_count, right.public_identity_count,
                            right.hidden_count);
        });
    });
    auto entities = shape.copy(kCopyTopLevelEntities, observation.entities.size(),
                               entities_copy_bytes,
                               [&observation]() { return observation.entities; });
    shape.record_legacy_copy_event();
    shape.sort(kSortEntities, entities.size(), [&entities]() {
        std::sort(entities.begin(), entities.end(), [](const ObservedCard& left, const ObservedCard& right) {
            return left.locator.value < right.locator.value;
        });
    });
    auto relationships = shape.copy(kCopyTopLevelRelationships, observation.relationships.size(),
                                    relationships_copy_bytes,
                                    [&observation]() { return observation.relationships; });
    shape.record_legacy_copy_event();
    shape.sort(kSortRelationships, relationships.size(), [&relationships]() {
        std::sort(relationships.begin(), relationships.end(),
                  [](const Relationship& left, const Relationship& right) {
                      return std::tie(left.kind, left.source.value, left.target.value) <
                             std::tie(right.kind, right.source.value, right.target.value);
                  });
    });
    auto events = shape.copy(kCopyTopLevelVisibleEvents, observation.visible_events.size(),
                             visible_events_copy_bytes,
                             [&observation]() { return observation.visible_events; });
    shape.record_legacy_copy_event();
    shape.sort(kSortVisibleEvents, events.size(), [&events]() {
        std::sort(events.begin(), events.end(), [](const VisibleGameEvent& left, const VisibleGameEvent& right) {
            return std::tie(left.event_index, left.engine_step_index) <
                   std::tie(right.event_index, right.engine_step_index);
        });
    });

    CanonicalOutput out;
    out << "{\"schema_version\":" << json_escape(observation.schema_version, shape)
        << ",\"perspective_player\":" << static_cast<unsigned>(observation.perspective_player)
        << ",\"decision_index\":" << observation.decision_index
        << ",\"engine_step_index\":" << observation.engine_step_index;
    shape.record_primitive(kPrimitiveNumeric);
    shape.record_primitive(kPrimitiveNumeric);
    shape.record_primitive(kPrimitiveNumeric);
    shape.record_section(0, 0, output_offset(out));

    auto section_start = output_offset(out);
    out << ",\"globals\":{";
    shape.record_primitive(kPrimitiveNumeric);
    out << "\"duel_flags\":" << observation.globals.duel_flags << ",\"life_points\":";
    write_array(out, observation.globals.life_points,
                [&shape](CanonicalOutput& stream, std::uint32_t value) {
                    shape.record_primitive(kPrimitiveNumeric);
                    stream << value;
                });
    out << ",\"player_to_act\":";
    write_optional_u8(out, observation.globals.player_to_act, shape);
    out << ",\"turn_player\":";
    write_optional_u8(out, observation.globals.turn_player, shape);
    out << ",\"turn_count\":";
    write_optional_number(out, observation.globals.turn_count, shape);
    out << ",\"phase\":";
    write_optional_number(out, observation.globals.phase, shape);
    shape.record_primitive(kPrimitiveNumeric);
    out << ",\"chain_length\":" << observation.globals.chain_length << ",\"winner\":";
    write_optional_u8(out, observation.globals.winner, shape);
    out << ",\"win_reason\":";
    write_optional_u8(out, observation.globals.win_reason, shape);
    shape.record_primitive(kPrimitiveBoolean);
    out << ",\"terminal\":" << (observation.globals.terminal ? "true" : "false") << '}';
    shape.record_section(1, section_start, output_offset(out));

    section_start = output_offset(out);
    out << ",\"zones\":";
    write_array(out, zones, [&shape](CanonicalOutput& stream, const ObservedZone& zone) {
        for (int index = 0; index < 5; ++index) {
            shape.record_primitive(kPrimitiveNumeric);
        }
        shape.record_primitive(kPrimitiveBoolean);
        stream << "{\"player\":" << static_cast<unsigned>(zone.player) << ",\"kind\":"
               << json_escape(semantic_zone_name(zone.kind), shape) << ",\"total_count\":" << zone.total_count
               << ",\"public_identity_count\":" << zone.public_identity_count
               << ",\"hidden_count\":" << zone.hidden_count
               << ",\"player_observable_order\":"
               << (zone.player_observable_order ? "true" : "false") << '}';
    });
    shape.record_section(2, section_start, output_offset(out));

    section_start = output_offset(out);
    out << ",\"entities\":[";
    for (std::size_t index = 0; index < entities.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        const auto entity_start = output_offset(out);
        write_card(out, entities[index], shape);
        shape.record_entity(output_offset(out) - entity_start);
    }
    out << ']';
    shape.record_section(3, section_start, output_offset(out));

    section_start = output_offset(out);
    out << ",\"relationships\":";
    write_array(out, relationships, [&shape](CanonicalOutput& stream, const Relationship& relationship) {
        stream << "{\"kind\":" << json_escape(relationship_kind_name(relationship.kind), shape)
               << ",\"source\":" << json_escape(relationship.source.value, shape)
               << ",\"target\":" << json_escape(relationship.target.value, shape) << '}';
    });
    shape.record_section(4, section_start, output_offset(out));

    section_start = output_offset(out);
    shape.record_primitive(kPrimitiveNumeric);
    out << ",\"chain\":{\"length\":" << observation.chain.length << ",\"links\":";
    write_array(out, observation.chain.links, [&shape](CanonicalOutput& stream, const ChainLink& link) {
        auto targets = shape.copy(kCopyChainTargets, link.targets.size(),
                                  approximate_locator_vector_bytes(link.targets), [&link]() {
                                      return link.targets;
                                  });
        shape.record_legacy_copy_event();
        shape.sort(kSortChainTargets, targets.size(), [&targets]() {
            std::sort(targets.begin(), targets.end());
        });
        shape.record_primitive(kPrimitiveNumeric);
        stream << "{\"index\":" << link.index << ",\"activating_player\":";
        write_optional_u8(stream, link.activating_player, shape);
        stream << ",\"source\":";
        write_optional_locator(stream, link.source, shape);
        stream << ",\"activation_zone\":";
        write_optional_zone(stream, link.activation_zone, shape);
        stream << ",\"effect_description\":";
        write_optional_number(stream, link.effect_description, shape);
        stream << ",\"targets\":";
        write_array(stream, targets, [&shape](CanonicalOutput& target_stream,
                                             const ObservationLocator& locator) {
            target_stream << json_escape(locator.value, shape);
        });
        stream << '}';
    });
    out << '}';
    shape.record_section(5, section_start, output_offset(out));

    section_start = output_offset(out);
    out << ",\"visible_events\":[";
    for (std::size_t index = 0; index < events.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        const auto event_start = output_offset(out);
        write_event(out, events[index], shape);
        shape.record_visible_event(events[index], output_offset(out) - event_start);
        shape.record_legacy_copy_event();
    }
    out << ']';
    shape.record_section(6, section_start, output_offset(out));

    section_start = output_offset(out);
    out << ",\"decision_context\":{";
    out << "\"decision_id\":";
    write_optional_string(out, observation.decision_context.decision_id, shape);
    out << ",\"kind\":";
    write_optional_string(out, observation.decision_context.kind, shape);
    out << ",\"engine_step_index\":";
    write_optional_number(out, observation.decision_context.engine_step_index, shape);
    out << ",\"player\":";
    write_optional_u8(out, observation.decision_context.player, shape);
    out << ",\"engine_message_type\":";
    write_optional_u8(out, observation.decision_context.engine_message_type, shape);
    out << ",\"engine_message_name\":";
    write_optional_string(out, observation.decision_context.engine_message_name, shape);
    out << ",\"continuation_id\":";
    write_optional_string(out, observation.decision_context.continuation_id, shape);
    out << ",\"referenced_entities\":";
    auto references = shape.copy(kCopyDecisionReferences,
                                observation.decision_context.referenced_entities.size(),
                                approximate_locator_vector_bytes(observation.decision_context.referenced_entities),
                                [&observation]() { return observation.decision_context.referenced_entities; });
    shape.record_legacy_copy_event();
    shape.sort(kSortDecisionReferences, references.size(), [&references]() {
        std::sort(references.begin(), references.end());
    });
    write_array(out, references, [&shape](CanonicalOutput& stream, const ObservationLocator& locator) {
        stream << json_escape(locator.value, shape);
    });
    out << '}';
    shape.record_section(7, section_start, output_offset(out));

    section_start = output_offset(out);
    out << ",\"match_context\":";
    write_match_context(out, observation.match_context, shape);
    out << '}';
    shape.record_section(8, section_start, output_offset(out));
    const auto extraction_start = SerializationClock::now();
    auto serialized = output_extract(out);
    shape.record_final_extraction(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(SerializationClock::now() - extraction_start).count()));
    shape.finish(serialized.size());
    return serialized;
}

}  // namespace

std::string semantic_zone_name(SemanticZone zone) {
    switch (zone) {
    case SemanticZone::MainDeck:
        return "MAIN_DECK";
    case SemanticZone::Hand:
        return "HAND";
    case SemanticZone::MonsterZone:
        return "MONSTER_ZONE";
    case SemanticZone::SpellTrapZone:
        return "SPELL_TRAP_ZONE";
    case SemanticZone::Graveyard:
        return "GRAVEYARD";
    case SemanticZone::Banished:
        return "BANISHED";
    case SemanticZone::ExtraDeck:
        return "EXTRA_DECK";
    case SemanticZone::FieldZone:
        return "FIELD_ZONE";
    case SemanticZone::PendulumRelevant:
        return "PENDULUM_RELEVANT_STATE";
    case SemanticZone::Overlay:
        return "OVERLAY";
    case SemanticZone::Unknown:
        return "UNKNOWN";
    }
    return "UNKNOWN";
}

std::string position_name(Position position) {
    switch (position) {
    case Position::FaceUpAttack:
        return "FACE_UP_ATTACK";
    case Position::FaceDownAttack:
        return "FACE_DOWN_ATTACK";
    case Position::FaceUpDefense:
        return "FACE_UP_DEFENSE";
    case Position::FaceDownDefense:
        return "FACE_DOWN_DEFENSE";
    case Position::Unknown:
        return "UNKNOWN";
    }
    return "UNKNOWN";
}

std::string link_marker_name(LinkMarker marker) {
    switch (marker) {
    case LinkMarker::BottomLeft:
        return "BOTTOM_LEFT";
    case LinkMarker::Bottom:
        return "BOTTOM";
    case LinkMarker::BottomRight:
        return "BOTTOM_RIGHT";
    case LinkMarker::Left:
        return "LEFT";
    case LinkMarker::Right:
        return "RIGHT";
    case LinkMarker::TopLeft:
        return "TOP_LEFT";
    case LinkMarker::Top:
        return "TOP";
    case LinkMarker::TopRight:
        return "TOP_RIGHT";
    }
    return "UNKNOWN";
}

std::string relationship_kind_name(RelationshipKind kind) {
    switch (kind) {
    case RelationshipKind::XyzMaterial:
        return "XYZ_MATERIAL";
    case RelationshipKind::Equip:
        return "EQUIP";
    case RelationshipKind::Target:
        return "TARGET";
    }
    return "TARGET";
}

std::string visible_event_kind_name(VisibleEventKind kind) {
    switch (kind) {
    case VisibleEventKind::TurnStarted:
        return "TURN_STARTED";
    case VisibleEventKind::PhaseChanged:
        return "PHASE_CHANGED";
    case VisibleEventKind::CardMoved:
        return "CARD_MOVED";
    case VisibleEventKind::CardRevealed:
        return "CARD_REVEALED";
    case VisibleEventKind::Summoned:
        return "SUMMONED";
    case VisibleEventKind::Set:
        return "SET";
    case VisibleEventKind::Draw:
        return "DRAW";
    case VisibleEventKind::Shuffle:
        return "SHUFFLE";
    case VisibleEventKind::RandomizationBoundary:
        return "RANDOMIZATION_BOUNDARY";
    case VisibleEventKind::LifePointsChanged:
        return "LIFE_POINTS_CHANGED";
    case VisibleEventKind::ChainActivated:
        return "CHAIN_ACTIVATED";
    case VisibleEventKind::ChainResolved:
        return "CHAIN_RESOLVED";
    case VisibleEventKind::ChainEnded:
        return "CHAIN_ENDED";
    case VisibleEventKind::CardDestroyed:
        return "CARD_DESTROYED";
    case VisibleEventKind::CardBanished:
        return "CARD_BANISHED";
    case VisibleEventKind::CardReturned:
        return "CARD_RETURNED";
    case VisibleEventKind::PositionChanged:
        return "POSITION_CHANGED";
    case VisibleEventKind::CounterChanged:
        return "COUNTER_CHANGED";
    case VisibleEventKind::Equipped:
        return "EQUIPPED";
    case VisibleEventKind::Unequipped:
        return "UNEQUIPPED";
    case VisibleEventKind::Targeted:
        return "TARGETED";
    case VisibleEventKind::Win:
        return "WIN";
    case VisibleEventKind::Unknown:
        return "UNKNOWN";
    }
    return "UNKNOWN";
}

std::string canonical_serialize_without_hash(const PlayerObservation& observation) {
    return serialize_without_hash(observation
    );
}

std::string observation_hash(const PlayerObservation& observation) {
    return ygo::util::sha256_string(canonical_serialize_without_hash(observation));
}

std::string canonical_serialize(const PlayerObservation& observation) {
    const auto without_hash = canonical_serialize_without_hash(observation);
    const auto hash = ygo::util::sha256_string(without_hash);
    return without_hash.substr(0, without_hash.size() - 1) + ",\"observation_hash\":" + json_escape(hash) + "}\n";
}


}  // namespace ygo::observation
