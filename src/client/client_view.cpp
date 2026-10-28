#include "client_view.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "ocgapi_constants.h"

namespace ygo::client {

using observation::detail::RawCardQuery;
using observation::detail::RawChainLink;
using observation::detail::RawCounter;
using observation::detail::RawFieldSnapshot;
using observation::detail::RawLocationSnapshot;

namespace {

class Reader final {
public:
    explicit Reader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}
    std::uint8_t u8() { return static_cast<std::uint8_t>(read(1)); }
    std::uint16_t u16() { return static_cast<std::uint16_t>(read(2)); }
    std::uint32_t u32() { return static_cast<std::uint32_t>(read(4)); }
    std::uint64_t u64() { return read(8); }
    std::vector<std::uint8_t> rest() {
        std::vector<std::uint8_t> out(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_), bytes_.end());
        offset_ = bytes_.size();
        return out;
    }

private:
    std::uint64_t read(std::size_t count) {
        if (bytes_.size() - offset_ < count) {
            throw std::runtime_error("truncated game message");
        }
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < count; ++i) {
            value |= static_cast<std::uint64_t>(bytes_[offset_++]) << (8 * i);
        }
        return value;
    }
    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_ = 0;
};

}  // namespace

ClientView::ClientView(std::uint32_t duel_flags, std::shared_ptr<const core::detail::CardDataStore> cards)
    : duel_flags_(duel_flags), cards_(std::move(cards)) {}

std::optional<ClientView::Card>* ClientView::slot(std::uint8_t controller, std::uint32_t location,
                                                   std::uint32_t sequence) {
    auto& side = sides_.at(controller & 1);
    if (location == LOCATION_MZONE && sequence < side.monsters.size()) {
        return &side.monsters[sequence];
    }
    if (location == LOCATION_SZONE && sequence < side.spells.size()) {
        return &side.spells[sequence];
    }
    return nullptr;
}

const std::optional<ClientView::Card>* ClientView::slot(std::uint8_t controller, std::uint32_t location,
                                                         std::uint32_t sequence) const {
    return const_cast<ClientView*>(this)->slot(controller, location, sequence);
}

std::vector<ClientView::Card>* ClientView::pile(std::uint8_t controller, std::uint32_t location) {
    auto& side = sides_.at(controller & 1);
    switch (location) {
    case LOCATION_HAND: return &side.hand;
    case LOCATION_GRAVE: return &side.graveyard;
    case LOCATION_REMOVED: return &side.banished;
    case LOCATION_EXTRA: return &side.extra;
    default: return nullptr;
    }
}

const std::vector<ClientView::Card>* ClientView::pile(std::uint8_t controller, std::uint32_t location) const {
    return const_cast<ClientView*>(this)->pile(controller, location);
}

// Removes the card at `from` and returns it (nothing for the deck, a token's creation or a slot
// this view had no card in).
std::optional<ClientView::Card> ClientView::take(const Location& from) {
    if (from.location == 0) {
        return std::nullopt;
    }
    if (from.location & LOCATION_OVERLAY) {
        if (auto* parent = slot(from.controller, LOCATION_MZONE, from.sequence); parent && *parent) {
            auto& materials = (*parent)->materials;
            if (from.position < materials.size()) {
                Card card;
                card.owner = from.controller;
                card.query.code = materials[from.position];
                materials.erase(materials.begin() + from.position);
                return card;
            }
        }
        return std::nullopt;
    }
    if (from.location == LOCATION_DECK) {
        auto& deck = sides_.at(from.controller & 1).deck;
        deck -= deck > 0 ? 1 : 0;
        Card card;
        card.owner = from.controller;
        return card;
    }
    if (auto* field = slot(from.controller, from.location, from.sequence)) {
        auto card = std::move(*field);
        field->reset();
        return card;
    }
    if (auto* cards = pile(from.controller, from.location); cards && from.sequence < cards->size()) {
        auto card = std::move((*cards)[from.sequence]);
        cards->erase(cards->begin() + from.sequence);
        return card;
    }
    return std::nullopt;
}

void ClientView::put(const Location& to, Card card) {
    card.query.position = to.position;
    if (to.location & LOCATION_OVERLAY) {
        if (auto* parent = slot(to.controller, LOCATION_MZONE, to.sequence); parent && *parent) {
            auto& materials = (*parent)->materials;
            const auto at = std::min<std::size_t>(to.position, materials.size());
            materials.insert(materials.begin() + static_cast<std::ptrdiff_t>(at), card.query.code.value_or(0));
        }
        return;
    }
    card.materials.clear();  // an Xyz Monster's materials are detached by their own moves
    card.counters.clear();
    if (to.location == LOCATION_DECK) {
        ++sides_.at(to.controller & 1).deck;
        return;
    }
    if (auto* field = slot(to.controller, to.location, to.sequence)) {
        *field = std::move(card);
        return;
    }
    if (auto* cards = pile(to.controller, to.location)) {
        const auto at = std::min<std::size_t>(to.sequence, cards->size());
        cards->insert(cards->begin() + static_cast<std::ptrdiff_t>(at), std::move(card));
    }
}

void ClientView::refresh(std::uint8_t controller, std::uint32_t location, const RawLocationSnapshot& data) {
    if (location == LOCATION_MZONE || location == LOCATION_SZONE) {
        const auto count = location == LOCATION_MZONE ? 7u : 8u;
        for (std::uint32_t sequence = 0; sequence < count; ++sequence) {
            auto& field = *slot(controller, location, sequence);
            const auto& entry = sequence < data.entries.size() ? data.entries[sequence] : std::nullopt;
            if (!entry) {
                field.reset();
            } else if (field) {
                field->query = *entry;
            } else {
                field = Card{*entry, controller, {}, {}};
            }
        }
        return;
    }
    if (location == LOCATION_DECK) {
        sides_.at(controller & 1).deck = static_cast<std::uint32_t>(data.entries.size());
        return;
    }
    if (auto* cards = pile(controller, location)) {
        cards->resize(data.entries.size(), Card{{}, controller, {}, {}});
        for (std::size_t i = 0; i < data.entries.size(); ++i) {
            (*cards)[i].query = data.entries[i].value_or(RawCardQuery{});
        }
    }
}

void ClientView::apply(const std::vector<std::uint8_t>& message) {
    if (message.empty()) {
        return;
    }
    Reader reader(message);
    const auto type = reader.u8();
    const auto read_location = [&reader] {
        Location location;
        location.controller = reader.u8();
        location.location = reader.u8();
        location.sequence = reader.u32();
        location.position = reader.u32();
        return location;
    };
    switch (type) {
    case MSG_START: {  // EDOPro's: perspective, LP, deck and extra deck sizes
        (void)reader.u8();
        sides_[0].life_points = reader.u32();
        sides_[1].life_points = reader.u32();
        sides_[0].deck = reader.u16();
        (void)reader.u16();  // extra deck: refreshed right after
        sides_[1].deck = reader.u16();
        break;
    }
    case MSG_UPDATE_DATA: {
        const auto controller = reader.u8();
        const auto location = reader.u8();
        refresh(controller, location, observation::detail::decode_location_query(reader.rest()));
        break;
    }
    case MSG_UPDATE_CARD: {
        const auto controller = reader.u8();
        const auto location = reader.u8();
        const auto sequence = reader.u8();
        auto query = observation::detail::decode_card_query(reader.rest());
        if (auto* field = slot(controller, location, sequence)) {
            if (*field) {
                (*field)->query = std::move(query);
            } else {
                *field = Card{std::move(query), controller, {}, {}};
            }
        } else if (auto* cards = pile(controller, location); cards && sequence < cards->size()) {
            (*cards)[sequence].query = std::move(query);
        }
        break;
    }
    case MSG_MOVE: {
        const auto code = reader.u32();
        const auto from = read_location();
        const auto to = read_location();
        // A chain link's card can move while the link is open (sent as cost, say); the engine
        // reports where it is now.
        for (auto& link : chain_) {
            if (link.source.controller == from.controller && link.source.location == from.location &&
                link.source.sequence == from.sequence) {
                link.source = {to.controller, to.location, to.sequence, to.position};
            }
        }
        auto card = take(from).value_or(Card{{}, to.controller, {}, {}});
        if (code != 0) {
            card.query.code = code;
        }
        if (to.location != 0) {
            put(to, std::move(card));
        }
        break;
    }
    case MSG_DRAW: {
        const auto player = reader.u8() & 1;
        const auto count = reader.u32();
        for (std::uint32_t i = 0; i < count; ++i) {
            Card card;
            card.owner = player;
            if (const auto code = reader.u32(); code != 0) {
                card.query.code = code;
            }
            card.query.position = reader.u32();
            sides_[player].deck -= sides_[player].deck > 0 ? 1 : 0;
            sides_[player].hand.push_back(std::move(card));
        }
        break;
    }
    case MSG_SWAP: {
        (void)reader.u32();
        const auto first = read_location();
        (void)reader.u32();
        const auto second = read_location();
        auto* a = slot(first.controller, first.location, first.sequence);
        auto* b = slot(second.controller, second.location, second.sequence);
        if (a && b) {
            std::swap(*a, *b);
        }
        break;
    }
    case MSG_POS_CHANGE: {
        (void)reader.u32();
        const auto controller = reader.u8();
        const auto location = reader.u8();
        const auto sequence = reader.u8();
        (void)reader.u8();
        const auto position = reader.u8();
        if (auto* field = slot(controller, location, sequence); field && *field) {
            (*field)->query.position = position;
        }
        break;
    }
    case MSG_ADD_COUNTER:
    case MSG_REMOVE_COUNTER: {
        const auto counter = reader.u16();
        const auto controller = reader.u8();
        const auto location = reader.u8();
        const auto sequence = reader.u8();
        const auto count = reader.u16();
        if (auto* field = slot(controller, location, sequence); field && *field) {
            auto& counters = (*field)->counters;
            auto it = std::find_if(counters.begin(), counters.end(),
                                   [&](const RawCounter& c) { return c.type == counter; });
            if (type == MSG_ADD_COUNTER) {
                if (it == counters.end()) {
                    counters.push_back({counter, count});
                } else {
                    it->count += count;
                }
            } else if (it != counters.end()) {
                it->count = it->count > count ? it->count - count : 0;
                if (it->count == 0) {
                    counters.erase(it);
                }
            }
        }
        break;
    }
    case MSG_DAMAGE:
    case MSG_PAY_LPCOST: {
        auto& side = sides_.at(reader.u8() & 1);
        const auto amount = reader.u32();
        side.life_points = side.life_points > amount ? side.life_points - amount : 0;
        break;
    }
    case MSG_RECOVER: {
        auto& side = sides_.at(reader.u8() & 1);
        side.life_points += reader.u32();
        break;
    }
    case MSG_LPUPDATE: {
        auto& side = sides_.at(reader.u8() & 1);
        side.life_points = reader.u32();
        break;
    }
    case MSG_CHAINING: {
        RawChainLink link;
        link.source_code = reader.u32();
        const auto source = read_location();
        link.source = {source.controller, source.location, source.sequence, source.position};
        link.triggering_player = reader.u8();
        link.triggering_location = reader.u8();
        link.triggering_sequence = reader.u32();
        link.description = reader.u64();
        chain_.push_back(link);
        break;
    }
    case MSG_CHAIN_SOLVED:
        if (!chain_.empty()) {
            chain_.pop_back();
        }
        break;
    case MSG_CHAIN_END:
        chain_.clear();
        break;
    case MSG_SWAP_GRAVE_DECK: {  // the graveyard is refreshed right after
        const auto player = reader.u8() & 1;
        sides_[player].deck = reader.u32();
        break;
    }
    default:
        break;
    }
}

RawCardQuery ClientView::query_of(const Card& card) const {
    auto query = card.query;
    query.owner = card.owner;
    // Graveyard and Extra Deck refreshes carry no link rating/markers; they never change there.
    if (!query.link_rating && query.code) {
        if (const auto data = printed(*query.code); data && (data->type & TYPE_LINK) != 0) {
            query.link_rating = data->level;
            query.link_markers = data->link_marker;
        }
    }
    query.overlay_codes = card.materials;
    query.counters = card.counters;
    return query;
}

RawFieldSnapshot ClientView::field() const {
    RawFieldSnapshot field;
    field.duel_options = duel_flags_;
    for (std::size_t player = 0; player < 2; ++player) {
        const auto& side = sides_[player];
        auto& out = field.players[player];
        out.life_points = side.life_points;
        for (std::size_t i = 0; i < side.monsters.size(); ++i) {
            if (const auto& card = side.monsters[i]) {
                out.monster_slots[i] = {true, static_cast<std::uint8_t>(card->query.position.value_or(0)),
                                        static_cast<std::uint32_t>(card->materials.size())};
            }
        }
        for (std::size_t i = 0; i < side.spells.size(); ++i) {
            if (const auto& card = side.spells[i]) {
                out.spell_trap_slots[i] = {true, static_cast<std::uint8_t>(card->query.position.value_or(0)), 0};
            }
        }
        out.main_deck_count = side.deck;
        out.hand_count = static_cast<std::uint32_t>(side.hand.size());
        out.graveyard_count = static_cast<std::uint32_t>(side.graveyard.size());
        out.banished_count = static_cast<std::uint32_t>(side.banished.size());
        out.extra_deck_count = static_cast<std::uint32_t>(side.extra.size());
        out.face_up_extra_deck_count = static_cast<std::uint32_t>(
            std::count_if(side.extra.begin(), side.extra.end(),
                          [](const Card& c) { return (c.query.position.value_or(0) & POS_FACEUP) != 0; }));
    }
    field.chain = chain_;
    return field;
}

RawLocationSnapshot ClientView::location(std::uint8_t controller, std::uint32_t location,
                                         std::uint32_t /*flags*/) const {
    RawLocationSnapshot snapshot;
    const auto& side = sides_.at(controller & 1);
    if (location == LOCATION_MZONE || location == LOCATION_SZONE) {
        const auto count = location == LOCATION_MZONE ? side.monsters.size() : side.spells.size();
        for (std::uint32_t sequence = 0; sequence < count; ++sequence) {
            const auto& card = *slot(controller, location, sequence);
            snapshot.entries.push_back(card ? std::optional(query_of(*card)) : std::nullopt);
        }
    } else if (const auto* cards = pile(controller, location)) {
        for (const auto& card : *cards) {
            auto query = query_of(card);
            // Off the field nothing stays disabled; refreshes there are rare, so the status a card
            // had when it left the field would linger.
            if (query.status) {
                *query.status &= ~static_cast<std::uint32_t>(STATUS_DISABLED);
            }
            snapshot.entries.push_back(std::move(query));
        }
    }
    return snapshot;
}

RawCardQuery ClientView::card(std::uint8_t controller, std::uint32_t location, std::uint32_t sequence,
                              std::optional<std::uint32_t> overlay_sequence, std::uint32_t /*flags*/) const {
    if (const auto* field = slot(controller, location, sequence); field && *field) {
        if (!overlay_sequence) {
            return query_of(**field);
        }
        const auto& materials = (*field)->materials;
        RawCardQuery material;
        if (*overlay_sequence < materials.size() && materials[*overlay_sequence] != 0) {
            material.code = materials[*overlay_sequence];
            material.position = POS_FACEUP;
        }
        return material;
    }
    if (const auto* cards = pile(controller, location); cards && sequence < cards->size() && !overlay_sequence) {
        return query_of((*cards)[sequence]);
    }
    return {};
}

std::vector<std::uint8_t> ClientView::digest() const {
    std::vector<std::uint8_t> out;
    const auto put32 = [&out](std::uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8) {
            out.push_back(static_cast<std::uint8_t>(value >> shift));
        }
    };
    const auto put_card = [&](const Card& card) {
        put32(card.query.code.value_or(0));
        put32(card.query.position.value_or(0));
        put32(static_cast<std::uint32_t>(card.materials.size()));
    };
    for (const auto& side : sides_) {
        put32(side.life_points);
        put32(side.deck);
        for (const auto& slot : side.monsters) {
            slot ? put_card(*slot) : put32(0xFFFFFFFF);
        }
        for (const auto& slot : side.spells) {
            slot ? put_card(*slot) : put32(0xFFFFFFFF);
        }
        for (const auto* cards : {&side.hand, &side.graveyard, &side.banished, &side.extra}) {
            put32(static_cast<std::uint32_t>(cards->size()));
            for (const auto& card : *cards) {
                put_card(card);
            }
        }
    }
    for (const auto& link : chain_) {
        put32(link.source_code);
    }
    return out;
}

std::optional<core::StaticCardData> ClientView::printed(std::uint32_t code) const {
    return cards_ ? cards_->snapshot(code) : std::nullopt;
}

}  // namespace ygo::client
