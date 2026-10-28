#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "decoded_queries.hpp"
#include "card_data_reader.hpp"

namespace ygo::client {

// The field as one player of an EDOPro duel knows it, kept from the game messages the server sends
// that player, and answering observation building like the engine would (detail::DecodedQueries).
//
// The server refreshes card data (MSG_UPDATE_DATA / MSG_UPDATE_CARD) for hand and field before
// every idle/battle question and after chains, and for the destination of each move; between
// refreshes this class applies moves, draws, swaps, counters, life points and the chain itself, as
// EDOPro's own client does. Its refreshes leave out owner, Xyz materials and counters, so those are
// tracked from the messages: a card keeps its owner as it moves (cards start owned by the player
// whose zone they appear in), materials follow moves into and out of LOCATION_OVERLAY.
class ClientView final : public observation::detail::DecodedQueries {
public:
    ClientView(std::uint32_t duel_flags, std::shared_ptr<const core::detail::CardDataStore> cards);

    // One game message (type byte + payload). Messages it does not track are ignored.
    void apply(const std::vector<std::uint8_t>& message);

    observation::detail::RawFieldSnapshot field() const override;
    observation::detail::RawLocationSnapshot location(std::uint8_t controller, std::uint32_t location,
                                                      std::uint32_t flags) const override;
    observation::detail::RawCardQuery card(std::uint8_t controller, std::uint32_t location, std::uint32_t sequence,
                                           std::optional<std::uint32_t> overlay_sequence,
                                           std::uint32_t flags) const override;
    std::optional<core::StaticCardData> printed(std::uint32_t code) const override;

    // The field as bytes (life points, every card's place, code and position, the chain): equal
    // whenever the duel is back in the same state, for loop detection.
    std::vector<std::uint8_t> digest() const;

private:
    struct Card {
        observation::detail::RawCardQuery query;  // as last refreshed or moved; no owner/materials/counters
        std::uint8_t owner = 0;
        std::vector<std::uint32_t> materials;  // passcodes, 0 = unknown
        std::vector<observation::detail::RawCounter> counters;
    };
    struct Side {
        std::uint32_t life_points = 8000;
        std::uint32_t deck = 0;
        std::array<std::optional<Card>, 7> monsters;
        std::array<std::optional<Card>, 8> spells;
        std::vector<Card> hand, graveyard, banished, extra;
    };
    struct Location {
        std::uint8_t controller = 0;
        std::uint32_t location = 0;
        std::uint32_t sequence = 0;
        std::uint32_t position = 0;
    };

    std::optional<Card>* slot(std::uint8_t controller, std::uint32_t location, std::uint32_t sequence);
    const std::optional<Card>* slot(std::uint8_t controller, std::uint32_t location, std::uint32_t sequence) const;
    std::vector<Card>* pile(std::uint8_t controller, std::uint32_t location);
    const std::vector<Card>* pile(std::uint8_t controller, std::uint32_t location) const;
    std::optional<Card> take(const Location& from);
    void put(const Location& to, Card card);
    void refresh(std::uint8_t controller, std::uint32_t location, const observation::detail::RawLocationSnapshot& data);
    observation::detail::RawCardQuery query_of(const Card& card) const;

    std::uint32_t duel_flags_;
    std::shared_ptr<const core::detail::CardDataStore> cards_;
    std::array<Side, 2> sides_;
    std::vector<observation::detail::RawChainLink> chain_;
};

}  // namespace ygo::client
