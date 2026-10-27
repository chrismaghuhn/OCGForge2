// Sum selections (e.g. Ritual and Synchro materials): the cards offered at each step must be
// exactly those after which a legal selection can still be completed. Checked against a brute
// force over all subsets on small random instances; the real search must also stay fast on
// large ones (it used to try every subset and never finished with ~20+ cards).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "ocgapi_constants.h"
#include "ygo/protocol/continuation.hpp"
#include "ygo/protocol/protocol_error.hpp"

using namespace ygo::protocol;

namespace {

ContinuationItem item(std::uint32_t index, std::uint32_t primary, std::uint32_t secondary) {
    ContinuationItem result;
    result.source_index = index;
    result.card.code = 1000 + index;
    result.primary_value = primary;
    result.secondary_value = secondary;
    return result;
}

// The rules of sum_terminal, written out plainly.
bool exact_reachable(const std::vector<ContinuationItem>& cards, std::size_t position, std::int64_t remaining) {
    if (remaining <= 0 || position >= cards.size()) {
        return false;
    }
    const auto& c = cards[position];
    if (position == cards.size() - 1) {
        return remaining == c.primary_value || (c.secondary_value > 0 && remaining == c.secondary_value);
    }
    return (remaining > c.primary_value && exact_reachable(cards, position + 1, remaining - c.primary_value)) ||
           (c.secondary_value > 0 && remaining > c.secondary_value &&
            exact_reachable(cards, position + 1, remaining - c.secondary_value));
}

bool legal(const SelectionContinuation& s, const std::vector<std::uint32_t>& chosen) {
    std::vector<ContinuationItem> cards = s.mandatory_items;
    for (const auto index : chosen) {
        cards.push_back(*std::find_if(s.items.begin(), s.items.end(),
                                      [&](const auto& c) { return c.source_index == index; }));
    }
    if (cards.empty()) {
        return false;
    }
    if (s.greater_sum) {
        std::int64_t maximum = 0, minimum = 0, smallest = INT64_MAX;
        for (const auto& c : cards) {
            const std::int64_t low = c.secondary_value == 0 ? c.primary_value : std::min(c.primary_value, c.secondary_value);
            maximum += std::max(c.primary_value, c.secondary_value);
            minimum += low;
            smallest = std::min(smallest, low);
        }
        return maximum >= s.target_sum && minimum - smallest < s.target_sum;
    }
    const auto count = static_cast<std::uint32_t>(chosen.size());
    return count >= s.min_count && count <= s.max_count && exact_reachable(cards, 0, s.target_sum);
}

std::set<std::uint32_t> brute_force_offers(const SelectionContinuation& s) {
    const auto last = s.selected_indices.empty() ? -1 : static_cast<std::int64_t>(s.selected_indices.back());
    std::set<std::uint32_t> offers;
    for (const auto& pick : s.items) {
        if (static_cast<std::int64_t>(pick.source_index) <= last) {
            continue;
        }
        std::vector<std::uint32_t> tail;
        for (const auto& c : s.items) {
            if (c.source_index > pick.source_index) {
                tail.push_back(c.source_index);
            }
        }
        for (std::uint64_t mask = 0; mask < (std::uint64_t{1} << tail.size()); ++mask) {
            auto chosen = s.selected_indices;
            chosen.push_back(pick.source_index);
            for (std::size_t b = 0; b < tail.size(); ++b) {
                if (mask >> b & 1) {
                    chosen.push_back(tail[b]);
                }
            }
            if (legal(s, chosen)) {
                offers.insert(pick.source_index);
                break;
            }
        }
    }
    return offers;
}

std::set<std::uint32_t> offered(const SelectionContinuation& s) {
    DecisionRequest request;
    try {
        request = make_continuation_request(DecisionRequestKind::Sum, 0, MSG_SELECT_SUM, "MSG_SELECT_SUM", 0, s);
    } catch (const ProtocolError&) {
        return {};  // no action at all: nothing offered
    }
    std::set<std::uint32_t> result;
    for (const auto& c : request.candidates) {
        if (c.continuation_operation == ContinuationOperation::Pick) {
            result.insert(*c.choice_index);
        }
    }
    return result;
}

SelectionContinuation random_sum(std::mt19937& rng, std::size_t items, bool greater) {
    SelectionContinuation s;
    s.continuation_id = "sum.test";
    s.continuation_kind = ContinuationKind::Sum;
    s.original_message_type = MSG_SELECT_SUM;
    s.greater_sum = greater;
    s.exact_sum = !greater;
    s.target_sum = 1 + rng() % 16;
    s.min_count = 1 + rng() % 2;
    s.max_count = s.min_count + rng() % 4;
    for (std::uint32_t i = 0; i < items; ++i) {
        s.items.push_back(item(i, 1 + rng() % 8, rng() % 3 == 0 ? 1 + rng() % 8 : 0));
    }
    if (rng() % 4 == 0) {
        s.mandatory_items.push_back(item(100, 1 + rng() % 6, 0));
    }
    // Some cards already picked, in increasing order, as the protocol does.
    for (std::uint32_t i = 0; i < items; ++i) {
        if (rng() % 5 == 0 && s.selected_indices.size() + 1 < s.max_count) {
            s.selected_indices.push_back(i);
        }
    }
    return s;
}

}  // namespace

TEST_CASE("sum selection offers exactly the cards that can still complete a legal selection") {
    std::mt19937 rng(7);
    for (int round = 0; round < 3000; ++round) {
        const auto s = random_sum(rng, 1 + rng() % 9, round % 2 == 0);
        CAPTURE(round, s.greater_sum, s.target_sum, s.min_count, s.max_count, s.selected_indices.size());
        REQUIRE(offered(s) == brute_force_offers(s));
    }
}

TEST_CASE("sum selection stays fast with many cards and no legal completion") {
    // No legal completion among 30 cards: exact needs an odd target from even values, greater
    // needs more than the 30 cards are worth. Trying every subset would take forever.
    for (const bool greater : {false, true}) {
        SelectionContinuation s;
        s.continuation_id = "sum.big";
        s.continuation_kind = ContinuationKind::Sum;
        s.original_message_type = MSG_SELECT_SUM;
        s.greater_sum = greater;
        s.exact_sum = !greater;
        s.target_sum = greater ? 40 : 31;  // exact: odd target from even values is impossible
        s.min_count = 1;
        s.max_count = 30;
        for (std::uint32_t i = 0; i < 30; ++i) {
            s.items.push_back(item(i, greater ? 1 : 2, 0));  // greater: 30 cards of 1 never reach 40
        }
        const auto start = std::chrono::steady_clock::now();
        const auto offers = offered(s);
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        CAPTURE(greater, seconds);
        CHECK(offers.empty());
        CHECK(seconds < 1.0);
    }
}
