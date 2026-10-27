#pragma once

#include <cstdint>
#include <vector>

namespace ygo::protocol {

struct ZoneSlot {
    std::uint8_t controller = 0;
    std::uint8_t location = 0;
    std::uint8_t sequence = 0;
};

std::vector<std::uint8_t> encode_int32_response(std::int32_t value);
std::vector<std::uint8_t> encode_uint32_response(std::uint32_t value);
std::vector<std::uint8_t> encode_uint64_response(std::uint64_t value);
std::vector<std::uint8_t> encode_card_index_response(const std::vector<std::uint32_t>& indices);
std::vector<std::uint8_t> encode_zone_response(const std::vector<ZoneSlot>& zones);
std::vector<std::uint8_t> encode_counter_response(const std::vector<std::uint16_t>& amounts);
std::vector<std::uint8_t> encode_order_response(const std::vector<std::uint32_t>& indices);
std::vector<std::uint8_t> encode_order_bypass_response();

}  // namespace ygo::protocol
