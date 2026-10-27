#include "ygo/protocol/response_builder.hpp"

#include <cstddef>

namespace ygo::protocol {
namespace {

template <typename Unsigned>
void append_unsigned(std::vector<std::uint8_t>& output, Unsigned value) {
    for (std::size_t byte = 0; byte < sizeof(Unsigned); ++byte) {
        output.push_back(static_cast<std::uint8_t>((value >> (byte * 8)) & static_cast<Unsigned>(0xff)));
    }
}

}  // namespace

std::vector<std::uint8_t> encode_int32_response(std::int32_t value) {
    std::vector<std::uint8_t> output;
    append_unsigned(output, static_cast<std::uint32_t>(value));
    return output;
}

std::vector<std::uint8_t> encode_uint32_response(std::uint32_t value) {
    std::vector<std::uint8_t> output;
    append_unsigned(output, value);
    return output;
}

std::vector<std::uint8_t> encode_uint64_response(std::uint64_t value) {
    std::vector<std::uint8_t> output;
    append_unsigned(output, value);
    return output;
}

std::vector<std::uint8_t> encode_card_index_response(const std::vector<std::uint32_t>& indices) {
    std::vector<std::uint8_t> output = encode_int32_response(0);
    const auto count = static_cast<std::uint32_t>(indices.size());
    append_unsigned(output, count);
    for (const auto index : indices) {
        append_unsigned(output, index);
    }
    return output;
}

std::vector<std::uint8_t> encode_zone_response(const std::vector<ZoneSlot>& zones) {
    std::vector<std::uint8_t> output;
    output.reserve(zones.size() * 3);
    for (const auto& zone : zones) {
        output.push_back(zone.controller);
        output.push_back(zone.location);
        output.push_back(zone.sequence);
    }
    return output;
}

std::vector<std::uint8_t> encode_counter_response(const std::vector<std::uint16_t>& amounts) {
    std::vector<std::uint8_t> output;
    output.reserve(amounts.size() * 2);
    for (const auto amount : amounts) {
        append_unsigned(output, amount);
    }
    return output;
}

std::vector<std::uint8_t> encode_order_response(const std::vector<std::uint32_t>& indices) {
    std::vector<std::uint8_t> output;
    output.reserve(indices.size());
    for (const auto index : indices) {
        output.push_back(static_cast<std::uint8_t>(index));
    }
    return output;
}

std::vector<std::uint8_t> encode_order_bypass_response() {
    return {0xffu};
}

}  // namespace ygo::protocol
