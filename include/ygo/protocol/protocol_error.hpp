#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ygo::protocol {

enum class ProtocolErrorCode {
    MalformedMessage,
    UnsupportedDecision,
    InvalidSemanticKey,
    IncompleteCandidates,
};

class ProtocolError final : public std::runtime_error {
public:
    ProtocolError(ProtocolErrorCode code, std::string context, std::uint8_t message_type = 0,
                  std::uint8_t player = 255, std::vector<std::uint8_t> raw_message = {})
        : std::runtime_error(std::move(context)), code_(code), message_type_(message_type), player_(player),
          raw_message_(std::move(raw_message)) {}

    ProtocolErrorCode code() const noexcept { return code_; }
    std::uint8_t message_type() const noexcept { return message_type_; }
    std::uint8_t player() const noexcept { return player_; }
    const std::vector<std::uint8_t>& raw_message() const noexcept { return raw_message_; }

private:
    ProtocolErrorCode code_;
    std::uint8_t message_type_;
    std::uint8_t player_;
    std::vector<std::uint8_t> raw_message_;
};

}  // namespace ygo::protocol
