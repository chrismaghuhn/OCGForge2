#include "ygo/util/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace ygo::util {
namespace {

class Sha256 final {
public:
    Sha256() : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

    void update(const std::uint8_t* data, std::size_t length) {
        total_bytes_ += length;
        while (length != 0) {
            const auto take = std::min(length, block_.size() - block_size_);
            std::copy(data, data + take, block_.begin() + static_cast<std::ptrdiff_t>(block_size_));
            block_size_ += take;
            data += take;
            length -= take;
            if (block_size_ == block_.size()) {
                transform();
                block_size_ = 0;
            }
        }
    }

    std::string finish() {
        const auto bit_length = total_bytes_ * 8;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0);
            transform();
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0);
        for (std::size_t index = 0; index < 8; ++index) {
            block_[63 - index] = static_cast<std::uint8_t>(bit_length >> (index * 8));
        }
        transform();

        static constexpr char hex[] = "0123456789abcdef";
        std::string result;
        result.reserve(64);
        for (const auto word : state_) {
            for (int shift = 28; shift >= 0; shift -= 4) {
                result.push_back(hex[(word >> shift) & 0xfu]);
            }
        }
        return result;
    }

private:
    static std::uint32_t rotr(std::uint32_t value, std::uint32_t amount) {
        return (value >> amount) | (value << (32 - amount));
    }

    void transform() {
        static constexpr std::array<std::uint32_t, 64> k = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
            0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
            0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
            0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u,
            0x106aa070u, 0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u,
            0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u,
            0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        std::array<std::uint32_t, 64> w{};
        for (std::size_t index = 0; index < 16; ++index) {
            w[index] = (static_cast<std::uint32_t>(block_[index * 4]) << 24) |
                       (static_cast<std::uint32_t>(block_[index * 4 + 1]) << 16) |
                       (static_cast<std::uint32_t>(block_[index * 4 + 2]) << 8) |
                       static_cast<std::uint32_t>(block_[index * 4 + 3]);
        }
        for (std::size_t index = 16; index < w.size(); ++index) {
            const auto s0 = rotr(w[index - 15], 7) ^ rotr(w[index - 15], 18) ^ (w[index - 15] >> 3);
            const auto s1 = rotr(w[index - 2], 17) ^ rotr(w[index - 2], 19) ^ (w[index - 2] >> 10);
            w[index] = w[index - 16] + s0 + w[index - 7] + s1;
        }
        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t index = 0; index < w.size(); ++index) {
            const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const auto choose = (e & f) ^ ((~e) & g);
            const auto temp1 = h + s1 + choose + k[index] + w[index];
            const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

}  // namespace

std::string sha256_bytes(const std::vector<std::uint8_t>& bytes) {
    Sha256 digest;
    digest.update(bytes.data(), bytes.size());
    return digest.finish();
}

std::string sha256_string(std::string_view value) {
    Sha256 digest;
    digest.update(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
    return digest.finish();
}

}  // namespace ygo::util
