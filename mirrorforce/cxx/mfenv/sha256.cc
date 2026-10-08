#include "sha256.h"

#include <cstring>
#include <vector>

namespace mfenv {
namespace {

constexpr uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t Ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void Block(uint32_t state[8], const uint8_t* p) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i)
    w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
           (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = Ror(w[i - 15], 7) ^ Ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = Ror(w[i - 2], 17) ^ Ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t s1 = Ror(e, 6) ^ Ror(e, 11) ^ Ror(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = h + s1 + ch + K[i] + w[i];
    const uint32_t s0 = Ror(a, 2) ^ Ror(a, 13) ^ Ror(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }
  state[0] += a; state[1] += b; state[2] += c; state[3] += d;
  state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

}  // namespace

std::string Sha256Hex(const std::string& data) {
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
  size_t n = data.size();
  size_t i = 0;
  for (; i + 64 <= n; i += 64) Block(state, p + i);

  uint8_t tail[128];
  const size_t rest = n - i;
  std::memcpy(tail, p + i, rest);
  tail[rest] = 0x80;
  size_t pad = (rest < 56) ? (56 - rest - 1) : (120 - rest - 1);
  std::memset(tail + rest + 1, 0, pad);
  const uint64_t bits = static_cast<uint64_t>(n) * 8;
  const size_t total = rest + 1 + pad + 8;
  for (int b = 0; b < 8; ++b)
    tail[total - 1 - b] = static_cast<uint8_t>((bits >> (8 * b)) & 0xFF);
  for (size_t off = 0; off < total; off += 64) Block(state, tail + off);

  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (int w = 0; w < 8; ++w)
    for (int b = 3; b >= 0; --b) {
      const uint8_t byte = static_cast<uint8_t>((state[w] >> (8 * b)) & 0xFF);
      out.push_back(kHex[byte >> 4]);
      out.push_back(kHex[byte & 0xF]);
    }
  return out;
}

}  // namespace mfenv
