#include "encoding.h"

#include <cstdio>
#include <cstring>

namespace mfenv {

void AppendJsonString(const std::string& value, std::string* out) {
  static const char* kHex = "0123456789abcdef";
  out->push_back('"');
  for (unsigned char c : value) {
    switch (c) {
      case '"': *out += "\\\""; break;
      case '\\': *out += "\\\\"; break;
      case '\n': *out += "\\n"; break;
      case '\r': *out += "\\r"; break;
      case '\t': *out += "\\t"; break;
      case '\b': *out += "\\b"; break;
      case '\f': *out += "\\f"; break;
      default:
        if (c < 0x20 || c > 0x7E) {
          *out += "\\u00";
          out->push_back(kHex[(c >> 4) & 0xF]);
          out->push_back(kHex[c & 0xF]);
        } else {
          out->push_back(static_cast<char>(c));
        }
    }
  }
  out->push_back('"');
}

void AppendFloatBits(float value, std::string* out) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  char buf[12];
  std::snprintf(buf, sizeof(buf), "%u", bits);
  *out += buf;
}

void AppendInt64List(const std::vector<int64_t>& values, std::string* out) {
  *out += '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) *out += ',';
    *out += std::to_string(values[i]);
  }
  *out += ']';
}

void AppendUint32List(const std::vector<uint32_t>& values, std::string* out) {
  *out += '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) *out += ',';
    *out += std::to_string(static_cast<int64_t>(values[i]));
  }
  *out += ']';
}

void AppendBoolList(const std::vector<uint8_t>& values, std::string* out) {
  *out += '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) *out += ',';
    *out += values[i] ? "true" : "false";
  }
  *out += ']';
}

void AppendFloatBitsList(const std::vector<float>& values, std::string* out) {
  *out += '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) *out += ',';
    AppendFloatBits(values[i], out);
  }
  *out += ']';
}

}  // namespace mfenv
