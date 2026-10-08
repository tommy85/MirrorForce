#include "semantics.h"

#include "encoding.h"

#include <cstring>
#include <fstream>
#include <mutex>

namespace mfenv {
namespace {

constexpr char kMagic[8] = {'M', 'F', 'E', 'N', 'V', 'S', 'E', 'M'};
constexpr uint32_t kVersion = 1;

class Cursor {
 public:
  Cursor(const std::vector<uint8_t>& data, const std::string& path)
      : data_(data), path_(path) {}

  void Bytes(void* out, size_t n) {
    if (pos_ + n > data_.size())
      throw SemanticsError(path_ + ": truncated at offset " +
                           std::to_string(pos_));
    std::memcpy(out, data_.data() + pos_, n);
    pos_ += n;
  }
  uint32_t U32() { uint32_t v; Bytes(&v, 4); return v; }
  int32_t I32() { int32_t v; Bytes(&v, 4); return v; }
  std::string Fixed(size_t width) {
    std::vector<char> buf(width);
    Bytes(buf.data(), width);
    size_t len = 0;
    while (len < width && buf[len] != '\0') ++len;
    return std::string(buf.data(), len);
  }
  size_t remaining() const { return data_.size() - pos_; }

 private:
  const std::vector<uint8_t>& data_;
  std::string path_;
  size_t pos_ = 0;
};

uint64_t IdentityKey(uint32_t code, int32_t index) {
  return (static_cast<uint64_t>(code) << 8) |
         static_cast<uint64_t>(static_cast<uint32_t>(index + 1) & 0xFFu);
}

uint64_t DescKey(uint32_t code, uint32_t desc) {
  return (static_cast<uint64_t>(code) << 32) | static_cast<uint64_t>(desc);
}

std::unique_ptr<SemanticTable> g_default;
std::string g_default_path;
std::mutex g_default_mutex;

}  // namespace

const char* EffectResolutionName(EffectResolution value) {
  switch (value) {
    case EffectResolution::EXACT: return "exact";
    case EffectResolution::NO_CARD: return "no_card";
    case EffectResolution::NO_DESCRIPTION: return "no_description";
    case EffectResolution::UNMAPPED_DESCRIPTION: return "unmapped_description";
    case EffectResolution::AMBIGUOUS_DESCRIPTION: return "ambiguous_description";
    case EffectResolution::MISSING_EFFECT_ARTIFACT: return "missing_effect_artifact";
  }
  return "?";
}

SemanticTable::SemanticTable(const std::unordered_map<uint32_t, int64_t>& card_rows,
                             const std::string& artifact_fingerprint)
    : artifact_fingerprint_(artifact_fingerprint), card_row_by_code_(card_rows) {}

SemanticTable::SemanticTable(const std::string& path) : path_(path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw SemanticsError("cannot open semantics sidecar " + path);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
  Cursor cursor(data, path);
  char magic[8];
  cursor.Bytes(magic, 8);
  if (std::memcmp(magic, kMagic, 8) != 0)
    throw SemanticsError(path + ": not an mfenv semantics sidecar");
  const uint32_t version = cursor.U32();
  if (version != kVersion)
    throw SemanticsError(path + ": sidecar version " + std::to_string(version) +
                         ", expected " + std::to_string(kVersion));
  description_limit_ = static_cast<int>(cursor.U32());
  script_revision_ = cursor.Fixed(64);
  artifact_fingerprint_ = cursor.Fixed(64);
  const uint32_t n_card = cursor.U32();
  const uint32_t n_effect = cursor.U32();
  const uint32_t n_desc = cursor.U32();
  const uint32_t n_amb = cursor.U32();
  const uint32_t n_amb_index = cursor.U32();

  card_row_by_code_.reserve(n_card * 2);
  for (uint32_t i = 0; i < n_card; ++i) {
    const uint32_t code = cursor.U32();
    const uint32_t row = cursor.U32();
    card_row_by_code_[code] = static_cast<int64_t>(row);
  }
  effect_row_by_identity_.reserve(n_effect * 2);
  for (uint32_t i = 0; i < n_effect; ++i) {
    const uint32_t code = cursor.U32();
    const int32_t index = cursor.I32();
    const uint32_t row = cursor.U32();
    effect_row_by_identity_[IdentityKey(code, index)] = static_cast<int64_t>(row);
    /* the sidecar is written sorted by (code, index), so appending here
     * reproduces Python's ``_effect_rows_by_card`` order exactly */
    effect_rows_by_card_[code].push_back(static_cast<int64_t>(row));
  }
  desc_unique_.reserve(n_desc * 2);
  for (uint32_t i = 0; i < n_desc; ++i) {
    const uint32_t code = cursor.U32();
    const uint32_t desc = cursor.U32();
    const int32_t index = cursor.I32();
    desc_unique_[DescKey(code, desc)] = index;
  }
  desc_ambiguous_.reserve(n_amb * 2);
  for (uint32_t i = 0; i < n_amb; ++i) {
    const uint32_t code = cursor.U32();
    const uint32_t desc = cursor.U32();
    const uint32_t off = cursor.U32();
    const uint32_t count = cursor.U32();
    desc_ambiguous_[DescKey(code, desc)] = {off, count};
  }
  ambiguous_index_.resize(n_amb_index);
  for (uint32_t i = 0; i < n_amb_index; ++i) ambiguous_index_[i] = cursor.I32();
  if (cursor.remaining() != 0)
    throw SemanticsError(path + ": " + std::to_string(cursor.remaining()) +
                         " trailing bytes");
}

const std::vector<int64_t>& SemanticTable::EffectRowsForCard(uint32_t code) const {
  static const std::vector<int64_t> kEmpty;
  auto hit = effect_rows_by_card_.find(code);
  return hit == effect_rows_by_card_.end() ? kEmpty : hit->second;
}

std::vector<int64_t> SemanticTable::CandidatesForDesc(uint32_t code,
                                                      uint32_t desc) const {
  std::vector<int64_t> out;
  if (!desc) return out;
  const uint64_t key = DescKey(code, desc);
  auto unique = desc_unique_.find(key);
  if (unique != desc_unique_.end()) {
    out.push_back(unique->second);
    return out;
  }
  auto amb = desc_ambiguous_.find(key);
  if (amb != desc_ambiguous_.end())
    for (uint32_t i = 0; i < amb->second.second; ++i)
      out.push_back(ambiguous_index_[amb->second.first + i]);
  return out;
}

int64_t SemanticTable::CardRow(uint32_t code) const {
  auto hit = card_row_by_code_.find(code);
  return hit == card_row_by_code_.end() ? 0 : hit->second;
}

int64_t SemanticTable::Passcode(const LegalAction& action) {
  int64_t code = static_cast<int64_t>(action.code);
  if (code <= 0) code = static_cast<int64_t>(action.response);
  return code > 0 ? code : 0;
}

ActionSemanticResolution SemanticTable::Resolve(const LegalAction& action) const {
  ActionSemanticResolution out;
  const int64_t code = Passcode(action);
  int64_t card_row = 0;
  if (code) {
    auto hit = card_row_by_code_.find(static_cast<uint32_t>(code));
    if (hit != card_row_by_code_.end()) card_row = hit->second;
  }
  const bool card_known = card_row != 0;
  if (!code) {
    out.resolution = EffectResolution::NO_CARD;
    return out;
  }
  out.passcode = code;
  out.card_row = card_row;
  out.card_known = card_known;

  /* Full desc only.  Never inspect action.effect: it is a string-table/menu
   * slot and is not the registered Lua effect index (action_semantics.py:150). */
  const int64_t desc = static_cast<int64_t>(action.desc);
  if (!desc) {
    out.resolution = EffectResolution::NO_DESCRIPTION;
    return out;
  }

  const uint64_t key = DescKey(static_cast<uint32_t>(code),
                               static_cast<uint32_t>(desc));
  auto unique = desc_unique_.find(key);
  if (unique == desc_unique_.end()) {
    std::vector<int64_t> candidates;
    auto amb = desc_ambiguous_.find(key);
    if (amb != desc_ambiguous_.end()) {
      const uint32_t off = amb->second.first;
      const uint32_t count = amb->second.second;
      for (uint32_t i = 0; i < count; ++i)
        candidates.push_back(ambiguous_index_[off + i]);
    }
    out.resolution = candidates.size() > 1
                         ? EffectResolution::AMBIGUOUS_DESCRIPTION
                         : EffectResolution::UNMAPPED_DESCRIPTION;
    if (candidates.size() > 1) out.ambiguous_candidates = std::move(candidates);
    return out;
  }

  const int32_t effect_index = unique->second;
  out.has_effect_index = true;
  out.effect_index = effect_index;
  auto row = effect_row_by_identity_.find(
      IdentityKey(static_cast<uint32_t>(code), effect_index));
  if (row == effect_row_by_identity_.end() || row->second == 0) {
    out.resolution = EffectResolution::MISSING_EFFECT_ARTIFACT;
    return out;
  }
  out.effect_row = row->second;
  out.effect_exact = true;
  out.resolution = EffectResolution::EXACT;
  return out;
}

void InstallSemantics(const std::string& path) {
  std::lock_guard<std::mutex> lock(g_default_mutex);
  if (g_default) {
    if (g_default_path != path)
      throw SemanticsError(
          "semantics already installed from " + g_default_path +
          "; two row spaces in one corpus would make the resolutions "
          "incomparable");
    return;
  }
  g_default.reset(new SemanticTable(path));
  g_default_path = path;
}

const SemanticTable* DefaultSemantics() {
  std::lock_guard<std::mutex> lock(g_default_mutex);
  return g_default.get();
}

}  // namespace mfenv
