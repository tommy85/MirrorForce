// The announce-card candidate law (design section 6.1): one pure function of public state, called by the
// training env, the evaluation path and the deployment client. Its reference implementation and test oracle is
// mirrorforce/tests/announce_law_oracle.py; the two must agree byte for byte.
//
// Order: five tiers -- the filter's ISCODE literals, cards publicly seen this duel (the opponent's, then the
// declarer's own), the declarer's own recipe, generic staples, the opponent-deck belief -- each by score descending
// then card id ascending; every candidate passes the core's declarable filter and keeps only its first position.
// Tiers 1-4 are never truncated (more than the cap is a configuration error and throws); tier 5 is cut at the cap and
// counted. With no candidate at all, the named branch "empty_union" takes every declarable card in card id order up
// to the cap and counts the rest.
//
// Belief "public_posterior_presence/v1": the library recipes of the room format (all when unknown) that contain every
// card the opponent has publicly shown, by presence; each cluster with such a recipe weighs 1/(clusters), split evenly
// over its compatible recipes; a card's score is the summed weight of the compatible recipes holding it, summed as
// doubles in library order, recipe by recipe and card by card (main, then extra), so the oracle matches bit for bit.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mfenv/sha256.h"

namespace duelenv {
namespace announce {

// v2 (2026-10-01): the same order and tiers as v1; registered with a cap of at least 192 (v1 ran at 128, which tiers
// 1-4 exceed in reachable states: up to 138 on MD80 and 134 on the friend pool with an any-card filter).
constexpr const char *kLaw = "announce_public_candidates/v2";
constexpr int64_t kMinCap = 192;
constexpr const char *kBeliefLaw = "public_posterior_presence/v1";
constexpr const char *kEmptyUnion = "empty_union";
constexpr uint32_t kOpcodeIsCode = 0x40000100u;

struct Recipe {
  std::vector<uint32_t> main;
  std::vector<uint32_t> extra;
  std::string cluster;
  std::string format;
};

// The registered tables a law instance uses: the public library and the staple table (code -> deck-type count).
struct Tables {
  std::vector<Recipe> library;
  std::unordered_map<uint32_t, int64_t> staples;
};

struct Result {
  std::vector<uint32_t> candidates;
  std::array<int, 5> tiers{};  // new cards per tier: literal, seen, own_recipe, staple, belief
  int truncated_belief = 0;
  int truncated_empty_union = 0;
  bool empty_union = false;
};

// ``card_id(code)`` is the env's card table order (0 for a code outside the table); ``declarable(code)`` is the core's
// filter for this prompt's program (false for a code outside the card table).
using CardId = std::function<int64_t(uint32_t)>;
using Declarable = std::function<bool(uint32_t)>;

inline std::vector<uint32_t> Literals(const std::vector<uint32_t> &opcodes) {
  std::vector<uint32_t> out;
  for (size_t i = 1; i < opcodes.size(); ++i)
    if (opcodes[i] == kOpcodeIsCode) out.push_back(opcodes[i - 1]);
  return out;
}

// card code -> posterior weight (insertion order irrelevant; scores are summed in library order).
inline std::unordered_map<uint32_t, double> BeliefScores(const std::vector<Recipe> &library,
                                                         const std::vector<uint32_t> &seen_opponent,
                                                         const std::optional<std::string> &room_format) {
  const std::set<uint32_t> shown(seen_opponent.begin(), seen_opponent.end());
  std::vector<const Recipe *> compatible;
  for (const Recipe &recipe : library) {
    if (room_format && recipe.format != *room_format) continue;
    std::unordered_set<uint32_t> held(recipe.main.begin(), recipe.main.end());
    held.insert(recipe.extra.begin(), recipe.extra.end());
    bool ok = true;
    for (uint32_t code : shown)
      if (!held.count(code)) { ok = false; break; }
    if (ok) compatible.push_back(&recipe);
  }
  std::map<std::string, int> cluster_sizes;
  for (const Recipe *recipe : compatible) ++cluster_sizes[recipe->cluster];
  std::unordered_map<uint32_t, double> scores;
  for (const Recipe *recipe : compatible) {
    const double weight = (1.0 / static_cast<double>(cluster_sizes.size())) /
                          static_cast<double>(cluster_sizes[recipe->cluster]);
    std::unordered_set<uint32_t> counted;
    for (const std::vector<uint32_t> *part : {&recipe->main, &recipe->extra})
      for (uint32_t code : *part)
        if (counted.insert(code).second) scores[code] += weight;
  }
  return scores;
}

inline Result Candidates(const std::vector<uint32_t> &opcodes, const std::vector<uint32_t> &seen_opponent,
                         const std::vector<uint32_t> &seen_own, const std::vector<uint32_t> &own_main,
                         const std::vector<uint32_t> &own_extra, const Tables &tables,
                         const std::optional<std::string> &room_format, int64_t cap,
                         const std::vector<uint32_t> &card_table, const CardId &card_id,
                         const Declarable &declarable) {
  if (cap < 1) throw std::runtime_error("the announce cap is a positive integer");
  auto eligible = [&](const std::vector<uint32_t> &codes) {
    std::set<uint32_t> out;
    for (uint32_t code : codes)
      if (card_id(code) > 0 && declarable(code)) out.insert(code);
    return std::vector<uint32_t>(out.begin(), out.end());
  };
  auto by_id = [&](std::vector<uint32_t> codes) {
    std::sort(codes.begin(), codes.end(), [&](uint32_t a, uint32_t b) { return card_id(a) < card_id(b); });
    return codes;
  };
  auto by_score = [&](std::vector<uint32_t> codes, const std::function<double(uint32_t)> &score) {
    std::sort(codes.begin(), codes.end(), [&](uint32_t a, uint32_t b) {
      const double sa = score(a), sb = score(b);
      if (sa != sb) return sa > sb;
      return card_id(a) < card_id(b);
    });
    return codes;
  };

  const auto beliefs = BeliefScores(tables.library, seen_opponent, room_format);
  std::vector<uint32_t> staple_codes, belief_codes;
  for (const auto &[code, count] : tables.staples) staple_codes.push_back(code);
  for (const auto &[code, score] : beliefs) belief_codes.push_back(code);

  std::vector<std::vector<uint32_t>> tiers(5);
  tiers[0] = by_id(eligible(Literals(opcodes)));
  tiers[1] = by_id(eligible(seen_opponent));
  {
    const std::set<uint32_t> opponent(seen_opponent.begin(), seen_opponent.end());
    std::vector<uint32_t> own;
    for (uint32_t code : eligible(seen_own))
      if (!opponent.count(code)) own.push_back(code);
    for (uint32_t code : by_id(own)) tiers[1].push_back(code);
  }
  {
    std::vector<uint32_t> recipe(own_main);
    recipe.insert(recipe.end(), own_extra.begin(), own_extra.end());
    tiers[2] = by_id(eligible(recipe));
  }
  tiers[3] = by_score(eligible(staple_codes),
                      [&](uint32_t code) { return static_cast<double>(tables.staples.at(code)); });
  tiers[4] = by_score(eligible(belief_codes), [&](uint32_t code) { return beliefs.at(code); });

  Result result;
  std::unordered_set<uint32_t> placed;
  std::vector<std::vector<uint32_t>> added(5);
  for (size_t t = 0; t < 5; ++t) {
    for (uint32_t code : tiers[t])
      if (placed.insert(code).second) added[t].push_back(code);
    result.tiers[t] = static_cast<int>(added[t].size());
  }
  int64_t fixed = 0;
  for (size_t t = 0; t < 4; ++t) fixed += static_cast<int64_t>(added[t].size());
  if (fixed > cap)
    throw std::runtime_error("announce tiers 1-4 hold " + std::to_string(fixed) + " cards, more than the cap " +
                             std::to_string(cap));
  for (size_t t = 0; t < 4; ++t)
    result.candidates.insert(result.candidates.end(), added[t].begin(), added[t].end());
  const int64_t room = cap - static_cast<int64_t>(result.candidates.size());
  const int64_t taken = std::min<int64_t>(room, static_cast<int64_t>(added[4].size()));
  result.candidates.insert(result.candidates.end(), added[4].begin(), added[4].begin() + taken);
  result.truncated_belief = static_cast<int>(static_cast<int64_t>(added[4].size()) - taken);
  if (result.candidates.empty()) {
    result.empty_union = true;
    const auto every = by_id(eligible(card_table));
    const int64_t keep = std::min<int64_t>(cap, static_cast<int64_t>(every.size()));
    result.candidates.assign(every.begin(), every.begin() + keep);
    result.truncated_empty_union = static_cast<int>(static_cast<int64_t>(every.size()) - keep);
  }
  return result;
}

// The belief candidates (S1, obs:candidates_): the same tiers with no filter (every card of the table is declarable),
// each candidate with its tier (2 seen, 3 own recipe, 4 staple, 5 posterior; no literals without a filter).
inline std::vector<std::pair<uint32_t, int>> BeliefCandidates(const std::vector<uint32_t> &seen_opponent,
                                                              const std::vector<uint32_t> &seen_own,
                                                              const std::vector<uint32_t> &own_main,
                                                              const std::vector<uint32_t> &own_extra,
                                                              const Tables &tables,
                                                              const std::optional<std::string> &room_format,
                                                              int64_t cap, const std::vector<uint32_t> &card_table,
                                                              const CardId &card_id) {
  const Result result = Candidates({}, seen_opponent, seen_own, own_main, own_extra, tables, room_format, cap,
                                   card_table, card_id, [](uint32_t) { return true; });
  if (result.empty_union) throw std::runtime_error("belief candidates: no candidate (an empty own recipe?)");
  std::vector<std::pair<uint32_t, int>> out;
  size_t at = 0;
  for (int tier = 0; tier < 5; ++tier) {
    const int count = tier < 4 ? result.tiers[tier] : result.tiers[4] - result.truncated_belief;
    for (int k = 0; k < count; ++k) out.emplace_back(result.candidates.at(at++), tier + 1);
  }
  if (at != result.candidates.size()) throw std::runtime_error("belief candidates: tiers do not cover the list");
  return out;
}

// ---- the registered law of a process (one per run: the training env, evaluation and the client register the same) ----

struct Registration {
  Tables tables;
  int64_t cap = 0;
  std::optional<std::string> room_format;
  double staple_threshold = 0;
  std::string tables_sha256;  // over Encode(tables, staple_threshold)
};

// The canonical text the tables digest covers: the library in order, then the staples by code.
inline std::string Encode(const Tables &tables, double staple_threshold) {
  char threshold[32];
  std::snprintf(threshold, sizeof(threshold), "%.17g", staple_threshold);
  std::string out = std::string(kLaw) + "\n" + kBeliefLaw + "\nthreshold " + threshold + "\n";
  for (const Recipe &recipe : tables.library) {
    out += "recipe\t" + recipe.cluster + "\t" + recipe.format + "\tmain";
    for (uint32_t code : recipe.main) out += " " + std::to_string(code);
    out += "\textra";
    for (uint32_t code : recipe.extra) out += " " + std::to_string(code);
    out += "\n";
  }
  std::vector<std::pair<uint32_t, int64_t>> staples(tables.staples.begin(), tables.staples.end());
  std::sort(staples.begin(), staples.end());
  for (const auto &[code, count] : staples) out += "staple " + std::to_string(code) + " " + std::to_string(count) + "\n";
  return out;
}

inline std::mutex &RegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::optional<Registration> &RegistrySlot() {
  static std::optional<Registration> slot;
  return slot;
}

// Registers the process's law; registering a different one afterwards is refused (one law per run).
inline const Registration &Register(Registration registration) {
  if (registration.cap < kMinCap)
    throw std::runtime_error("announce_public_candidates/v2 needs a cap of at least " + std::to_string(kMinCap));
  if (registration.tables.library.empty() && registration.tables.staples.empty())
    throw std::runtime_error("an announce law needs its library or staple table");
  registration.tables_sha256 = mfenv::Sha256Hex(Encode(registration.tables, registration.staple_threshold));
  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto &slot = RegistrySlot();
  if (slot) {
    if (slot->tables_sha256 != registration.tables_sha256 || slot->cap != registration.cap ||
        slot->room_format != registration.room_format)
      throw std::runtime_error("a different announce law is already registered in this process");
    return *slot;
  }
  slot = std::move(registration);
  return *slot;
}

inline const Registration &Registered() {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  const auto &slot = RegistrySlot();
  if (!slot) throw std::runtime_error("no announce law is registered (register_announce_law before an announce prompt)");
  return *slot;
}

}  // namespace announce
}  // namespace duelenv
