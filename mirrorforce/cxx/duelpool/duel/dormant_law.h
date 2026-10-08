// Dormant identities (belief-world search): the run's registered load-side-effect table.
//
// A card's script may make duel-level registrations when it loads -- guarded global watchers, activity counters,
// global flags. In a normal duel they exist because some card of that code is in a deck, from that card's creation.
// A belief world whose hidden cards differ from the truth's would then inherit watchers (and their histories) from
// the true decks: the world, and anything searched in it, would depend on the truth. With a registered table the
// core creates, before any deck card, one inert object per table code (duel_create_dormant): every such registration
// is made once, at duel creation, in code order, whichever cards the decks hold, and a later replacement of hidden
// identities (duel_replace_hidden, search_api.h) finds them already made. A code outside the table that registers at
// duel level while it loads is a script error, and a table code whose load does more than watch (a rule effect, an
// effect given to another card) is refused when the duel is created.
//
// The table comes from tools/mf_runtime_dormant_table.py, which runs every code of the run's candidate universe
// in a scratch duel (dormant_scan); it is content-addressed and part of the checkpoint identity. One table per process
// (registering another is refused); none registered: duels are created as before.
#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
int32_t duel_create_dormant(intptr_t pduel, const uint32_t codes[], int32_t count, uint32_t report[]);
int32_t duel_probe_load(intptr_t pduel, uint32_t code, uint32_t *report);
int32_t duel_replace_hidden(intptr_t pduel, uint8_t playerid, const uint32_t hand[], int32_t n_hand,
                            const uint32_t deck[], int32_t n_deck, const uint32_t facedown[], int32_t n_facedown,
                            const uint32_t extra[], int32_t n_extra);
int32_t duel_arena_digest(intptr_t pduel, uint64_t *digest);
int32_t duel_hidden_blockers(intptr_t pduel, uint8_t playerid, uint32_t out[], int32_t cap);
int32_t duel_collect_garbage(intptr_t pduel);
}

namespace duelenv {
namespace dormant {

constexpr const char *kLaw = "dormant_identities/v1";
// duel_create_dormant's report bits: event watchers, activity counters, global flags, registrations the loading card
// does not own (Effect.GlobalEffect) -- a table may hold codes whose first load does these; rule effects and effects
// given to other cards (it may not)
constexpr uint32_t kWatchers = 1, kActivityCounters = 2, kGlobalFlags = 4, kRules = 8, kGranted = 16, kUnowned = 32;
constexpr uint32_t kAllowed = kWatchers | kActivityCounters | kGlobalFlags | kUnowned;

struct Registration {
  std::vector<uint32_t> codes;  // strictly ascending
  std::string sha256;           // the table file's content address
};

inline std::mutex &RegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::optional<Registration> &RegistrySlot() {
  static std::optional<Registration> slot;
  return slot;
}

inline const Registration &Register(Registration registration) {
  for (size_t i = 1; i < registration.codes.size(); ++i)
    if (registration.codes[i] <= registration.codes[i - 1])
      throw std::runtime_error("a dormant table lists its codes strictly ascending");
  if (registration.sha256.size() != 64) throw std::runtime_error("a dormant table is named by its sha256");
  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto &slot = RegistrySlot();
  if (slot) {
    if (slot->sha256 != registration.sha256 || slot->codes != registration.codes)
      throw std::runtime_error("a different dormant table is already registered in this process");
    return *slot;
  }
  slot = std::move(registration);
  return *slot;
}

inline std::optional<Registration> Registered() {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  return RegistrySlot();
}

// The table generator's scan: each code loaded twice in one scratch duel (code order), the reports of both loads --
// the second tells per-instance registrations from guarded ones.
inline std::vector<std::pair<uint32_t, uint32_t>> Scan(intptr_t pduel, const std::vector<uint32_t> &codes) {
  std::vector<std::pair<uint32_t, uint32_t>> out;
  for (uint32_t code : codes) {
    uint32_t first = 0, second = 0;
    if (duel_probe_load(pduel, code, &first) != 0 || duel_probe_load(pduel, code, &second) != 0)
      throw std::runtime_error("duel_probe_load refused " + std::to_string(code));
    out.emplace_back(first, second);
  }
  return out;
}

// The core's report for each code: what its load did at duel level (a fresh duel, before any other card).
inline std::vector<uint32_t> Create(intptr_t pduel, const std::vector<uint32_t> &codes) {
  std::vector<uint32_t> report(codes.size());
  const int32_t rc = duel_create_dormant(pduel, codes.data(), static_cast<int32_t>(codes.size()), report.data());
  if (rc != 0) throw std::runtime_error("duel_create_dormant returned " + std::to_string(rc));
  return report;
}

// Called right after a duel is created, before its decks: the registered table's dormant identities, if any.
inline void CreateIn(intptr_t pduel) {
  const auto law = Registered();
  if (!law) return;
  const auto report = Create(pduel, law->codes);
  for (size_t i = 0; i < report.size(); ++i)
    if (report[i] & ~kAllowed)
      throw std::runtime_error("dormant table code " + std::to_string(law->codes[i]) + " does more than watch when it "
                               "loads (report " + std::to_string(report[i]) + "): the table is invalid");
}

}  // namespace dormant
}  // namespace duelenv
