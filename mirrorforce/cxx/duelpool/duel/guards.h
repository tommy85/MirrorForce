// No-progress guards of the env (design item 4), ported from (the design notes;
// mirrorforce/cardrules/activation_guard.py, mirrorforce/common/cycle_guard.py). They read nothing but the acting player's own
// observation, prompt and choices, and they withhold menu rows; they never end a game and never empty a menu.
//
// own_view_no_progress_exclusion/v1, two scopes that never meet on one prompt:
// - activation: MSG_SELECT_CHAIN while the public chain stack is empty, and the idle and battle command menus; in
//   one (turn, phase) window of the acting player, if choosing "activate X" at a guarded prompt brought the player
//   back to a guarded prompt with the same own-view key, X is withheld at that key for the rest of the window. Only
//   ACTIVATE rows are ever withheld.
// - resolution: every multi-option prompt while a chain link resolves (the duel's ordinal of the current
//   MSG_CHAIN_SOLVING until its MSG_CHAIN_SOLVED or MSG_CHAIN_END); in the window of the acting player and that
//   resolution, the row last chosen at each key is withheld when the player meets the key again. Any row.
// no_progress_command_cycle_exclusion/v2: at the idle and battle command menus, every command already made at the
// same key since the player's last public fact is withheld (an attack whose target prompt was cancelled, and so on).
//
// The own-view key (built by the env, OwnViewKey) is a SHA-256 over the prompt message and the acting player's
// observation with its full menu, except the history arrays and the activation tallies, which change on every lap of
// a loop; tallies are reduced to their support (which effects were activated, whether a count is nonzero).
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "mfenv/sha256.h"

namespace duelenv {
namespace guards {

constexpr const char *kNoProgressLaw = "own_view_no_progress_exclusion/v1";
constexpr const char *kCommandCycleLaw = "no_progress_command_cycle_exclusion/v2";
constexpr int kMsgSelectBattleCmd = 10, kMsgSelectIdleCmd = 11, kMsgSelectChain = 16;
constexpr int kMsgChaining = 70, kMsgChainSolving = 72, kMsgChainSolved = 73, kMsgChainEnd = 74;

// The public chain coordinates every player computes from the message stream.
struct ChainState {
  int links = 0;           // MSG_CHAINING pushes, MSG_CHAIN_END clears
  int64_t solving = 0;     // MSG_CHAIN_SOLVING messages so far
  int64_t resolution = 0;  // ordinal of the resolving link, 0 outside a resolution

  void Observe(int msg) {
    if (msg == kMsgChaining) ++links;
    if (msg == kMsgChainSolving) resolution = ++solving;
    if (msg == kMsgChainSolved) resolution = 0;
    if (msg == kMsgChainEnd) links = 0, resolution = 0;
  }
};

// One multi-option prompt as the guards see it.
struct Prompt {
  int player = 0, msg = 0, turn = 0, phase = 0;
  ChainState chain;
  int64_t facts = 0;           // the player's public facts so far
  std::string key;             // own-view key
  std::vector<bool> activate;  // per menu row: an ACTIVATE row
};

struct Exclusion {
  std::set<int> no_progress, cycle;
  bool kept_menu = false;  // both together would have emptied the menu: nothing withheld
  std::set<int> rows() const {
    std::set<int> out(no_progress);
    out.insert(cycle.begin(), cycle.end());
    return out;
  }
};

inline bool Command(int msg) { return msg == kMsgSelectIdleCmd || msg == kMsgSelectBattleCmd; }

// "resolution", "activation" or "" for a prompt (own_view_no_progress_exclusion/v1).
inline const char *Scope(const Prompt &p) {
  if (p.chain.resolution > 0) return "resolution";
  if ((p.msg == kMsgSelectChain && p.chain.links == 0) || Command(p.msg)) return "activation";
  return "";
}

inline bool Guarded(const Prompt &p) { return *Scope(p) != '\0' || Command(p.msg); }

class Guards {
 public:
  Exclusion Exclusions(const Prompt &p) const {
    Exclusion out;
    const std::string scope = Scope(p);
    const size_t rows = p.activate.size();
    if (scope == "activation") {
      const Activation &a = activation_[p.player];
      if (a.turn == p.turn && a.phase == p.phase) {
        std::set<int> held;
        if (const auto hit = a.withheld.find(p.key); hit != a.withheld.end()) held = hit->second;
        if (a.has_last && a.last_key == p.key && a.last_activate) held.insert(a.last_row);
        for (int row : held)
          if (row >= 0 && static_cast<size_t>(row) < rows && p.activate[row]) out.no_progress.insert(row);
      }
    } else if (scope == "resolution") {
      const Resolution &r = resolution_[p.player];
      if (r.window == p.chain.resolution) {
        std::set<int> held;
        if (const auto hit = r.withheld.find(p.key); hit != r.withheld.end()) held = hit->second;
        if (const auto last = r.last.find(p.key); last != r.last.end()) held.insert(last->second);
        for (int row : held)
          if (row >= 0 && static_cast<size_t>(row) < rows) out.no_progress.insert(row);
      }
    }
    if (Command(p.msg)) {
      const Cycle &c = cycle_[p.player];
      if (c.turn == p.turn && c.facts == p.facts)
        if (const auto hit = c.tried.find(p.key); hit != c.tried.end())
          for (int row : hit->second)
            if (row >= 0 && static_cast<size_t>(row) < rows) out.cycle.insert(row);
    }
    if (out.rows().size() >= rows && rows > 0) {
      out = Exclusion{};
      out.kept_menu = true;
    }
    return out;
  }

  // The player chose ``row`` (of the full menu) at a guarded prompt.
  void Record(const Prompt &p, int row) {
    const std::string scope = Scope(p);
    if (scope == "activation") {
      Activation &a = activation_[p.player];
      if (a.turn != p.turn || a.phase != p.phase) a = Activation{p.turn, p.phase};
      if (a.has_last && a.last_key == p.key && a.last_activate) a.withheld[p.key].insert(a.last_row);
      a.has_last = true;
      a.last_key = p.key;
      a.last_row = row;
      a.last_activate = row >= 0 && static_cast<size_t>(row) < p.activate.size() && p.activate[row];
    } else if (scope == "resolution") {
      Resolution &r = resolution_[p.player];
      if (r.window != p.chain.resolution) r = Resolution{p.chain.resolution};
      if (const auto last = r.last.find(p.key); last != r.last.end()) r.withheld[p.key].insert(last->second);
      r.last[p.key] = row;
    }
    if (Command(p.msg)) {
      Cycle &c = cycle_[p.player];
      if (c.turn != p.turn || c.facts != p.facts) c = Cycle{p.turn, p.facts};
      c.tried[p.key].insert(row);
    }
  }

 private:
  struct Activation {
    int turn = -1, phase = -1;
    bool has_last = false;
    std::string last_key;
    int last_row = -1;
    bool last_activate = false;
    std::map<std::string, std::set<int>> withheld;
  };
  struct Resolution {
    int64_t window = 0;
    std::map<std::string, int> last;
    std::map<std::string, std::set<int>> withheld;
  };
  struct Cycle {
    int turn = -1;
    int64_t facts = -1;
    std::map<std::string, std::set<int>> tried;
  };
  std::array<Activation, 2> activation_;
  std::array<Resolution, 2> resolution_;
  std::array<Cycle, 2> cycle_;
};

// The own-view key's canonical bytes: each named part as name, a NUL, its length (8 bytes, little endian) and bytes.
class KeyBuilder {
 public:
  explicit KeyBuilder(int msg) : text_(std::string(kNoProgressLaw) + "\n" + std::to_string(msg) + "\n") {}
  void Add(const char *name, const uint8_t *data, size_t size) {
    text_ += name;
    text_ += '\0';
    for (int i = 0; i < 8; ++i) text_ += static_cast<char>((static_cast<uint64_t>(size) >> (8 * i)) & 0xFF);
    text_.append(reinterpret_cast<const char *>(data), size);
  }
  void Add(const char *name, const std::vector<uint8_t> &bytes) { Add(name, bytes.data(), bytes.size()); }
  std::string Finish() const { return mfenv::Sha256Hex(text_); }

 private:
  std::string text_;
};

}  // namespace guards
}  // namespace duelenv
