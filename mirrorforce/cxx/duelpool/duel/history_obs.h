// Public history observation for the env (design section 7.2, H (A) and (B)).
//
// The env hands every core message buffer to ``History::ConsumeBuffer`` and every multi-option decision to
// ``History::ConsumeOwnChoice``. Facts come from mfenv's per-viewer interval records (mirrorforce/cxx/mfenv/history.cc:
// each viewer's audience-split facts, the same records an in-process host's clients reproduce), so a viewer never
// sees what its client would not receive. mfenv's public tracker follows stable public entities (the row links) and
// the turn ledger; this header adds the condensed current-turn window, the chain table, the activation table with
// outcomes and the extended ledger. ``Write`` fills the observation arrays for one observer against its card rows.
//
// Condensed window: phase boundaries are not rows (every row carries the phase in force); each chain link is one
// row, its activation, whose state is updated when the link resolves, is negated or disabled; own decisions are rows
// built from the chosen legal action. Overflow drops the oldest rows that are not activations, and counts them.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "duel/guards.h"
#include "duel/public_status.h"
#include "duel/unpositioned.h"
#include "mfenv/history.h"
#include "mfenv/messages.h"
#include "mfenv/public_tracker.h"
#include "mfenv/semantics.h"

namespace duelenv {
namespace history {

// The current turn's window holds up to kEvents rows; rows before it leave in chunks of kChunkRows (at most
// kChunkCap per turn), delivered to each observer kChunkSlots per decision (Observer::Write's Delivery). The cap sizes a
// turn of 512 decisions of both players at the measured rows per decision (p99.9 over turns of 10,000 random games
// per pool on 2026-10-01: MD80 6.0, friend pool 7.0): 512 x 7 = 3,584 rows = the 512-row window and 48 chunks.
constexpr int kEvents = 512, kEventWidth = 27, kRefWidth = 4;
constexpr int kChunkRows = 64, kChunkSlots = 4, kChunkCap = 48, kChunkMetaWidth = 4;

// The completed turns one game can have, at most: the bound on the closed turns awaiting delivery to one observer
// (each is kept until its window is delivered, two per decision of that observer). A player's turns, but the first
// player's first, begin with a draw (master rule 5, draw count at least 1), so a player has at most (main deck + 1)
// turns plus one per card returned to its deck; a card returns to a deck only through an effect resolution chosen at
// a decision (public_effects/v1's reviewed pool has no forced return and no skipped draw; a pool outside it must be
// allowed explicitly), and one decision returns at most every card of the game. Under the env's step limit
// (max_steps decisions) a game completes at most 2 x (60 + 1) + 2 x (60 + 15) x max_steps turns, so the cap below is
// never reached in a legal game; exceeding it is an env error.
inline int64_t CompletedTurnBound(int64_t max_steps) {
  constexpr int64_t kMainDeck = 60, kGameCards = 2 * (60 + 15);
  return 2 * (kMainDeck + 1) + kGameCards * std::max<int64_t>(0, max_steps);
}

// The window law of a run: the constants above unless an env config names others (tests use small ones to reach the
// chunk paths in ordinary games; the obs shapes follow the law).
struct WindowLaw {
  int window = kEvents, chunk = kChunkRows, cap = kChunkCap, slots = kChunkSlots;
};
constexpr int kChainRows = 8, kChainWidth = 8;
constexpr int kActivationRows = 64, kActivationWidth = 12;
constexpr int kLedgerWidth = 16;
constexpr int kHintRows = 16, kHintWidth = 5;
constexpr int kCardTurnWidth = 8;
constexpr int kExtraLedger = 8;  // ledger columns this header counts (6..13)
constexpr int kUnpositionedRows = 32, kUnpositionedWidth = 4;
constexpr int kClosedTurns = 2, kClosedMetaWidth = 5;  // obs:closed_turns_ windows (completed turns, each once)

// Chain link states (the value column of an activation row and the state column of chain_).
enum LinkState : int { kPending = 0, kResolving = 1, kResolved = 2, kNegated = 3, kDisabled = 4 };

// The engine core's common.h values, spelled out so this header does not depend on macro order.
constexpr int kLocDeck = 0x01, kLocHand = 0x02, kLocMzone = 0x04, kLocSzone = 0x08, kLocGrave = 0x10,
              kLocRemoved = 0x20, kLocExtra = 0x40, kLocOverlay = 0x80;
constexpr int kPhaseDraw = 0x01;
constexpr int kMsgSwap = 55, kMsgShuffleSetCard = 36;
constexpr int kDescriptionLimit = 10000;

inline uint8_t Clip(int64_t value, int64_t high = 255) {
  return static_cast<uint8_t>(value < 0 ? 0 : (value > high ? high : value));
}

inline uint8_t LocationId(int location) {
  switch (location & 0x7F) {
    case kLocDeck: return 1;
    case kLocHand: return 2;
    case kLocMzone: return 3;
    case kLocSzone: return 4;
    case kLocGrave: return 5;
    case kLocRemoved: return 6;
    case kLocExtra: return 7;
    default: return 0;
  }
}

// PHASE_* bit -> 1..10 (0 none).
inline uint8_t PhaseId(int phase) {
  for (int bit = 0; bit < 10; ++bit)
    if (phase == (1 << bit)) return static_cast<uint8_t>(bit + 1);
  return 0;
}

// A card description's string index (0-15), or 15 for a system description.
inline int DescIndex(int64_t desc) { return desc >= kDescriptionLimit ? static_cast<int>(desc & 0xF) : 15; }

// common/card_exact.py's artwork rule: a nonzero alias within 20 of the code is the variant's base card, except Black
// Luster Soldier - Envoy of the Evening Twilight's rule-code alias.
constexpr uint32_t kArtworkOffset = 20, kBlackLusterSoldier2 = 5405695;
inline bool IsArtworkVariant(uint32_t code, uint32_t alias) {
  return alias != 0 && code != kBlackLusterSoldier2 && alias < code + kArtworkOffset && code < alias + kArtworkOffset;
}

// Artwork variants' base codes, registered with the card table (init_module).
inline std::unordered_map<uint32_t, uint32_t> &ArtworkBases() {
  static std::unordered_map<uint32_t, uint32_t> bases;
  return bases;
}
// A card code with an artwork variant mapped to its base card.
inline uint32_t ArtworkBase(uint32_t code) {
  const auto hit = ArtworkBases().find(code);
  return hit == ArtworkBases().end() ? code : hit->second;
}

// The string index of a description as one of ``code``'s own effects: its ordinal when the description is the card's
// (its code's, or its artwork base's), else 15 as for a system description -- a description of another card names
// that card's string, not one of this card's effects (the effect-unit evidence is per card row and ordinal).
inline int OwnDescIndex(int64_t code, int64_t desc) {
  if (desc < kDescriptionLimit) return 15;
  const int64_t owner = desc >> 4;
  if (owner == code) return static_cast<int>(desc & 0xF);
  const auto base = ArtworkBases().find(static_cast<uint32_t>(code));
  return base != ArtworkBases().end() && owner == base->second ? static_cast<int>(desc & 0xF) : 15;
}

// The decision an observer made at a multi-option prompt, from the env's chosen legal action.
struct OwnChoice {
  int msg = 0;
  int act = 0;
  int64_t card_row = 0;  // the env's card id (0 when the action names no card)
  int effect = -1;
  bool has_place = false;
  int side = 0;
  int location = 0;
  int sequence = 0;
  int overlay_index = 0;
  bool overlay = false;
};

struct Entry {
  mfenv::HistoryTokenRecord record;
  mfenv::TrackerRowAnnotation note;
  bool own = false;
  int msg = 0, act = 0, effect = -1;
  int phase = 0;
  int state = kPending;  // chain activations only
  bool activation = false;
};

struct ChainLink {
  int link = 0, side = 0, desc_index = 0, location = 0, sequence = 0, state = kPending;
  int64_t card_row = 0;
};

struct Activation {
  std::array<int64_t, 2> turn{}, resolved{}, duel{};
  int64_t negated = 0, last_turn = 0, order = 0;
};

using ActivationKey = std::pair<int64_t, int>;  // (card row, description index)

// Whose card a fact shows, relative to its viewer: the side of the owner's zone it enters or leaves (deck, hand,
// graveyard, banished, extra deck), else the controller it moves to or from, else the acting player; -1 if none.
inline int SeenSide(const mfenv::HistoryTokenRecord &r) {
  auto owner_zone = [](int location) {
    const int l = location & 0xFF;
    return l == kLocDeck || l == kLocHand || l == kLocGrave || l == kLocRemoved || l == kLocExtra;
  };
  auto side = [](int relative) { return relative == 0 || relative == 1; };
  if (r.to_location && owner_zone(r.to_location) && side(r.to_controller_relative)) return r.to_controller_relative;
  if (r.from_location && owner_zone(r.from_location) && side(r.from_controller_relative))
    return r.from_controller_relative;
  if (r.to_location && side(r.to_controller_relative)) return r.to_controller_relative;
  if (r.from_location && side(r.from_controller_relative)) return r.from_controller_relative;
  return side(r.player_relative) ? r.player_relative : -1;
}

// A completed turn as it stood when it ended (activation outcomes settled, every event after the viewer's last decision
// in it included): its rows, how many left in chunks, and how much of it this viewer has been delivered.
struct ClosedTurn {
  int turn = 0;
  int turn_player = -1;        // relative to the viewer: 0 its own turn, 1 the opponent's
  std::vector<Entry> entries;
  int chunks = 0;              // rows [0, 64 chunks) left in chunks; the closed window is the rest (at most kEvents)
  int chunks_delivered = 0;
  bool window_delivered = false;
};

// One decision's deliveries to an observer (Observer::staged): chunks and closed windows exported at that decision,
// committed when the observer's next decision is written (or its turn ends), so writing one decision's observation
// twice exports the same.
struct Delivery {
  int64_t decision = -1;
  std::vector<int> closed_chunks;    // per pending turn: its chunks delivered at this decision
  std::vector<char> closed_window;   // per pending turn: its closed window delivered at this decision
  int current_chunks = 0;            // the current turn's chunks delivered at this decision
};

class Observer {
 public:
  mfenv::PublicTracker tracker;
  HandUnpositioned hand;       // the opponent's identities this viewer knows without their positions: hand,
  OpponentZones zones;         // deck and face-down extra deck (unpositioned.h)
  public_effects::Tracker status;  // lingering public effects: card statuses, equips, player effects (public_status.h)
  std::vector<Entry> entries;  // the current turn
  std::vector<ChainLink> chain;
  std::map<int, size_t> open_rows;           // chain link -> its activation row in ``entries``
  std::map<int, ActivationKey> link_keys;    // chain link -> its activation key
  std::map<int, int> link_sides;             // chain link -> the activating side
  std::set<int> root_links;                 // live links seeded without a past event row
  std::map<ActivationKey, Activation> activations;
  std::array<std::array<int64_t, kExtraLedger>, 2> ledger{};
  int phase = 0;
  int turn = 0;
  int turn_player = -1;  // the current turn's player relative to the viewer (0 own, 1 opponent)
  int64_t order = 0;
  std::array<std::set<uint32_t>, 2> seen;  // card codes this viewer's facts showed this duel: own, opponent's
  int64_t facts = 0;                        // public facts this viewer has received this duel
  std::deque<ClosedTurn> pending;           // completed turns not yet fully delivered (oldest first)
  int chunks = 0;                           // the current turn's rows [0, 64 chunks) have left the window
  int chunks_delivered = 0;                 // of those chunks, delivered to this viewer
  Delivery staged;                          // the last written decision's deliveries, not yet committed
  std::deque<std::array<int64_t, 2>> recent;  // measurement: the last completed turns (turn, rows), newest first
  WindowLaw law;
  int64_t pending_cap = CompletedTurnBound(0);  // the closed turns it may await (History::SetPendingCap)
  bool passive = false;  // client mode: the other seat's observer, which writes no observation

  // Appends a row of the current turn; a row that makes the window exceed kEvents emits the window's first
  // kChunkRows rows as the turn's next chunk. A turn needing more than kChunkCap chunks is an env error.
  void Append(Entry &&e) {
    entries.push_back(std::move(e));
    while (static_cast<int64_t>(entries.size()) - static_cast<int64_t>(law.chunk) * chunks > law.window) {
      if (++chunks > law.cap)
        throw std::runtime_error("a turn of more than " + std::to_string(law.window + law.chunk * law.cap) +
                                 " condensed history rows (" + std::to_string(law.cap) + " chunks): beyond the window's"
                                 " capacity");
    }
  }

  // Applies the staged deliveries; drops completed turns delivered in full.
  void Commit() {
    for (size_t i = 0; i < staged.closed_chunks.size() && i < pending.size(); ++i) {
      pending[i].chunks_delivered += staged.closed_chunks[i];
      pending[i].window_delivered = pending[i].window_delivered || staged.closed_window[i];
    }
    chunks_delivered += staged.current_chunks;
    staged = Delivery{};
    while (!pending.empty() && pending.front().window_delivered) pending.pop_front();
  }

  void NewTurn(int new_turn, int player_relative) {
    if (turn > 0) {
      Commit();  // the deliveries of the turn's last decision belong to it
      // a passive observer (client mode: the seat whose stream this is not) never decides, so nothing is delivered
      if (!passive) pending.push_back(ClosedTurn{turn, turn_player, entries, chunks, chunks_delivered, false});
      if (static_cast<int64_t>(pending.size()) > pending_cap)
        throw std::runtime_error("more than " + std::to_string(pending_cap) + " completed turns await delivery to "
                                 "one observer (beyond the completed-turn bound of a game)");
      recent.push_front({turn, static_cast<int64_t>(entries.size())});
      if (recent.size() > static_cast<size_t>(kClosedTurns)) recent.pop_back();
    }
    entries.clear();
    chunks = 0;
    chunks_delivered = 0;
    chain.clear();
    open_rows.clear();
    link_keys.clear();
    link_sides.clear();
    root_links.clear();
    for (auto &entry : activations) {
      entry.second.turn = {0, 0};
      entry.second.resolved = {0, 0};
      entry.second.negated = 0;
    }
    ledger = {};
    phase = 0;
    turn = new_turn;
    turn_player = player_relative;
  }

  void Resolve(int link, int state) {
    const auto row = open_rows.find(link);
    // A current-root observer has the live chain, but no invented activation event in its past window.
    if (row == open_rows.end() && !root_links.count(link)) return;
    if (row != open_rows.end()) entries[row->second].state = state;
    for (ChainLink &l : chain)
      if (l.link == link) l.state = state;
    if (state == kResolving) return;
    if (const auto key = link_keys.find(link); key != link_keys.end()) {
      Activation &a = activations[key->second];
      if (state == kResolved) ++a.resolved[std::min(1, std::max(0, link_sides[link]))];
      else ++a.negated;
    }
    open_rows.erase(link);
    root_links.erase(link);
    link_keys.erase(link);
    link_sides.erase(link);
    if (link <= 1) chain.clear();  // the last link to resolve closes the chain
  }

  void Fact(const mfenv::HistoryTokenRecord &r, const mfenv::TrackerRowAnnotation &note) {
    using mfenv::HistorySubtype;
    const auto subtype = static_cast<HistorySubtype>(r.subtype);
    if (r.card_code > 0) {
      const int owner = SeenSide(r);
      if (owner >= 0) seen[owner].insert(static_cast<uint32_t>(r.card_code));
    }
    if (subtype == HistorySubtype::NEW_TURN) NewTurn(r.turn, r.player_relative);
    if (subtype == HistorySubtype::NEW_PHASE) {
      phase = r.phase;
      return;
    }
    if (subtype == HistorySubtype::CHAIN && r.value != mfenv::CHAIN_CHAINING) {
      const int state = r.value == mfenv::CHAIN_SOLVING ? kResolving
                        : r.value == mfenv::CHAIN_SOLVED ? kResolved
                        : r.value == mfenv::CHAIN_NEGATED ? kNegated : kDisabled;
      Resolve(r.link, state);
      return;
    }
    Entry e;
    e.record = r;
    e.note = note;
    e.phase = phase;
    const int side = r.player_relative == 1 ? 1 : 0;
    if (subtype == HistorySubtype::CHAIN) {
      e.activation = true;
      const ActivationKey key{r.card_row, OwnDescIndex(r.card_code, r.detail)};
      Activation &a = activations[key];
      ++a.turn[side];
      ++a.duel[side];
      a.last_turn = turn;
      a.order = ++order;
      open_rows[r.link] = entries.size();
      link_keys[r.link] = key;
      link_sides[r.link] = side;
      if (r.link == 1) ++ledger[side][7];
      ChainLink link;
      link.link = r.link;
      link.side = side;
      link.card_row = r.card_row;
      link.desc_index = OwnDescIndex(r.card_code, r.detail);
      link.location = r.from_location;
      link.sequence = r.from_sequence;
      chain.push_back(link);
    } else if (subtype == HistorySubtype::DRAW) {
      ledger[side][(phase == kPhaseDraw && r.link == 0) ? 0 : 1] += r.count;
    } else if (subtype == HistorySubtype::MOVE) {
      const int to = r.to_controller_relative == 1 ? 1 : 0;
      const int from_loc = r.from_location & 0x7F, to_loc = r.to_location & 0x7F;
      if (from_loc == kLocDeck && to_loc == kLocHand) ++ledger[to][2];
      if (to_loc == kLocGrave) ++ledger[to][3];
      if (to_loc == kLocRemoved) ++ledger[to][4];
    } else if (subtype == HistorySubtype::LP) {
      if (r.value == mfenv::LP_PAY_COST) ledger[side][5] += r.amount;
      if (r.value == mfenv::LP_DAMAGE) ledger[side][6] += r.amount;
    }
    Append(std::move(e));
  }
};

struct CurrentRootSeed {
  mfenv::TrackerSeed tracker;  // relative to the initialized viewer
  Counts hand, deck, extra;
  std::vector<int> hand_group;
  std::map<int, int64_t> shown_deck, shown_extra;
  std::array<std::array<int64_t, kExtraLedger>, 2> ledger{};
  std::map<std::pair<int64_t, int64_t>, Activation> activations;  // code, raw description
  std::vector<std::pair<public_effects::SeedLink, int>> chain;  // public link and state
  public_effects::Seed status;
  guards::ChainState guard_chain;
  mfenv::ChainCarry carry;
};

// Deliberately excludes trackers, private choices, known hands, hints, counters, temporal rows and delivery state.
struct PublicRootSeed {
  int turn = 0, turn_player = -1, phase = 0;
  std::vector<std::pair<public_effects::SeedLink, int>> chain;
  public_effects::Seed status;
  guards::ChainState guard_chain;
  mfenv::ChainCarry carry;
};

// The card-row table history records resolve codes against: the env's card ids (code -> id), built once per process
// after the card table is loaded; a later call with a different card table is refused.
template <class Map>
const mfenv::SemanticTable &CardRows(const Map &ids) {
  static std::once_flag once;
  static std::unique_ptr<mfenv::SemanticTable> table;
  static std::unordered_map<uint32_t, int64_t> rows;
  std::call_once(once, [&] {
    for (const auto &[code, id] : ids) rows.emplace(static_cast<uint32_t>(code), static_cast<int64_t>(id));
    if (rows.empty()) throw std::runtime_error("history card rows need the loaded card table");
    table = std::make_unique<mfenv::SemanticTable>(rows, "card table");
  });
  bool same = rows.size() == ids.size();
  for (auto it = ids.begin(); same && it != ids.end(); ++it) {
    const auto hit = rows.find(static_cast<uint32_t>(it->first));
    same = hit != rows.end() && hit->second == static_cast<int64_t>(it->second);
  }
  if (!same) throw std::runtime_error("the card table changed after the history card rows were built");
  return *table;
}

class History {
 public:
  explicit History(const mfenv::SemanticTable *rows = nullptr) : rows_(rows) {}

  void SetLaw(const WindowLaw &law) {
    if (law.window < 1 || law.chunk < 1 || law.chunk > law.window || law.cap < 0 || law.slots < 1)
      throw std::runtime_error("a history window law needs window >= chunk >= 1, cap >= 0 and slots >= 1");
    law_ = law;
    observers_[0].law = observers_[1].law = law;
  }
  const WindowLaw &Law() const { return law_; }
  // The closed turns one observer may await: the completed-turn bound of a game under the step limit.
  void SetPendingCap(int64_t max_steps) {
    pending_cap_ = CompletedTurnBound(max_steps);
    observers_[0].pending_cap = observers_[1].pending_cap = pending_cap_;
  }

  void Reset(const mfenv::SemanticTable &rows) {
    rows_ = &rows;
    cursor_ = 0;
    current_turn_ = 0;
    carry_ = mfenv::ChainCarry{};
    observers_[0] = Observer{};
    observers_[1] = Observer{};
    observers_[0].law = observers_[1].law = law_;
    observers_[0].pending_cap = observers_[1].pending_cap = pending_cap_;
    messages_.clear();
    facts_[0].clear();
    facts_[1].clear();
    chain_ = guards::ChainState{};
  }

  // Seed only facts still known at this root. No old events, choices, chunks or completed turns are made up.
  void InitializeCurrentRoot(int viewer, int turn, int turn_player, int phase, const CurrentRootSeed &seed) {
    if (rows_ == nullptr || viewer < 0 || viewer > 1 || turn < 1 || turn_player < 0 || turn_player > 1)
      throw std::runtime_error("invalid current-root history clock/viewer");
    Reset(*rows_);
    current_turn_ = turn;
    chain_ = seed.guard_chain;
    carry_ = seed.carry;
    for (int seat = 0; seat < 2; ++seat) {
      Observer &o = observers_[seat];
      o.turn = turn;
      o.turn_player = turn_player == seat ? 0 : 1;
      o.phase = phase;
      mfenv::TrackerSeed tracker = seed.tracker;
      if (seat != viewer) {
        for (auto &card : tracker.cards) card.side = 1 - card.side;
        tracker.counts = {};
        tracker.hints = {};
        o.passive = true;
      }
      o.tracker.SeedCurrent(turn, tracker);
    }
    Observer &o = observers_[viewer];
    for (const auto &card : seed.tracker.cards)
      if (card.code > 0 && card.location != kLocDeck && card.location != kLocExtra)
        o.seen[card.side].insert(static_cast<uint32_t>(card.code));
    for (const auto *counts : {&seed.hand, &seed.deck, &seed.extra})
      for (const auto &[code, count] : *counts) o.seen[1].insert(static_cast<uint32_t>(code));
    for (const auto *shown : {&seed.shown_deck, &seed.shown_extra})
      for (const auto &[sequence, code] : *shown) o.seen[1].insert(static_cast<uint32_t>(code));
    o.hand.SeedCurrent(o.tracker.ListEntities(1, kLocHand), seed.hand, seed.hand_group);
    o.zones.SeedCurrent(seed.deck, seed.extra, seed.shown_deck, seed.shown_extra);
    o.ledger = seed.ledger;
    for (const auto &[key, value] : seed.activations) {
      o.activations[{rows_->CardRow(static_cast<uint32_t>(key.first)), OwnDescIndex(key.first, key.second)}] = value;
      o.order = std::max(o.order, value.order);
    }
    for (const auto &[link, state] : seed.chain) {
      const ActivationKey key{rows_->CardRow(static_cast<uint32_t>(link.code)), OwnDescIndex(link.code, link.desc)};
      o.seen[link.side].insert(static_cast<uint32_t>(link.code));
      o.chain.push_back({link.number, link.side, key.second, link.origin[1], link.origin[2], state, key.first});
      if (state == kPending || state == kResolving) o.root_links.insert(link.number);
      if (o.activations.count(key)) {
        o.link_keys[link.number] = key;
        o.link_sides[link.number] = link.side;
      }
    }
    o.status.SeedCurrent(turn, seed.status, o.tracker);
  }

  // A const projection of the common-public portion of the existing incremental state, once at the real root.
  // ``public_places`` is limited to current field/grave/face-up-banished coordinates (relative to this viewer).
  PublicRootSeed CurrentPublicSeed(int viewer, const std::vector<public_effects::SeedPlace> &public_places) const {
    using public_effects::PublicRootSeedError;
    const Observer &o = observers_.at(viewer);
    if (o.turn != current_turn_ || o.turn < 1 || o.turn_player < 0 || o.turn_player > 1)
      throw PublicRootSeedError("observer_clock", "current observer clock is not a started public root");
    public_effects::PublicPlaces places;
    for (const auto &p : public_places) {
      if ((p[0] != 0 && p[0] != 1) ||
          (p[1] != kLocMzone && p[1] != kLocSzone && p[1] != kLocGrave && p[1] != kLocRemoved))
        throw PublicRootSeedError("private_anchor", "export was offered a private/non-public place");
      const int64_t entity = o.tracker.EntityAt(p[0], p[1], p[2]);
      if (!entity) throw PublicRootSeedError("untracked_public_place", "current public place has no tracked entity");
      if (!places.emplace(entity, p).second)
        throw PublicRootSeedError("ambiguous_public_entity", "one entity has multiple current public places");
    }
    PublicRootSeed out;
    out.turn = o.turn;
    out.turn_player = o.turn_player == 0 ? viewer : 1 - viewer;
    out.phase = o.phase;
    out.guard_chain = chain_;
    out.carry = carry_;
    std::set<int> numbers;
    for (const auto &link : o.chain)
      if (link.link < 1 || link.link > chain_.links || !numbers.insert(link.link).second)
        throw PublicRootSeedError("chain_clock", "displayed chain and public guard context disagree");
    out.status = o.status.CurrentPublicSeed(o.tracker, places, numbers);
    std::map<int, const Entry *> activation;
    if (!numbers.empty()) {
      // All current-turn rows remain in entries even when already delivered in chunks. Read only CHAINING facts
      // of the current chain; no event replay, own choices or other private records enter the returned DTO.
      for (auto it = o.entries.rbegin(); it != o.entries.rend(); ++it) {
        const auto &r = it->record;
        if (it->own || !it->activation || r.kind != static_cast<int>(mfenv::HistoryTokenKind::STATE_DELTA) ||
            r.subtype != static_cast<int>(mfenv::HistorySubtype::CHAIN) || r.value != mfenv::CHAIN_CHAINING) continue;
        if (numbers.count(r.link)) activation.emplace(r.link, &*it);
        if (r.link == 1) break;
      }
    }
    for (const auto &current : o.chain) {
      public_effects::SeedLink link;
      const auto projected = std::find_if(out.status.links.begin(), out.status.links.end(),
                                         [&](const auto &l) { return l.number == current.link; });
      if (projected != out.status.links.end()) link = *projected;
      else if (current.state == kPending || current.state == kResolving)
        throw PublicRootSeedError("missing_live_chain", "a live chain link has no incremental public effect state");
      else {
        link.number = current.link;
        link.side = current.side;
        link.negated = current.state == kNegated || current.state == kDisabled;
        link.left = true;  // settled: no subsequent application depends on its handler
      }
      if (const auto found = activation.find(current.link); found != activation.end()) {
        const auto &r = found->second->record;
        const public_effects::SeedPlace origin{static_cast<int>(r.from_controller_relative),
                                              static_cast<int>(r.from_location), static_cast<int>(r.from_sequence)};
        if (projected != out.status.links.end() &&
            (ArtworkBase(static_cast<uint32_t>(r.card_code)) != link.code || r.detail != link.desc ||
             origin != link.origin))
          throw PublicRootSeedError("chain_activation_mismatch", "current public chain records disagree");
        link.code = r.card_code;
        link.desc = r.detail;
        link.origin = origin;
      } else if (projected == out.status.links.end()) {
        throw PublicRootSeedError("missing_public_chain_activation", "current link lacks its public activation metadata");
      }
      // The fallback above is only the already-explicit current-root link of a cold observer; no private lookup.
      if (rows_->CardRow(static_cast<uint32_t>(link.code)) != current.card_row ||
          OwnDescIndex(link.code, link.desc) != current.desc_index || link.side != current.side)
        throw PublicRootSeedError("chain_activation_mismatch", "current chain display and public metadata disagree");
      out.chain.emplace_back(std::move(link), current.state);
    }
    return out;
  }

  // The public chain stack length and resolving-link ordinal after the latest buffer (guards.h).
  const guards::ChainState &Chain() const { return chain_; }
  // The public facts ``viewer`` has received this duel.
  int64_t FactCount(int viewer) const { return observers_.at(viewer).facts; }

  // Keep the duel's core messages and each viewer's facts (scripted duels and audience tests; off in training).
  void Keep(bool keep) { keep_ = keep; }
  // Client mode: ``viewer`` writes no observation (it is not the seat whose stream this is); its completed turns
  // are not held for delivery.
  void Passive(int viewer) { observers_.at(viewer).passive = true; }
  const std::vector<mfenv::Message> &Messages() const { return messages_; }
  const std::vector<mfenv::HistoryTokenRecord> &Facts(int viewer) const { return facts_.at(viewer); }
  // The code ``viewer``'s public tracker places at a relative place (0 when unknown there): known identities (#10).
  int64_t KnownCode(int viewer, int side, int location, int sequence) const {
    return observers_.at(viewer).tracker.KnownCode(side, location, sequence);
  }
  // The opponent's hand places (sequences) whose cards are in ``viewer``'s shuffled group: the hand's identities known
  // without positions are among these (unpositioned.h), and a reassignment of the hand must keep them there.
  std::vector<int> HandGroup(int viewer) const {
    const Observer &o = observers_.at(viewer);
    std::vector<int> out;
    const auto hand = o.tracker.ListEntities(1, kLocHand);
    for (size_t i = 0; i < hand.size(); ++i)
      if (o.hand.InGroup(hand[i].first)) out.push_back(static_cast<int>(i));
    return out;
  }
  // The location the card ``viewer``'s tracker places at a relative place came from (0 when unknown).
  int ArrivedFrom(int viewer, int side, int location, int sequence) const {
    return observers_.at(viewer).tracker.ArrivedFrom(side, location, sequence);
  }
  // The opponent's identities ``viewer`` knows without their positions, (location, code) -> copies: hand, deck and
  // face-down extra deck (unpositioned.h).
  std::map<std::pair<int, int64_t>, int64_t> Unpositioned(int viewer) const {
    const Observer &o = observers_.at(viewer);
    std::map<std::pair<int, int64_t>, int64_t> out;
    for (const auto &[code, copies] : o.hand.counts()) out[{kLocHand, code}] = copies;
    for (const auto &[code, copies] : o.zones.deck()) out[{kLocDeck, code}] = copies;
    for (const auto &[code, copies] : o.zones.extra()) out[{kLocExtra, code}] = copies;
    return out;
  }
  // obs:unpositioned_ of ``viewer`` (zeroed by the caller): rows (card id high byte, low byte, location id as in
  // cards_, copies) in (location, card id) order. ``view`` is the viewer's card rows: a location's copies beyond its
  // unshown opponent cards are a contradiction, and more rows than the table holds is refused like the card table's
  // overflow; both throw.
  void WriteUnpositioned(int viewer, const mfenv::TrackerView &view, uint8_t *out) const {
    std::map<int, int64_t> unshown, claimed;
    for (const mfenv::TrackerToken &t : view.tokens)
      if (t.side == 1 && t.code == 0 && (t.kind == mfenv::TrackerToken::CARD || t.kind == mfenv::TrackerToken::OTHER))
        ++unshown[t.location];
    std::vector<std::array<int64_t, 3>> rows;  // location id, card id, copies
    for (const auto &[key, copies] : Unpositioned(viewer)) {
      const int64_t id = rows_->CardRow(static_cast<uint32_t>(key.second));
      if (id <= 0)
        throw std::runtime_error("an identity without position has no card row: " + std::to_string(key.second));
      rows.push_back({LocationId(key.first), id, copies});
      claimed[key.first] += copies;
    }
    for (const auto &[location, copies] : claimed)
      if (copies > unshown[location])
        throw std::runtime_error("identities without positions: " + std::to_string(copies) + " copies at location " +
                                 std::to_string(location) + ", " + std::to_string(unshown[location]) +
                                 " unshown opponent cards there");
    if (rows.size() > static_cast<size_t>(kUnpositionedRows))
      throw std::runtime_error("identities without positions overflow: " + std::to_string(rows.size()) + " rows, " +
                               std::to_string(kUnpositionedRows) + " in the table");
    std::sort(rows.begin(), rows.end());
    for (size_t i = 0; i < rows.size(); ++i) {
      uint8_t *row = out + i * kUnpositionedWidth;
      row[0] = Clip(rows[i][1] >> 8);
      row[1] = Clip(rows[i][1] & 0xFF);
      row[2] = Clip(rows[i][0]);
      row[3] = Clip(rows[i][2]);
    }
  }
  // obs:card_status_ (zeroed, ``card_rows`` rows) and obs:public_effects_ (zeroed) of ``observer`` for its card rows
  // ``view`` (public_status.h); after Write, whose tracker export may resync a graveyard. Returns the player effects
  // beyond the table.
  int64_t WriteStatus(int observer, const mfenv::TrackerView &view, uint8_t *status, size_t card_rows,
                      uint8_t *effects) const {
    const Observer &o = observers_.at(observer);
    const std::vector<int64_t> rows = o.tracker.RowEntities(view);
    if (rows.size() > card_rows) throw std::runtime_error("the history view has more card rows than card_status_");
    return o.status.Write(rows, o.tracker, [&](int64_t code) { return rows_->CardRow(static_cast<uint32_t>(code)); },
                          status, effects);
  }
  // The card codes ``viewer``'s facts have shown this duel: its own, then its opponent's (the announce law's seen sets).
  const std::array<std::set<uint32_t>, 2> &Seen(int viewer) const { return observers_.at(viewer).seen; }

  void ConsumeBuffer(const uint8_t *data, size_t len) {
    if (rows_ == nullptr) throw std::runtime_error("history consumed a buffer before its card rows were set");
    const std::vector<mfenv::Message> messages = mfenv::SplitMessages(data, len);
    if (keep_) messages_.insert(messages_.end(), messages.begin(), messages.end());
    for (const mfenv::Message &message : messages) chain_.Observe(message.msg);
    const mfenv::CarriedInterval interval =
        mfenv::CarriedIntervalTokens(messages, cursor_, &current_turn_, &carry_, *rows_);
    std::vector<std::pair<int64_t, const mfenv::Message *>> swaps;
    for (size_t i = 0; i < messages.size(); ++i)
      if (messages[i].msg == kMsgSwap) swaps.emplace_back(cursor_ + static_cast<int64_t>(i), &messages[i]);
    for (int viewer = 0; viewer < 2; ++viewer) {
      Observer &o = observers_[viewer];
      size_t next = 0;
      auto swap_before = [&](int64_t trace) {
        while (next < swaps.size() && swaps[next].first < trace) {
          const auto &body = swaps[next++].second->payload;
          if (body.size() < 16 || body[4] > 1 || body[12] > 1) throw std::runtime_error("malformed swap message");
          o.tracker.Swap(body[4] == viewer ? 0 : 1, body[5] & 0x7F, body[6], body[12] == viewer ? 0 : 1,
                         body[13] & 0x7F, body[14]);
        }
      };
      if (const auto hit = interval.by_viewer.find(viewer); hit != interval.by_viewer.end())
        for (const mfenv::HistoryTokenRecord &record : hit->second) {
          swap_before(record.trace_index);
          if (keep_) facts_[viewer].push_back(record);
          ++o.facts;
          const HandUnpositioned::Hand before = o.tracker.ListEntities(1, kLocHand);
          const int64_t known_from = KnownFrom(o, record);
          std::vector<int64_t> grave;
          if (record.kind == static_cast<int>(mfenv::HistoryTokenKind::PUBLIC_EVENT) &&
              record.public_event_kind == mfenv::PUB_SWAP_GRAVE_DECK && record.player_relative == 1)
            for (const auto &[entity, code] : o.tracker.ListEntities(1, kLocGrave)) grave.push_back(code);
          const bool shuffle_set = record.kind == static_cast<int>(mfenv::HistoryTokenKind::PUBLIC_EVENT) &&
                                   record.public_event_kind == mfenv::PUB_SHUFFLE_SET;
          std::vector<std::array<int64_t, 4>> facedown;  // a shuffle of set cards: (side, location, sequence, entity)
          if (shuffle_set) facedown = FaceDownField(o.tracker);
          const mfenv::TrackerRowAnnotation note = o.tracker.Consume(record);
          o.hand.Update(record, before, o.tracker.ListEntities(1, kLocHand));
          o.zones.Update(record, known_from, grave);
          o.status.Update(record, note, o.tracker,
                          record.card_code > 0 ? ArtworkBase(static_cast<uint32_t>(record.card_code)) : 0);
          if (shuffle_set) RenewStatus(o, viewer, facedown, messages, record.trace_index);
          o.Fact(record, note);
        }
      swap_before(INT64_MAX);
    }
    cursor_ += static_cast<int64_t>(messages.size());
  }

  // After each buffer: each player's public (face-up) hand cards, (sequence, code); the other viewer's client is sent
  // them with their places (unpositioned.h).
  void ShowHands(const std::array<std::vector<std::pair<int, int64_t>>, 2> &public_hands) {
    for (int viewer = 0; viewer < 2; ++viewer) {
      Observer &o = observers_[viewer];
      o.hand.Show(o.tracker.ListEntities(1, kLocHand), public_hands[1 - viewer]);
    }
  }

  void ConsumeOwnChoice(int viewer, const OwnChoice &c) {
    Observer &o = observers_[viewer];
    Entry e;
    e.own = true;
    e.msg = c.msg;
    e.act = c.act;
    e.effect = c.effect;
    e.phase = o.phase;
    mfenv::HistoryTokenRecord &r = e.record;
    r.kind = static_cast<int>(mfenv::HistoryTokenKind::ACTION);
    r.subtype = static_cast<int>(mfenv::HistorySubtype::ACTION_SELECTED);
    r.turn = o.turn;
    r.player_relative = 0;
    r.card_row = c.card_row;
    mfenv::TrackerToken token;
    if (c.has_place) {
      r.from_controller_relative = c.side;
      r.from_location = c.location;
      r.from_sequence = c.sequence;
      token.kind = c.overlay ? mfenv::TrackerToken::OVERLAY_MATERIAL : mfenv::TrackerToken::CARD;
      token.side = c.side;
      token.location = c.location;
      token.sequence = c.sequence;
      token.overlay_index = c.overlay_index;
    }
    e.note = o.tracker.ConsumeOwnChoice(r, c.has_place ? &token : nullptr);
    o.Append(std::move(e));
  }

  // Uncapped condensed-row counts of one observer (measurement: the history builder's full rows, whatever the windows
  // keep): its current turn (turn number, rows) and its closed windows ((turn, rows) each, most recent first, (0, 0)
  // where none).
  std::array<int64_t, 2> CurrentTurnRows(int observer) const {
    const Observer &o = observers_.at(observer);
    return {o.turn, static_cast<int64_t>(o.entries.size())};
  }
  std::array<std::array<int64_t, 2>, kClosedTurns> ClosedTurnRows(int observer) const {
    const Observer &o = observers_.at(observer);
    std::array<std::array<int64_t, 2>, kClosedTurns> out{};
    for (size_t k = 0; k < o.recent.size() && k < static_cast<size_t>(kClosedTurns); ++k) out[k] = o.recent[k];
    return out;
  }

  // Fills one observer's arrays (each zeroed by the caller, with the shapes of this header's constants) against its
  // card rows ``view`` (row i of cards_ is token i) at decision ``decision``: the current turn's window (its rows from
  // the first not yet chunked), the chunks and closed windows delivered at this decision (Delivery: oldest first,
  // kChunkSlots chunks, kClosedTurns windows, a closed window only once all its chunks have reached the observer; the
  // rest wait, never skipped), the chain, activation, ledger, hint and card-turn tables. Chunk meta (int): valid, turn,
  // chunk index in the turn, rows; closed meta (int): valid, turn, rows, chunks of that turn. Every row's references
  // resolve against the current card rows. Returns (rows of the current turn outside the window -- all chunked --,
  // chunks still waiting for this observer, completed turns still queued for it after this decision's deliveries).
  std::array<int64_t, 3> Write(int observer, int64_t decision, const mfenv::TrackerView &view, uint8_t *events,
                               uint8_t *refs, uint8_t *chain, uint8_t *activations, uint8_t *ledger, uint8_t *hints,
                               uint8_t *card_turn, size_t card_rows, uint8_t *chunk_events, uint8_t *chunk_refs,
                               int32_t *chunk_meta, uint8_t *closed_events, uint8_t *closed_refs,
                               int32_t *closed_meta) {
    Observer &o = observers_[observer];
    const int turn = o.turn;
    const WindowLaw &law = law_;
    if (o.staged.decision != decision) o.Commit();
    // this decision's deliveries, from what has been committed (the same whenever this decision is written again)
    Delivery plan;
    plan.decision = decision;
    plan.closed_chunks.assign(o.pending.size(), 0);
    plan.closed_window.assign(o.pending.size(), 0);
    struct Chunk { const std::vector<Entry> *entries; int turn, index; };
    std::vector<Chunk> chunks;
    int64_t backlog = 0;
    for (size_t p = 0; p < o.pending.size(); ++p)
      for (int c = o.pending[p].chunks_delivered; c < o.pending[p].chunks; ++c) {
        if (static_cast<int>(chunks.size()) < law.slots) {
          chunks.push_back({&o.pending[p].entries, o.pending[p].turn, c});
          ++plan.closed_chunks[p];
        } else {
          ++backlog;
        }
      }
    for (int c = o.chunks_delivered; c < o.chunks; ++c) {
      if (static_cast<int>(chunks.size()) < law.slots) {
        chunks.push_back({&o.entries, turn, c});
        ++plan.current_chunks;
      } else {
        ++backlog;
      }
    }
    std::vector<size_t> windows;
    for (size_t p = 0; p < o.pending.size() && static_cast<int>(windows.size()) < kClosedTurns; ++p) {
      const ClosedTurn &t = o.pending[p];
      if (t.window_delivered) continue;
      if (t.chunks_delivered + plan.closed_chunks[p] < t.chunks) break;  // its chunks first, and older turns first
      plan.closed_window[p] = 1;
      windows.push_back(p);
    }
    o.staged = plan;

    // one export resolves every row's references: the current window, the chunks, the closed windows
    const size_t first = static_cast<size_t>(law.chunk) * o.chunks;
    std::vector<mfenv::TrackerRowAnnotation> notes;
    for (size_t i = first; i < o.entries.size(); ++i) notes.push_back(o.entries[i].note);
    for (const Chunk &c : chunks)
      for (int k = 0; k < law.chunk; ++k) notes.push_back((*c.entries)[c.index * law.chunk + k].note);
    for (size_t p : windows)
      for (size_t i = static_cast<size_t>(law.chunk) * o.pending[p].chunks; i < o.pending[p].entries.size(); ++i)
        notes.push_back(o.pending[p].entries[i].note);
    const mfenv::TrackerExport exported = o.tracker.Export(view, turn, notes);

    size_t link = 0;
    for (size_t i = first, k = 0; i < o.entries.size(); ++i, ++k)
      WriteRow(o.entries[i], exported.links[link++], events + k * kEventWidth, refs + k * kRefWidth);
    for (size_t s = 0; s < chunks.size(); ++s) {
      const Chunk &c = chunks[s];
      for (int k = 0; k < law.chunk; ++k)
        WriteRow((*c.entries)[c.index * law.chunk + k], exported.links[link++],
                 chunk_events + (s * law.chunk + k) * kEventWidth, chunk_refs + (s * law.chunk + k) * kRefWidth);
      int32_t *meta = chunk_meta + s * kChunkMetaWidth;
      meta[0] = 1;
      meta[1] = c.turn;
      meta[2] = c.index;
      meta[3] = law.chunk;
    }
    for (size_t w = 0; w < windows.size(); ++w) {
      const ClosedTurn &t = o.pending[windows[w]];
      const size_t start = static_cast<size_t>(law.chunk) * t.chunks;
      for (size_t i = start, k = 0; i < t.entries.size(); ++i, ++k)
        WriteRow(t.entries[i], exported.links[link++], closed_events + (w * law.window + k) * kEventWidth,
                 closed_refs + (w * law.window + k) * kRefWidth);
      int32_t *meta = closed_meta + w * kClosedMetaWidth;
      meta[0] = 1;
      meta[1] = t.turn;
      meta[2] = static_cast<int32_t>(t.entries.size() - start);
      meta[3] = t.chunks;
      meta[4] = t.turn_player == 0 ? observer : (t.turn_player == 1 ? 1 - observer : -1);  // the absolute seat
    }

    for (size_t i = 0; i < o.chain.size() && i < static_cast<size_t>(kChainRows); ++i) {
      const ChainLink &l = o.chain[i];
      uint8_t *row = chain + i * kChainWidth;
      row[0] = 1;
      row[1] = Clip(l.side);
      row[2] = Clip(l.card_row >> 8);
      row[3] = Clip(l.card_row & 0xFF);
      row[4] = Clip(l.desc_index);
      row[5] = LocationId(l.location);
      row[6] = Clip(l.sequence);
      row[7] = Clip(l.state);
    }

    std::vector<std::pair<const ActivationKey *, const Activation *>> ordered;
    for (const auto &entry : o.activations) ordered.emplace_back(&entry.first, &entry.second);
    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
      const bool ta = a.second->turn[0] + a.second->turn[1] > 0, tb = b.second->turn[0] + b.second->turn[1] > 0;
      if (ta != tb) return ta;
      return a.second->order > b.second->order;
    });
    for (size_t i = 0; i < ordered.size() && i < static_cast<size_t>(kActivationRows); ++i) {
      const ActivationKey &key = *ordered[i].first;
      const Activation &a = *ordered[i].second;
      uint8_t *row = activations + i * kActivationWidth;
      row[0] = 1;
      row[1] = Clip(key.first >> 8);
      row[2] = Clip(key.first & 0xFF);
      row[3] = Clip(key.second);
      row[4] = Clip(a.turn[0]);
      row[5] = Clip(a.turn[1]);
      row[6] = Clip(a.resolved[0]);
      row[7] = Clip(a.resolved[1]);
      row[8] = Clip(a.negated);
      row[9] = Clip(a.duel[0]);
      row[10] = Clip(a.duel[1]);
      row[11] = Clip(turn - a.last_turn);
    }

    for (int side = 0; side < 2; ++side) {
      uint8_t *row = ledger + side * kLedgerWidth;
      const size_t base = static_cast<size_t>(side) * (mfenv::kTrackerTurnCounts + 1);
      for (int i = 0; i < mfenv::kTrackerTurnCounts; ++i) row[i] = Clip(exported.ledger[base + i]);
      row[6] = Clip(o.ledger[side][0]);
      row[7] = Clip(o.ledger[side][1]);
      row[8] = Clip(o.ledger[side][2]);
      row[9] = Clip(o.ledger[side][3]);
      row[10] = Clip(o.ledger[side][4]);
      row[11] = Clip(o.ledger[side][5] / 100);
      row[12] = Clip(o.ledger[side][6] / 100);
      row[13] = Clip(o.ledger[side][7]);
      row[14] = Clip(exported.ledger[base + mfenv::kTrackerTurnCounts]);
      row[15] = Clip(exported.ledger[mfenv::kTrackerLedgerWidth - 1]);
    }

    for (size_t i = 0; i < exported.hints.size() && i < static_cast<size_t>(kHintRows); ++i) {
      const auto &h = exported.hints[i];  // side, desc, count
      const int64_t code = h[1] >> 4;
      const int64_t row_id = code > 0 ? rows_->CardRow(static_cast<uint32_t>(code)) : 0;
      uint8_t *row = hints + i * kHintWidth;
      row[0] = Clip(h[2]);
      row[1] = Clip(h[0]);
      row[2] = Clip(row_id >> 8);
      row[3] = Clip(row_id & 0xFF);
      row[4] = Clip(DescIndex(h[1]));
    }

    if (exported.cards.size() > card_rows) throw std::runtime_error("the history view has more card rows than cards_");
    for (size_t i = 0; i < exported.cards.size(); ++i) {
      const auto &c = exported.cards[i];
      if (!c[0]) continue;
      uint8_t *row = card_turn + i * kCardTurnWidth;
      row[0] = Clip(c[3]);   // activations this turn
      row[1] = Clip(c[4]);   // attacks this turn
      row[2] = Clip(c[1]);   // arrival bucket
      row[3] = Clip(c[2]);   // arrival kind
      row[4] = Clip(c[6]);   // counter total
      row[5] = Clip(c[7]);   // counter kinds
      row[6] = Clip(c[12]);  // card hints
      row[7] = 0;            // entity code (reserved)
    }
    int64_t queued = 0;  // completed turns whose window this decision does not deliver (two per decision)
    for (size_t p = 0; p < o.pending.size(); ++p) queued += !o.pending[p].window_delivered && !plan.closed_window[p];
    return {static_cast<int64_t>(first), backlog, queued};
  }

  static int mfenv_card_effect_offset() { return 10010; }

  // One window row (kEventWidth columns) and its references (kRefWidth) from an entry and its resolved links.
  static void WriteRow(const Entry &e, const std::array<int64_t, mfenv::kTrackerLinkWidth> &link, uint8_t *row,
                       uint8_t *ref) {
    const mfenv::HistoryTokenRecord &r = e.record;
    row[0] = 1;
    row[1] = Clip(r.kind);
    row[2] = Clip(r.subtype);
    row[3] = Clip(e.own ? e.msg : r.public_event_kind);  // an own decision: the prompt message it answered
    row[4] = Clip(r.player_relative);
    row[5] = PhaseId(e.phase);
    const int64_t card = r.card_row > 0 ? r.card_row : 0;
    row[6] = Clip(card >> 8);
    row[7] = Clip(card & 0xFF);
    row[8] = Clip(r.from_controller_relative);
    row[9] = LocationId(r.from_location);
    row[10] = (r.from_location & 0x7F) == kLocDeck ? 0 : Clip(r.from_sequence);  // deck order: design 5a
    row[11] = Clip(r.to_controller_relative);
    row[12] = LocationId(r.to_location);
    row[13] = (r.to_location & 0x7F) == kLocDeck ? 0 : Clip(r.to_sequence);
    row[14] = Clip(r.from_position);
    row[15] = Clip(r.to_position);
    row[16] = Clip(r.reason & 0xFF);
    row[17] = e.activation ? Clip(e.state) : Clip(r.value);
    const int64_t amount = std::min<int64_t>(65535, std::max<int64_t>(0, r.amount));
    row[18] = Clip(amount >> 8);
    row[19] = Clip(amount & 0xFF);
    row[20] = Clip(r.count);
    row[21] = Clip(r.link);
    row[22] = Clip(e.own ? (e.effect >= mfenv_card_effect_offset() ? e.effect - mfenv_card_effect_offset() : 15)
                         : OwnDescIndex(r.card_code, r.detail));
    row[23] = Clip(e.own ? e.act : 0);
    // the full reason (public_effects/v1): row[16] holds bits 0-7, these bits 8-15, 16-23 and 24-31 (REASON_DISCARD,
    // RETURN, the summon-material reasons, REDIRECT, LINK)
    const uint64_t reason = static_cast<uint64_t>(r.reason);
    row[24] = static_cast<uint8_t>((reason >> 8) & 0xFF);
    row[25] = static_cast<uint8_t>((reason >> 16) & 0xFF);
    row[26] = static_cast<uint8_t>((reason >> 24) & 0xFF);
    ref[0] = link[0] >= 0 ? Clip(link[0] + 1) : 0;
    ref[1] = Clip(link[1]);
    ref[2] = link[2] >= 0 ? Clip(link[2] + 1) : 0;
    ref[3] = Clip(link[3]);
  }

 private:
  // The face-down field cards ``tracker`` follows: (side, location, sequence, entity).
  static std::vector<std::array<int64_t, 4>> FaceDownField(const mfenv::PublicTracker &tracker) {
    std::vector<std::array<int64_t, 4>> out;
    for (int side = 0; side < 2; ++side)
      for (int location : {kLocMzone, kLocSzone})
        for (int sequence = 0; sequence < 8; ++sequence) {
          const int64_t entity = tracker.EntityAt(side, location, sequence);
          if (entity && tracker.FaceDown(entity)) out.push_back({side, location, sequence, entity});
        }
    return out;
  }

  // After a shuffle of set cards (the tracker renewed every face-down field entity): the public status of the cards
  // at untouched places carries over, the shuffled places (the message's) keep what all their cards had.
  void RenewStatus(Observer &o, int viewer, const std::vector<std::array<int64_t, 4>> &facedown,
                   const std::vector<mfenv::Message> &messages, int64_t trace) {
    const int64_t local = trace - cursor_;
    if (local < 0 || local >= static_cast<int64_t>(messages.size()) || messages[local].msg != kMsgShuffleSetCard)
      throw std::runtime_error("a shuffle-set fact without its MSG_SHUFFLE_SET_CARD");
    const auto &body = messages[local].payload;
    if (body.size() < 2 || body.size() < 2 + 4 * static_cast<size_t>(body[1]))
      throw std::runtime_error("malformed MSG_SHUFFLE_SET_CARD");
    std::set<std::array<int64_t, 3>> shuffled;
    for (int i = 0; i < body[1]; ++i) {
      const uint8_t *at = body.data() + 2 + 4 * i;
      shuffled.insert({at[0] == viewer ? 0 : 1, at[1], at[2]});
    }
    std::vector<std::pair<int64_t, int64_t>> kept;
    std::vector<int64_t> group_old, group_new;
    std::vector<std::array<int, 3>> group_places;
    for (const auto &[side, location, sequence, entity] : facedown) {
      const int64_t now = o.tracker.EntityAt(static_cast<int>(side), static_cast<int>(location),
                                             static_cast<int>(sequence));
      if (shuffled.count({side, location, sequence})) {
        group_old.push_back(entity);
        if (now) {
          group_new.push_back(now);
          group_places.push_back({static_cast<int>(side), static_cast<int>(location), static_cast<int>(sequence)});
        }
      } else if (now) {
        kept.emplace_back(entity, now);
      }
    }
    o.status.Renew(kept, group_old, group_new, group_places);
  }

  // The identity ``o``'s viewer knows for the card a move record takes from its origin, before the record: the
  // record's code, the tracker's, or a public face-up hand card's (0 when unknown, or not a move).
  static int64_t KnownFrom(const Observer &o, const mfenv::HistoryTokenRecord &r) {
    if (r.kind != static_cast<int>(mfenv::HistoryTokenKind::STATE_DELTA) ||
        r.subtype != static_cast<int>(mfenv::HistorySubtype::MOVE) || r.from_location == 0)
      return 0;
    if (r.card_code > 0) return r.card_code;
    const int64_t entity = o.tracker.EntityFrom(r);
    if (entity == 0) return 0;
    const int64_t code = o.tracker.CodeOf(entity);
    return code > 0 ? code : o.hand.ShownCode(entity);
  }

  const mfenv::SemanticTable *rows_ = nullptr;
  WindowLaw law_;
  int64_t pending_cap_ = CompletedTurnBound(0);
  int64_t cursor_ = 0;
  int current_turn_ = 0;
  mfenv::ChainCarry carry_;
  std::array<Observer, 2> observers_;
  bool keep_ = false;
  guards::ChainState chain_;
  std::vector<mfenv::Message> messages_;
  std::array<std::vector<mfenv::HistoryTokenRecord>, 2> facts_;
};

}  // namespace history
}  // namespace duelenv
