// The env's public status tracker (law public_effects/v1): lingering effects a player knows from its public facts
// and card text, per viewer, written into obs:card_status_ and obs:public_effects_.
//
// The engine keeps lingering effects in its registry, which no client receives. A client sees the chain link that
// creates one (card, description, activating player, targets), whether it resolved, and what its resolution moved;
// the card's text says what it leaves behind. The table (public_effects_table.h, generated from
// mirrorforce/agent/env/public_effects/public-effects-v1.json) holds those rules per (card, description); a rule applies
// when its link resolves without being negated, to the cards its selector names:
// - moved: the cards a move of the resolution brought from a 'from' location to a 'to' location, still there at its
//   end (Multirole's re-set spells, Shark Cannon's summon, Halqifibrax's tuner);
// - targets: the link's targets still face-up at their targeted monster zone (optionally only the activator's
//   opponent's), Effect Veiler and Widow Anchor;
// - control_taken: monsters whose control the resolution moved to the activator, who is not the player they came
//   to the field under (Widow Anchor; the engine registers no lingering control when a monster returns to that
//   player, so the bit is not set);
// - self: the link's card, still face-up at its place (Borrelsword Dragon);
// - player: a player-level effect owned by the activator (Maxx "C", Multirole's first effect, Topologic Bomber Dragon).
// Each status bit ends as the law says (the reset its script registers): at the end of the turn, when the card leaves
// the field, is turned face-down or changes control (see the bits below). Equip relations come from MSG_EQUIP and
// MSG_UNEQUIP and end when either card leaves the field. Every input is a public fact the viewer received, so two
// truths behind one public stream give one table, and a client building the same facts (client_driver.h) writes the
// same arrays.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "mfenv/history.h"
#include "mfenv/public_tracker.h"

namespace duelenv {
namespace public_effects {

enum class Selector : int { kMoved, kTargets, kControlTaken, kSelf, kPlayer };
enum class TargetSide : int { kAny, kOpponent };
struct Rule {
  uint32_t code;
  int64_t desc;  // code * 16 + index, or 0 for an activation without a description
  Selector selector;
  int from_mask, to_mask;  // moved: base location masks
  uint8_t status;          // the status bit it sets (0 for a player rule)
  int kind;                // the player effect kind (player rules)
  TargetSide side;
  bool stays;  // a continuous spell/trap's effect: applies only if the card is still where it was activated
};

}  // namespace public_effects
}  // namespace duelenv

#include "duel/public_effects_table.h"

namespace duelenv {
namespace public_effects {

constexpr const char *kLaw = "public_effects/v1";
constexpr int kStatusWidth = 6;                    // obs:card_status_ columns
constexpr int kEffectRows = 16, kEffectWidth = 8;  // obs:public_effects_

// Status bits (obs:card_status_ column 3).
constexpr uint8_t kBanishOnLeave = 1 << 0;   // banished when it leaves the field (Multirole's re-set)
constexpr uint8_t kCannotAttack = 1 << 1;    // cannot attack (Shark Cannon's summon)
constexpr uint8_t kCannotActivate = 1 << 2;  // cannot activate its effects this turn (Halqifibrax's tuner)
constexpr uint8_t kNegatedUntilEnd = 1 << 3; // effects negated until the end of the turn (Veiler, Widow Anchor)
constexpr uint8_t kControlReturns = 1 << 4;  // control taken until the End Phase (Widow Anchor)
constexpr uint8_t kAttacksTwice = 1 << 5;    // can make a second attack this turn (Borrelsword Dragon)
// What ends them: every bit ends when the card leaves the field (a move off the monster and spell zones, or between
// them); these end at the end of the turn, when it is turned face-down, when its control changes again.
constexpr uint8_t kEndsAtTurnEnd = kCannotActivate | kNegatedUntilEnd | kControlReturns | kAttacksTwice;
constexpr uint8_t kEndsFaceDown = kBanishOnLeave | kCannotAttack | kCannotActivate | kNegatedUntilEnd | kAttacksTwice;
constexpr uint8_t kEndsControlChange = kControlReturns;

// Player effect kinds (obs:public_effects_ column 5).
constexpr int kDrawOnOpponentSpecialSummon = 1, kNoResponseToSpells = 2, kOtherMonstersCannotAttack = 3;

// Whether a card's lingering public effects are reviewed (the table holds its rules, or it has none).
inline bool Reviewed(uint32_t code) { return std::binary_search(std::begin(kReviewed), std::end(kReviewed), code); }

// Current public facts use places, never engine/private entity ids. An absent source is {-1, 0, 0}.
using SeedPlace = std::array<int, 3>;
struct SeedMoved {
  SeedPlace place;  // current public anchor of the entity
  int from_location = 0;
  bool control = false;
  SeedPlace destination{-1, 0, 0};  // this move's original destination, not necessarily the current place
};
struct SeedLink {
  int number = 0, side = 0;
  int64_t code = 0, desc = 0;
  SeedPlace origin{-1, 0, 0}, source{-1, 0, 0};
  bool negated = false, left = false;
  std::vector<SeedPlace> targets;
  std::vector<SeedMoved> moved;
};
struct SeedStatus { SeedPlace place, source; uint8_t bits = 0; int turn = 0; };
struct Seed {
  std::vector<SeedStatus> cards;
  std::vector<std::pair<SeedPlace, SeedPlace>> equips;
  std::vector<std::pair<SeedPlace, int>> field_origins;
  std::vector<std::array<int64_t, 5>> effects;  // code, desc, relative owner, kind, turn
  std::vector<SeedLink> links;
  int resolving = 0;
};

class PublicRootSeedError : public std::runtime_error {
 public:
  PublicRootSeedError(const std::string &reason, const std::string &detail)
      : std::runtime_error("current_public_root_seed/" + reason + ": " + detail) {}
};
using PublicPlaces = std::map<int64_t, SeedPlace>;

class Tracker {
 public:
  void SeedCurrent(int turn, const Seed &seed, const mfenv::PublicTracker &tracker) {
    *this = Tracker{};
    turn_ = turn;
    auto entity = [&](const SeedPlace &place) -> int64_t {
      if (place[0] == -1) return 0;
      const int64_t found = tracker.EntityAt(place[0], place[1], place[2]);
      if (!found) throw std::runtime_error("current public effect names an absent card");
      return found;
    };
    for (const auto &s : seed.cards) cards_[entity(s.place)] = Status{s.bits, entity(s.source), s.turn};
    for (const auto &[equip, host] : seed.equips) equips_[entity(equip)] = entity(host);
    for (const auto &[place, side] : seed.field_origins) owners_[entity(place)] = side;
    for (const auto &e : seed.effects)
      effects_.push_back({e[0], e[1], static_cast<int>(e[2]), static_cast<int>(e[3]), static_cast<int>(e[4])});
    for (const auto &s : seed.links) {
      Link link;
      link.code = s.code;
      link.desc = s.desc;
      link.entity = entity(s.source);
      link.side = s.side;
      link.place_side = s.origin[0];
      link.location = s.origin[1];
      link.sequence = s.origin[2];
      link.negated = s.negated;
      link.left = s.left;
      for (const auto &p : s.targets) link.targets.push_back({entity(p), p[0], p[1], p[2]});
      for (const auto &m : s.moved)
        {
          const auto &to = m.destination[0] < 0 ? m.place : m.destination;
          link.moved.push_back({entity(m.place), m.from_location, to[0], to[1], to[2], m.control});
        }
      links_[s.number] = std::move(link);
    }
    resolving_ = seed.resolving;
  }

  // Common-public state only. Entity ids never leave this method; private source references become absent.
  // ``places`` contains only current public field/grave/face-up-banished anchors supplied by the client.
  Seed CurrentPublicSeed(const mfenv::PublicTracker &tracker, const PublicPlaces &places,
                         const std::set<int> &current_links) const {
    Seed out;
    auto source = [&](int64_t entity) -> SeedPlace {
      const auto found = places.find(entity);
      return found == places.end() ? SeedPlace{-1, 0, 0} : found->second;
    };
    auto required = [&](int64_t entity, const std::string &reason, const std::string &detail,
                        bool field_only = false) -> SeedPlace {
      const auto p = source(entity);
      if (p[0] < 0 || (field_only && !OnField(p[1]))) throw PublicRootSeedError(reason, detail);
      return p;
    };
    for (const auto &[entity, status] : cards_)
      if (status.bits)
        out.cards.push_back({required(entity, "field_status_reference", "active status has no public field anchor", true),
                             source(status.source), status.bits, status.turn});
    for (const auto &[equip, host] : equips_)
      out.equips.emplace_back(required(equip, "equip_reference", "equip has no public field anchor", true),
                              required(host, "equip_reference", "equip target has no public field anchor", true));
    for (const auto &[entity, side] : owners_)
      out.field_origins.emplace_back(required(entity, "field_origin_reference", "field origin has no public anchor", true), side);
    for (const auto &effect : effects_) {
      const bool supported = std::any_of(std::begin(kRules), std::end(kRules), [&](const Rule &r) {
        return r.code == effect.code && r.desc == effect.desc && r.selector == Selector::kPlayer && r.kind == effect.kind;
      });
      if (!supported) throw PublicRootSeedError("unsupported_player_effect", "current effect has no reviewed public rule");
      out.effects.push_back({effect.code, effect.desc, effect.owner, effect.kind, effect.turn});
    }
    for (int number : current_links) {
      const auto found = links_.find(number);
      if (found == links_.end()) continue;  // a settled link can remain in the displayed chain until link 1 settles
      const Link &link = found->second;
      if (!Reviewed(static_cast<uint32_t>(link.code)))
        throw PublicRootSeedError("unreviewed_chain_handler", "link " + std::to_string(number) + " has no reviewed capability");
      SeedLink projected;
      projected.number = number;
      projected.side = link.side;
      projected.code = link.code;
      projected.desc = link.desc;
      projected.origin = {link.place_side, link.location, link.sequence};
      projected.source = source(link.entity);
      projected.negated = link.negated;
      projected.left = link.left;
      auto applies = [&](const Rule &rule) {
        if (rule.code != link.code || rule.desc != link.desc || link.negated) return false;
        // A left/absent/off-field handler cannot become a stayed continuous handler again during this link.
        if (rule.stays && (link.left || !link.entity || !OnField(link.location))) return false;
        return true;
      };
      bool targets = false, handler_required = false;
      for (const auto &rule : kRules) {
        if (!applies(rule)) continue;
        targets = targets || rule.selector == Selector::kTargets;
        handler_required = handler_required || rule.stays ||
            (rule.selector == Selector::kSelf && !link.left && link.entity && OnField(link.location));
      }
      if (handler_required) {
        projected.source = required(link.entity, "chain_handler_reference",
                                    "link " + std::to_string(number) + " needs a current public handler", true);
        if (tracker.EntityAt(link.place_side, link.location, link.sequence) != link.entity)
          throw PublicRootSeedError("chain_handler_reference", "stayed handler disagrees with its public origin");
      }
      if (targets)
        for (const Target &target : link.targets) {
          // Every v1 target rule reads only MZONE. A move from any other location invalidates the target, never
          // promotes it into an MZONE target (Move); therefore these private/non-MZONE records are irrelevant.
          if (target.location != kMzone) continue;
          const auto p = required(target.entity, "private_chain_target",
                                  "link " + std::to_string(number) + " has an unanchored relevant target", true);
          if (p != SeedPlace{target.side, target.location, target.sequence})
            throw PublicRootSeedError("chain_target_reference", "current target and public anchor disagree");
          projected.targets.push_back(p);
        }
      for (const Moved &moved : link.moved) {
        bool relevant = false;
        for (const auto &rule : kRules) {
          if (!applies(rule)) continue;
          relevant = relevant || (rule.selector == Selector::kMoved && (moved.from_location & rule.from_mask) &&
                                   (moved.to_location & rule.to_mask)) ||
                     (rule.selector == Selector::kControlTaken && moved.control && moved.to_side == link.side);
        }
        // Rule predicates here are immutable for this move. A current EntityAt mismatch alone is NOT a proof:
        // the same entity may return to the original destination before the link resolves.
        if (!relevant) continue;
        const auto p = required(moved.entity, "private_chain_move", "link " + std::to_string(number) +
                                " has a rule-relevant moved participant without a public anchor");
        projected.moved.push_back({p, moved.from_location, moved.control,
                                   {moved.to_side, moved.to_location, moved.to_sequence}});
      }
      out.links.push_back(std::move(projected));
    }
    out.resolving = current_links.count(resolving_) ? resolving_ : 0;
    // Public coordinates determine ordering; internal entity allocation can depend on a viewer's private stream.
    std::sort(out.cards.begin(), out.cards.end(), [](const auto &a, const auto &b) { return a.place < b.place; });
    std::sort(out.equips.begin(), out.equips.end());
    std::sort(out.field_origins.begin(), out.field_origins.end());
    return out;
  }
  // One fact of the viewer's stream, after the public tracker consumed it (``note`` its annotation); ``code`` the
  // fact's card code with an artwork variant mapped to its base card.
  void Update(const mfenv::HistoryTokenRecord &r, const mfenv::TrackerRowAnnotation &note,
              const mfenv::PublicTracker &tracker, int64_t code) {
    using mfenv::HistorySubtype;
    using mfenv::HistoryTokenKind;
    const auto kind = static_cast<HistoryTokenKind>(r.kind);
    const auto subtype = static_cast<HistorySubtype>(r.subtype);
    if (kind == HistoryTokenKind::BOUNDARY && subtype == HistorySubtype::NEW_TURN) {
      turn_ = r.turn;
      for (auto it = cards_.begin(); it != cards_.end();) {
        it->second.bits &= static_cast<uint8_t>(~kEndsAtTurnEnd);
        it = it->second.bits ? std::next(it) : cards_.erase(it);
      }
      effects_.clear();
      links_.clear();
      resolving_ = 0;
      return;
    }
    if (kind == HistoryTokenKind::PUBLIC_EVENT) {
      if (r.public_event_kind == mfenv::PUB_EQUIP) {
        const int64_t equip = tracker.EntityAt(r.from_controller_relative, r.from_location, r.from_sequence);
        const int64_t host = tracker.EntityAt(r.to_controller_relative, r.to_location, r.to_sequence);
        if (equip && host) equips_[equip] = host;
      } else if (r.public_event_kind == mfenv::PUB_UNEQUIP) {
        equips_.erase(tracker.EntityAt(r.from_controller_relative, r.from_location, r.from_sequence));
      }
      return;
    }
    if (kind != HistoryTokenKind::STATE_DELTA) return;
    switch (subtype) {
      case HistorySubtype::CHAIN:
        Chain(r, note, code, tracker);
        return;
      case HistorySubtype::TARGET: {
        auto hit = links_.find(r.link);
        if (hit != links_.end() && resolving_ != r.link && note.subject)
          hit->second.targets.push_back({note.subject, r.to_controller_relative, r.to_location, r.to_sequence});
        return;
      }
      case HistorySubtype::MOVE:
        Move(r, note.subject);
        return;
      case HistorySubtype::POSITION:
        if (note.subject && (r.from_position & kFaceUp) && (r.to_position & kFaceDown)) {
          End(note.subject, kEndsFaceDown);
          for (auto &[number, link] : links_) {  // a target or a link's card turned face-down loses the link
            if (link.entity == note.subject) link.left = true;
            for (Target &t : link.targets)
              if (t.entity == note.subject) t.location = 0;
          }
        }
        return;
      default:
        return;
    }
  }

  // A shuffle of set cards (MSG_SHUFFLE_SET_CARD): the public tracker gives every face-down field card a new entity.
  // ``kept`` pairs a card's entity before and after at a place the shuffle did not touch; ``group_old`` and
  // ``group_new`` are the entities at the shuffled places before and after, place by place (the cards were permuted
  // among them): each keeps the status bits, side and source every card of the group had (a bit some had is no
  // longer placed). An open link that moved every card of the group (a set of several cards, then their shuffle)
  // moved the group's new entities; a link that moved some of them no longer places those.
  void Renew(const std::vector<std::pair<int64_t, int64_t>> &kept, const std::vector<int64_t> &group_old,
             const std::vector<int64_t> &group_new, const std::vector<std::array<int, 3>> &group_places) {
    std::map<int64_t, Status> cards;
    std::map<int64_t, int> owners;
    for (const auto &[before, after] : kept) {
      if (const auto hit = cards_.find(before); hit != cards_.end()) cards[after] = hit->second;
      if (const auto hit = owners_.find(before); hit != owners_.end()) owners[after] = hit->second;
    }
    Status common;
    common.bits = 0xFF;
    int owner = -2;
    for (size_t i = 0; i < group_old.size(); ++i) {
      const auto hit = cards_.find(group_old[i]);
      const Status s = hit == cards_.end() ? Status{} : hit->second;
      common.bits &= s.bits;
      common.source = i == 0 || common.source == s.source ? s.source : 0;
      common.turn = i == 0 ? s.turn : std::min(common.turn, s.turn);
      const auto o = owners_.find(group_old[i]);
      const int side = o == owners_.end() ? -1 : o->second;
      owner = owner == -2 || owner == side ? side : -1;
    }
    for (int64_t after : group_new) {
      if (!group_old.empty() && common.bits) cards[after] = common;
      if (owner >= 0) owners[after] = owner;
    }
    std::set<int64_t> renewed;
    for (const auto &[before, after] : kept) renewed.insert(before);
    renewed.insert(group_old.begin(), group_old.end());
    for (auto it = cards_.begin(); it != cards_.end();) it = renewed.count(it->first) ? cards_.erase(it) : std::next(it);
    for (auto it = owners_.begin(); it != owners_.end();) it = renewed.count(it->first) ? owners_.erase(it) : std::next(it);
    for (auto it = equips_.begin(); it != equips_.end();)
      it = renewed.count(it->first) || renewed.count(it->second) ? equips_.erase(it) : std::next(it);
    for (auto &[entity, status] : cards) cards_[entity] = status;
    for (auto &[entity, side] : owners) owners_[entity] = side;
    // an open link's targets and moved cards follow a kept card; a shuffled one is no longer placed, unless the link
    // moved the whole group alike
    std::map<int64_t, int64_t> successor(kept.begin(), kept.end());
    for (int64_t before : group_old) successor[before] = 0;
    auto follow = [&](int64_t &entity) {
      if (const auto hit = successor.find(entity); hit != successor.end()) entity = hit->second;
    };
    const std::set<int64_t> group(group_old.begin(), group_old.end());
    for (auto &[number, link] : links_) {
      follow(link.entity);
      for (Target &t : link.targets) follow(t.entity);
      std::vector<size_t> hits;
      for (size_t i = 0; i < link.moved.size(); ++i)
        if (group.count(link.moved[i].entity)) hits.push_back(i);
      bool whole = !hits.empty() && hits.size() == group.size() && group_new.size() == group.size() &&
                   group_places.size() == group.size();
      for (size_t i : hits)
        whole = whole && link.moved[i].from_location == link.moved[hits[0]].from_location &&
                link.moved[i].to_location == link.moved[hits[0]].to_location &&
                link.moved[i].to_side == link.moved[hits[0]].to_side;
      for (size_t k = 0; k < hits.size(); ++k) {
        Moved &m = link.moved[hits[k]];
        if (whole) {
          m.entity = group_new[k];
          m.to_side = group_places[k][0];
          m.to_location = group_places[k][1];
          m.to_sequence = group_places[k][2];
        } else {
          m.entity = 0;
        }
      }
      for (Moved &m : link.moved)
        if (!group.count(m.entity)) follow(m.entity);
    }
  }

  // obs:card_status_ rows (zeroed by the caller) for the viewer's card rows, whose tracked entities are ``rows``
  // (PublicTracker::RowEntities), and obs:public_effects_ rows (card ids through ``card_row``). Returns the player
  // effects beyond the table (oldest kept).
  template <class CardRow>
  int64_t Write(const std::vector<int64_t> &rows, const mfenv::PublicTracker &tracker, const CardRow &card_row,
                uint8_t *status, uint8_t *effects) const {
    std::map<int64_t, int64_t> row_of;  // entity -> card row
    for (size_t i = 0; i < rows.size(); ++i)
      if (rows[i]) row_of[rows[i]] = static_cast<int64_t>(i);
    auto ref = [&](int64_t entity) -> uint8_t {
      const auto hit = row_of.find(entity);
      return hit == row_of.end() ? 0 : Clip(hit->second + 1);
    };
    std::map<int64_t, int64_t> equipped;  // host entity -> equip cards
    for (const auto &[equip, host] : equips_) ++equipped[host];
    for (size_t i = 0; i < rows.size(); ++i) {
      const int64_t entity = rows[i];
      if (!entity) continue;
      uint8_t *row = status + i * kStatusWidth;
      if (const auto host = equips_.find(entity); host != equips_.end()) row[0] = ref(host->second);
      if (const auto count = equipped.find(entity); count != equipped.end()) row[1] = Clip(count->second);
      row[2] = LocationId(tracker.ArrivedFromEntity(entity));
      if (const auto hit = cards_.find(entity); hit != cards_.end()) {
        row[3] = hit->second.bits;
        row[4] = ref(hit->second.source);
        row[5] = Clip(turn_ - hit->second.turn, 7);
      }
    }
    for (size_t i = 0; i < effects_.size() && i < static_cast<size_t>(kEffectRows); ++i) {
      const Effect &e = effects_[i];
      const int64_t id = card_row(e.code);
      if (id <= 0) throw std::runtime_error("a public effect's card has no card row: " + std::to_string(e.code));
      uint8_t *row = effects + i * kEffectWidth;
      row[0] = 1;
      row[1] = Clip(id >> 8);
      row[2] = Clip(id & 0xFF);
      row[3] = Clip(e.desc >= 10000 ? (e.desc & 0xF) : 15);
      row[4] = static_cast<uint8_t>(e.owner + 1);  // 1 the viewer, 2 its opponent
      row[5] = Clip(e.kind);
      row[6] = 1;  // expires at the end of this turn (every v1 kind)
      row[7] = Clip(turn_ - e.turn, 7);
    }
    return std::max<int64_t>(0, static_cast<int64_t>(effects_.size()) - kEffectRows);
  }

 private:
  static constexpr int kMzone = 0x04, kSzone = 0x08, kOverlay = 0x80, kFaceUp = 0x5, kFaceDown = 0xA;

  struct Target {
    int64_t entity;
    int side, location, sequence;
  };
  struct Moved {
    int64_t entity;
    int from_location, to_side, to_location, to_sequence;
    bool control;  // a monster zone move across controllers
  };
  struct Link {
    int64_t code = 0, desc = 0, entity = 0;
    int side = 0;                                  // the activating player
    int place_side = 0, location = 0, sequence = 0;  // the link's card when it was activated
    bool negated = false;
    bool left = false;  // the link's card left its place, or was turned face-down, before the link resolved
    std::vector<Target> targets;
    std::vector<Moved> moved;
  };
  struct Status {
    uint8_t bits = 0;
    int64_t source = 0;  // the entity of the card whose link set the latest bit
    int turn = 0;        // the turn of the latest bit
  };
  struct Effect {
    int64_t code = 0, desc = 0;
    int owner = 0, kind = 0, turn = 0;
  };

  static uint8_t Clip(int64_t value, int64_t high = 255) {
    return static_cast<uint8_t>(value < 0 ? 0 : (value > high ? high : value));
  }
  static uint8_t LocationId(int location) {
    switch (location & 0x7F) {
      case 0x01: return 1;
      case 0x02: return 2;
      case 0x04: return 3;
      case 0x08: return 4;
      case 0x10: return 5;
      case 0x20: return 6;
      case 0x40: return 7;
      default: return 0;
    }
  }
  static bool OnField(int location) { return !(location & kOverlay) && (location == kMzone || location == kSzone); }

  void Chain(const mfenv::HistoryTokenRecord &r, const mfenv::TrackerRowAnnotation &note, int64_t code,
             const mfenv::PublicTracker &tracker) {
    switch (r.value) {
      case mfenv::CHAIN_CHAINING: {
        // nothing resolves while a link is activated; a new chain's first link ends the last chain (a negated link
        // is reported resolving but never resolved)
        resolving_ = 0;
        if (r.link <= 1) links_.clear();
        Link &link = links_[r.link];
        link = Link{};
        link.code = code;
        link.desc = r.detail;
        link.entity = note.subject;
        link.side = r.player_relative == 1 ? 1 : 0;
        link.place_side = r.from_controller_relative;
        link.location = r.from_location;
        link.sequence = r.from_sequence;
        return;
      }
      case mfenv::CHAIN_SOLVING:
        resolving_ = r.link;
        return;
      case mfenv::CHAIN_NEGATED:
      case mfenv::CHAIN_DISABLED:
        if (auto hit = links_.find(r.link); hit != links_.end()) hit->second.negated = true;
        return;
      case mfenv::CHAIN_SOLVED: {
        if (auto hit = links_.find(r.link); hit != links_.end()) {
          if (!hit->second.negated) Apply(hit->second, tracker);
          links_.erase(hit);
        }
        resolving_ = 0;
        return;
      }
      default:
        return;
    }
  }

  void Apply(const Link &link, const mfenv::PublicTracker &tracker) {
    const bool stayed = link.entity && OnField(link.location) && !link.left &&
                        tracker.EntityAt(link.place_side, link.location, link.sequence) == link.entity;
    for (const Rule &rule : kRules) {
      if (rule.code != link.code || rule.desc != link.desc) continue;
      if (rule.stays && !stayed) continue;  // the engine skips a continuous card's effect once the card has left
      switch (rule.selector) {
        case Selector::kMoved:
          for (const Moved &m : link.moved)
            if ((m.from_location & rule.from_mask) && (m.to_location & rule.to_mask) &&
                tracker.EntityAt(m.to_side, m.to_location, m.to_sequence) == m.entity)
              Set(m.entity, rule.status, link.entity);
          break;
        case Selector::kTargets:
          for (const Target &t : link.targets) {
            if (t.location != kMzone || tracker.EntityAt(t.side, t.location, t.sequence) != t.entity) continue;
            if (rule.side == TargetSide::kOpponent && t.side == link.side) continue;
            Set(t.entity, rule.status, link.entity);
          }
          break;
        case Selector::kControlTaken:
          for (const Moved &m : link.moved) {
            const auto owner = owners_.find(m.entity);
            if (m.control && m.to_side == link.side && owner != owners_.end() && owner->second != link.side &&
                tracker.EntityAt(m.to_side, m.to_location, m.to_sequence) == m.entity)
              Set(m.entity, rule.status, link.entity);
          }
          break;
        case Selector::kSelf:
          if (stayed) Set(link.entity, rule.status, link.entity);
          break;
        case Selector::kPlayer:
          effects_.push_back({link.code, link.desc, link.side, rule.kind, turn_});
          break;
      }
    }
  }

  void Set(int64_t entity, uint8_t bit, int64_t source) {
    Status &s = cards_[entity];
    s.bits |= bit;
    s.source = source;
    s.turn = turn_;
  }

  void End(int64_t entity, uint8_t mask) {
    auto hit = cards_.find(entity);
    if (hit == cards_.end()) return;
    hit->second.bits &= static_cast<uint8_t>(~mask);
    if (!hit->second.bits) cards_.erase(hit);
  }

  void Move(const mfenv::HistoryTokenRecord &r, int64_t entity) {
    if (!entity) return;
    const int from = r.from_location, to = r.to_location;
    const bool control = from == kMzone && to == kMzone && r.from_controller_relative != r.to_controller_relative;
    if (resolving_ != 0)
      if (auto hit = links_.find(resolving_); hit != links_.end())
        hit->second.moved.push_back({entity, from & 0x7F, r.to_controller_relative, to, r.to_sequence, control});
    // a target that moves within the monster zones (its control taken, a zone change) stays the link's target there;
    // a link's card that moves has left its place
    for (auto &[number, link] : links_) {
      if (link.entity == entity) link.left = true;
      for (Target &t : link.targets)
        if (t.entity == entity) {
          if (from == kMzone && to == kMzone) {
            t.side = r.to_controller_relative;
            t.sequence = r.to_sequence;
          } else {
            t.location = 0;
          }
        }
    }
    if (!OnField(from) && OnField(to))  // a card comes to the field: the player it comes under
      owners_[entity] = r.to_controller_relative;
    if (OnField(from) && (!OnField(to) || (from & 0x7F) != (to & 0x7F))) {
      cards_.erase(entity);  // it left the field: every status ends, and its equip relations
      equips_.erase(entity);
      for (auto it = equips_.begin(); it != equips_.end();) it = it->second == entity ? equips_.erase(it) : std::next(it);
      if (!OnField(to)) owners_.erase(entity);
      return;
    }
    if (control) End(entity, kEndsControlChange);
  }

  int turn_ = 0;
  int resolving_ = 0;
  std::map<int, Link> links_;
  std::map<int64_t, Status> cards_;
  std::map<int64_t, int64_t> equips_;  // equip card entity -> host entity
  std::map<int64_t, int> owners_;      // field card entity -> the side it came to the field under
  std::vector<Effect> effects_;        // this turn's player effects, oldest first
};

}  // namespace public_effects
}  // namespace duelenv
