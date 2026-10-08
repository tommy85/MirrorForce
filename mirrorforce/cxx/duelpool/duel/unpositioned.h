// Identities without positions (search plan S0; the rule, the design notes): the rows
// of obs:unpositioned_ and the permute_hidden constraint. Per viewer, for each of its opponent's hidden locations, a
// multiset lower bound of the identities the viewer knows are among the cards there whose rows show no code. Every
// rule below keeps "the unshown cards of the location contain the multiset" true for every hidden truth behind the
// viewer's public information, and the multisets are functions of that information only (its records, and the
// opponent's public face-up hand cards), so two truths behind one public stream give one table. A contradiction (more
// copies claimed than unshown cards) throws.
//
// Hand (HandUnpositioned). A viewer that saw a card of its opponent's hand places it there through its public tracker,
// and the card row shows it. A hand shuffle takes the place away but not the knowledge. The bound is kept over the
// "shuffled group", the hand cards present at the last hand shuffle whose identity the viewer has not learnt since:
// - a shuffle adds every code the viewer placed in that hand (tracker-known, or a public face-up hand card) to the
//   multiset; the group becomes the whole hand;
// - a group card leaving with a public code X decrements X; leaving hidden (set face-down, returned to the deck
//   unrevealed) decrements every code by one, the bound that holds whichever card it was;
// - a group card whose identity the viewer learns in place decrements its code and leaves the group: the tracker
//   learns it (a confirm), or the card is public face-up in the hand (Show, after every core buffer: the opponent's
//   client is sent the public hand cards with their places);
// - cards that arrive after the shuffle are not in the group and never touch the multiset.
//
// Deck and face-down extra deck (OpponentZones; their order is never shown, so every known identity there is
// unpositioned):
// - a card entering with an identity the viewer knows adds it: the record's code, or the code the viewer had for the
//   card where it came from (a face-up field card, graveyard, banished, placed or public hand card, material);
// - a card leaving with a public code X decrements X; leaving hidden (a draw, a search to the hand) decrements every
//   code by one; a revealed draw decrements its code;
// - a confirm or excavation of one of those cards decrements its code: it is shown (its row shows the code while the
//   reveal holds), and may be one of the counted copies; the viewer keeps the shown cards by place;
// - a reveal ends as the env's reveals do (duel_env.h end_reveals): a move from or to the location, a shuffle of it, a
//   draw, a grave/deck swap, a deck reversal. A shown card that leaves by that move leaves without touching the
//   multiset (it was not among the unshown cards); every other shown card is again unshown with its known identity
//   and re-enters the multiset (reveal_returns/v1: the excavated cards Area Zero shuffles back), after the move's own
//   update. A draw forgets the shown deck cards instead (the drawn top cards are not placed here: a looser, still
//   true bound), and a grave/deck swap makes the deck the graveyard's known codes;
// - a public deck top (no reveal in the env) decrements its code and is not re-added (looser, still true);
// - moves within the location change nothing else.
// Face-up extra deck cards (pendulum) are shown and never counted.
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mfenv/history.h"
#include "mfenv/messages.h"

namespace duelenv {
namespace history {

using Counts = std::map<int64_t, int64_t>;  // code -> copies

inline void DecrementCount(Counts &counts, int64_t code) {
  const auto hit = counts.find(code);
  if (hit != counts.end() && --hit->second == 0) counts.erase(hit);
}
inline void WidenCounts(Counts &counts) {
  for (auto it = counts.begin(); it != counts.end();) it = --it->second == 0 ? counts.erase(it) : std::next(it);
}

class HandUnpositioned {
 public:
  using Hand = std::vector<std::pair<int64_t, int64_t>>;  // (tracker entity, known code or 0) in sequence order

  void SeedCurrent(const Hand &hand, const Counts &counts, const std::vector<int> &group) {
    *this = HandUnpositioned{};
    counts_ = counts;
    for (int sequence : group) {
      if (sequence < 0 || sequence >= static_cast<int>(hand.size()) || hand[sequence].second != 0 ||
          !group_.insert(hand[sequence].first).second)
        throw std::runtime_error("invalid current-root unpositioned hand group");
    }
    Check(hand);
  }

  // One record of the viewer's stream, with the opponent's hand as the tracker held it before and after the record.
  void Update(const mfenv::HistoryTokenRecord &r, const Hand &before, const Hand &after) {
    const auto kind = static_cast<mfenv::HistoryTokenKind>(r.kind);
    if (kind == mfenv::HistoryTokenKind::PUBLIC_EVENT && r.public_event_kind == mfenv::PUB_SHUFFLE_HAND &&
        r.player_relative == 1) {
      for (const auto &[entity, code] : before) {
        const int64_t known = code > 0 ? code : ShownCode(entity);
        if (known > 0) ++counts_[known];
      }
      shown_.clear();
      group_.clear();
      for (const auto &[entity, code] : after) group_.insert(entity);
      Check(after);
      return;
    }
    std::set<int64_t> now;
    for (const auto &[entity, code] : after) now.insert(entity);
    for (const auto &[entity, code] : before) {
      if (now.count(entity)) continue;
      shown_.erase(entity);
      if (!group_.erase(entity)) continue;
      const bool moved_out = kind == mfenv::HistoryTokenKind::STATE_DELTA &&
                             static_cast<mfenv::HistorySubtype>(r.subtype) == mfenv::HistorySubtype::MOVE &&
                             r.from_controller_relative == 1 && r.from_location == kHand;
      if (moved_out && r.card_code > 0) DecrementCount(counts_, r.card_code);
      else WidenCounts(counts_);
    }
    for (const auto &[entity, code] : after)
      if (code > 0 && group_.erase(entity)) DecrementCount(counts_, code);
    Check(after);
  }

  // The opponent's public (face-up) hand cards after a core buffer, (sequence, code), against the tracker's hand.
  void Show(const Hand &hand, const std::vector<std::pair<int, int64_t>> &shown) {
    for (const auto &[sequence, code] : shown) {
      if (sequence < 0 || sequence >= static_cast<int>(hand.size()))
        throw std::runtime_error("a public hand card at sequence " + std::to_string(sequence) +
                                 " of a tracked hand of " + std::to_string(hand.size()));
      const int64_t entity = hand[static_cast<size_t>(sequence)].first;
      shown_[entity] = code;
      if (group_.erase(entity)) DecrementCount(counts_, code);
    }
    Check(hand);
  }

  // The code of a public face-up hand card's entity (0 when it is not one).
  int64_t ShownCode(int64_t entity) const {
    const auto hit = shown_.find(entity);
    return hit == shown_.end() ? 0 : hit->second;
  }

  const Counts &counts() const { return counts_; }
  // Whether a hand entity is in the shuffled group: the multiset's copies are among these cards, not among cards that
  // came to the hand since (whose places the viewer also follows) -- what a reassignment of the hand must respect.
  bool InGroup(int64_t entity) const { return group_.count(entity) != 0; }

 private:
  static constexpr int kHand = 0x02;

  void Check(const Hand &hand) const {
    int64_t claimed = 0, unidentified = 0;
    for (const auto &entry : counts_) claimed += entry.second;
    for (const auto &[entity, code] : hand) unidentified += group_.count(entity) && code == 0 ? 1 : 0;
    if (claimed > unidentified || group_.size() != static_cast<size_t>(unidentified))
      throw std::runtime_error("hand identities without positions: " + std::to_string(claimed) + " copies over " +
                               std::to_string(unidentified) + " unidentified shuffled cards (group " +
                               std::to_string(group_.size()) + ")");
  }

  std::set<int64_t> group_;
  std::map<int64_t, int64_t> shown_;  // public face-up hand cards: tracker entity -> code
  Counts counts_;
};

class OpponentZones {
 public:
  void SeedCurrent(const Counts &deck, const Counts &extra, const std::map<int, int64_t> &shown_deck,
                   const std::map<int, int64_t> &shown_extra) {
    *this = OpponentZones{};
    deck_ = deck;
    extra_ = extra;
    shown_deck_ = shown_deck;
    shown_extra_ = shown_extra;
  }
  // One record of the viewer's stream. ``known_from``: the identity the viewer knows for the card at the record's
  // origin, before the record (0 when unknown); ``grave``: the codes the viewer knows of its opponent's graveyard
  // before the record (read for a grave/deck swap only).
  void Update(const mfenv::HistoryTokenRecord &r, int64_t known_from, const std::vector<int64_t> &grave) {
    const auto kind = static_cast<mfenv::HistoryTokenKind>(r.kind);
    const auto subtype = static_cast<mfenv::HistorySubtype>(r.subtype);
    if (kind == mfenv::HistoryTokenKind::STATE_DELTA && subtype == mfenv::HistorySubtype::MOVE) {
      const bool from_deck = r.from_controller_relative == 1 && r.from_location == kDeck;
      const bool to_deck = r.to_controller_relative == 1 && r.to_location == kDeck;
      const bool from_extra = r.from_controller_relative == 1 && r.from_location == kExtra &&
                              (r.from_position & kFacedown);
      const bool to_extra = r.to_controller_relative == 1 && r.to_location == kExtra && (r.to_position & kFacedown);
      // a shown card leaving its location was not among the unshown cards: the multiset keeps its copies
      const bool deck_shown = from_deck && shown_deck_.erase(static_cast<int>(r.from_sequence));
      const bool extra_shown = from_extra && shown_extra_.erase(static_cast<int>(r.from_sequence));
      if (from_deck != to_deck) {
        if (from_deck) {
          if (!deck_shown) Leave(deck_, r.card_code);
        } else {
          Enter(deck_, known_from);
        }
      }
      if (from_extra != to_extra) {
        if (from_extra) {
          if (!extra_shown) Leave(extra_, r.card_code);
        } else {
          Enter(extra_, known_from);
        }
      }
      // the env ends a location's reveals when a card leaves it or arrives in it (any position for the extra deck)
      const bool any_from_extra = r.from_controller_relative == 1 && r.from_location == kExtra;
      const bool any_to_extra = r.to_controller_relative == 1 && r.to_location == kExtra;
      if (from_deck || to_deck) Return(shown_deck_, deck_);
      if (any_from_extra || any_to_extra) Return(shown_extra_, extra_);
      return;
    }
    if (kind == mfenv::HistoryTokenKind::STATE_DELTA && subtype == mfenv::HistorySubtype::DRAW) {
      if (r.player_relative != 1) return;
      for (int64_t i = 0; i < r.count; ++i)
        Leave(deck_, i < static_cast<int64_t>(draw_reveals_.size()) ? draw_reveals_[static_cast<size_t>(i)] : 0);
      draw_reveals_.clear();
      shown_deck_.clear();
      return;
    }
    if (kind != mfenv::HistoryTokenKind::PUBLIC_EVENT) return;
    switch (r.public_event_kind) {
      case mfenv::PUB_DRAW_REVEAL:
        if (r.player_relative == 1) draw_reveals_.push_back(r.card_code);
        break;
      case mfenv::PUB_CONFIRM:
        if (r.from_controller_relative != 1 || r.card_code <= 0) break;
        if (r.from_location == kDeck) Show(shown_deck_, deck_, static_cast<int>(r.from_sequence), r.card_code);
        if (r.from_location == kExtra) Show(shown_extra_, extra_, static_cast<int>(r.from_sequence), r.card_code);
        break;
      case mfenv::PUB_DECK_TOP:
        if (r.from_controller_relative != 1 || r.card_code <= 0) break;
        if (r.from_location == kDeck) DecrementCount(deck_, r.card_code);
        break;
      case mfenv::PUB_SHUFFLE_DECK:
        if (r.player_relative == 1) Return(shown_deck_, deck_);
        break;
      case mfenv::PUB_SHUFFLE_EXTRA:
        if (r.player_relative == 1) Return(shown_extra_, extra_);
        break;
      case mfenv::PUB_REVERSE_DECK:
        Return(shown_deck_, deck_);
        break;
      case mfenv::PUB_SWAP_GRAVE_DECK:
        if (r.player_relative != 1) break;
        shown_deck_.clear();
        deck_.clear();
        for (int64_t code : grave)
          if (code > 0) ++deck_[code];
        break;
      default:
        break;
    }
  }

  const Counts &deck() const { return deck_; }
  const Counts &extra() const { return extra_; }

 private:
  static constexpr int kDeck = 0x01, kExtra = 0x40, kFacedown = 0x0A;

  static void Enter(Counts &counts, int64_t code) {
    if (code > 0) ++counts[code];
  }
  static void Leave(Counts &counts, int64_t code) {
    if (code > 0) DecrementCount(counts, code);
    else WidenCounts(counts);
  }
  // A card of the location shown at ``sequence``: out of the unshown cards while the reveal holds.
  static void Show(std::map<int, int64_t> &shown, Counts &counts, int sequence, int64_t code) {
    if (shown.emplace(sequence, code).second) DecrementCount(counts, code);
  }
  // The location's reveals end: its shown cards are unshown again, with the identities the viewer saw.
  static void Return(std::map<int, int64_t> &shown, Counts &counts) {
    for (const auto &[sequence, code] : shown) ++counts[code];
    shown.clear();
  }

  Counts deck_, extra_;
  std::map<int, int64_t> shown_deck_, shown_extra_;  // shown cards of the location: sequence -> code
  std::vector<int64_t> draw_reveals_;  // revealed draws of the opponent awaiting its draw record
};

}  // namespace history
}  // namespace duelenv
