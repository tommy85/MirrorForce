// Pure public-belief world assembly shared by the engine-view and engine-free client adapters.
// Inputs have already been filtered to what the observer may see. No engine pointer, true opponent code, query,
// labels or history mutation is accepted here. This preserves the established SearchDuel world field/order law.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace duelenv::public_belief {

struct World {
  std::vector<uint32_t> hand, deck, extra;
  std::vector<int> hand_group;
  std::vector<std::array<uint32_t, 4>> facedown;
  bool decklist_public = false;
  std::map<uint32_t, int64_t> pool_main, pool_extra;
  std::vector<std::pair<uint32_t, bool>> public_owned;
  std::map<std::pair<int, int64_t>, int64_t> unpositioned;
  std::vector<uint32_t> own_deck;
  std::vector<std::pair<int, uint32_t>> own_deck_fixed;
  std::map<uint32_t, uint32_t> types;
};

struct CardView {
  uint32_t location = 0, sequence = 0, shown = 0, position = 0;
  bool extra_kind = false;  // printed type of a shown card, or its public arrival from the extra deck
};

struct Input {
  std::vector<CardView> opponent;
  std::vector<std::pair<uint32_t, bool>> public_owned;
  std::vector<int> hand_group;
  std::map<std::pair<int, int64_t>, int64_t> unpositioned;
  std::vector<uint32_t> own_deck;
  std::vector<std::pair<int, uint32_t>> own_deck_fixed;
};

template <typename CardType>
World Build(Input input, bool declared, const std::vector<uint32_t> &main,
            const std::vector<uint32_t> &extra, CardType card_type) {
  World w;
  w.decklist_public = declared;
  // Input order is the established zone/sequence order; neither hidden hand places nor field slots are sorted by
  // their true identity. Only the owner's deck MULTISET is sorted below, never exposed as its true draw order.
  for (const CardView &card : input.opponent) {
    if (card.location == 0x02) w.hand.push_back(card.shown);
    else if (card.location == 0x01) w.deck.push_back(card.shown);
    else if (card.location == 0x40 && (card.position & 0x0a)) w.extra.push_back(card.shown);
    else if ((card.location == 0x04 || card.location == 0x08 || card.location == 0x20) && (card.position & 0x0a))
      w.facedown.push_back({card.location, card.sequence, card.shown, card.extra_kind ? 1u : 0u});
  }
  w.public_owned = std::move(input.public_owned);
  if (declared) {
    for (uint32_t code : main) ++w.pool_main[code];
    for (uint32_t code : extra) ++w.pool_extra[code];
    for (const auto &[code, extra_kind] : w.public_owned) {
      auto &pool = extra_kind ? w.pool_extra : w.pool_main;
      if (--pool[code] < 0)
        throw std::runtime_error("public_world: the declared decklist holds fewer " + std::to_string(code) +
                                 " than the opponent has shown");
    }
    for (auto *pool : {&w.pool_main, &w.pool_extra})
      for (auto it = pool->begin(); it != pool->end();) it = it->second == 0 ? pool->erase(it) : std::next(it);
    int64_t main_slots = 0, extra_slots = 0, main_pool = 0, extra_pool = 0;
    for (uint32_t code : w.hand) main_slots += code == 0;
    for (uint32_t code : w.deck) main_slots += code == 0;
    for (const auto &slot : w.facedown) (slot[3] ? extra_slots : main_slots) += slot[2] == 0;
    for (uint32_t code : w.extra) extra_slots += code == 0;
    for (const auto &[code, n] : w.pool_main) main_pool += n;
    for (const auto &[code, n] : w.pool_extra) extra_pool += n;
    if (main_pool != main_slots || extra_pool != extra_slots)
      throw std::runtime_error("public_world: the pool (" + std::to_string(main_pool) + " main, " +
                               std::to_string(extra_pool) + " extra) does not fill the hidden places (" +
                               std::to_string(main_slots) + ", " + std::to_string(extra_slots) + ")");
  }
  w.unpositioned = std::move(input.unpositioned);
  w.hand_group = std::move(input.hand_group);
  w.own_deck = std::move(input.own_deck);
  std::sort(w.own_deck.begin(), w.own_deck.end());
  w.own_deck_fixed = std::move(input.own_deck_fixed);
  for (const auto *pool : {&w.pool_main, &w.pool_extra})
    for (const auto &[code, n] : *pool) w.types[code] = card_type(code);
  return w;
}

}  // namespace duelenv::public_belief
