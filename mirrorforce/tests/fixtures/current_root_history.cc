// Small engine-free test of the real C++ history/tracker implementation. Not a native duel build or game.
#include <cassert>
#include <iostream>
#include "duel/history_obs.h"

using namespace duelenv;

struct Output {
  std::vector<uint8_t> events = std::vector<uint8_t>(4 * history::kEventWidth);
  std::vector<uint8_t> refs = std::vector<uint8_t>(4 * history::kRefWidth);
  std::array<uint8_t, history::kChainRows * history::kChainWidth> chain{};
  std::array<uint8_t, history::kActivationRows * history::kActivationWidth> activations{};
  std::array<uint8_t, 2 * history::kLedgerWidth> ledger{};
  std::array<uint8_t, history::kHintRows * history::kHintWidth> hints{};
  std::array<uint8_t, 6 * history::kCardTurnWidth> cards{};
  std::array<uint8_t, 2 * 2 * history::kEventWidth> chunks{};
  std::array<uint8_t, 2 * 2 * history::kRefWidth> chunk_refs{};
  std::array<int32_t, 2 * history::kChunkMetaWidth> chunk_meta{};
  std::array<uint8_t, 2 * 4 * history::kEventWidth> closed{};
  std::array<uint8_t, 2 * 4 * history::kRefWidth> closed_refs{};
  std::array<int32_t, 2 * history::kClosedMetaWidth> closed_meta{};
  std::array<uint8_t, 6 * public_effects::kStatusWidth> status{};
  std::array<uint8_t, public_effects::kEffectRows * public_effects::kEffectWidth> effects{};
};

int main() {
  mfenv::SemanticTable table({{10, 1}, {20, 2}, {30, 3}, {23434538, 4}}, "current-root-test");
  history::CurrentRootSeed seed;
  auto add = [&](int side, int location, int sequence, int position, int code) {
    mfenv::TrackerSeedCard card;
    card.side = side;
    card.location = location;
    card.sequence = sequence;
    card.position = position;
    card.code = code;
    seed.tracker.cards.push_back(card);
  };
  add(0, 2, 0, 8, 20);
  add(0, 2, 1, 8, 10);  // own hand order is not sorted
  add(1, 2, 0, 8, 0);
  add(0, 4, 2, 1, 30);  // field holes at 0 and 1
  seed.tracker.cards.back().counters[1] = 3;
  add(0, 4 | 128, 2, 0, 10);
  add(0, 8, 0, 1, 20);
  seed.tracker.counts[0][0] = 2;
  seed.tracker.hints[1][23434538 * 16LL] = 1;
  seed.hand[10] = 1;
  seed.hand_group = {0};
  seed.status.cards.push_back({{0, 4, 2}, {0, 8, 0}, public_effects::kCannotActivate, 7});
  seed.status.equips.push_back({{0, 8, 0}, {0, 4, 2}});
  seed.status.effects.push_back({23434538, 0, 1, 1, 7});
  public_effects::SeedLink link;
  link.number = 1;
  link.side = 1;
  link.code = 23434538;
  link.origin = {1, 2, 0};
  link.source = {-1, 0, 0};
  seed.chain.emplace_back(link, history::kPending);
  seed.status.links.push_back(link);
  seed.guard_chain.links = 1;
  seed.guard_chain.solving = 12;
  seed.carry.chain_link = 1;
  mfenv::TrackerView view;
  for (const auto &card : seed.tracker.cards) {
    mfenv::TrackerToken t;
    t.kind = card.location & 128 ? mfenv::TrackerToken::OVERLAY_MATERIAL : mfenv::TrackerToken::CARD;
    t.side = card.side;
    t.location = card.location & 127;
    t.sequence = card.sequence;
    t.overlay_index = card.position;
    t.code = card.code;
    view.tokens.push_back(t);
    if (!(card.location & 128)) ++view.zone_counts[{card.side, card.location}];
  }
  history::History h(&table);
  h.SetLaw({4, 2, 64, 2});
  h.InitializeCurrentRoot(0, 7, 1, 4, seed);
  assert((h.CurrentTurnRows(0) == std::array<int64_t, 2>{7, 0}));
  assert(h.ClosedTurnRows(0)[0][0] == 0);
  assert(h.Chain().links == 1 && h.Chain().solving == 12 && h.Chain().resolution == 0);
  auto write = [&](history::History &observer, int decision) {
    Output o;
    observer.Write(0, decision, view, o.events.data(), o.refs.data(), o.chain.data(), o.activations.data(),
                   o.ledger.data(), o.hints.data(), o.cards.data(), 6, o.chunks.data(), o.chunk_refs.data(),
                   o.chunk_meta.data(), o.closed.data(), o.closed_refs.data(), o.closed_meta.data());
    observer.WriteStatus(0, view, o.status.data(), 6, o.effects.data());
    return o;
  };
  auto initial = write(h, 0);
  assert(initial.chain[0] == 1 && initial.chain[7] == history::kPending);
  assert(initial.ledger[0] == 2 && initial.cards[3 * history::kCardTurnWidth + 4] == 3);
  assert(initial.status[3 * public_effects::kStatusWidth + 3] == public_effects::kCannotActivate);
  assert(initial.effects[0] == 1 && initial.hints[0] == 1);
  assert(initial.chunk_meta[0] == 0 && initial.closed_meta[0] == 0);
  assert(std::all_of(initial.events.begin(), initial.events.end(), [](uint8_t b) { return b == 0; }));
  history::History branch = h;
  const uint8_t solving[]{72, 1}, solved[]{73, 1}, end[]{74}, next[]{40, 1};
  branch.ConsumeBuffer(solving, sizeof(solving));
  assert(branch.Chain().resolution == 13);
  assert(write(branch, 1).chain[7] == history::kResolving);
  assert(write(h, 0).chain[7] == history::kPending);  // isolated branch
  branch.ConsumeBuffer(solved, sizeof(solved));
  assert(write(branch, 2).chain[0] == 0);
  branch.ConsumeBuffer(end, sizeof(end));
  assert(branch.Chain().links == 0 && branch.Chain().resolution == 0);
  // Events after the root use the seeded current turn; the next turn closes only those new rows.
  for (int i = 0; i < 7; ++i) {
    const uint8_t lp[]{94, 0, 0x40, 0x1f, 0, 0};
    branch.ConsumeBuffer(lp, sizeof(lp));
  }
  assert(branch.CurrentTurnRows(0)[0] == 7 && branch.CurrentTurnRows(0)[1] == 7);
  assert(write(branch, 3).chunk_meta[0] == 1);
  branch.ConsumeBuffer(next, sizeof(next));
  assert(branch.CurrentTurnRows(0)[0] == 8);
  assert(branch.ClosedTurnRows(0)[0][0] == 7 && branch.ClosedTurnRows(0)[0][1] == 7);
  auto later = write(branch, 4);
  assert(later.ledger[0] == 0 && later.effects[0] == 0);
  assert(later.status[3 * public_effects::kStatusWidth + 3] == 0);
  std::cout << "current-root history/chain/status/chunk/clone checks passed\n";
}
