// Test the actual const common-public projection, including private entities and a continuing public chain.
#include <cassert>
#include <iostream>
#include <sstream>
#include "duel/history_obs.h"

using namespace duelenv;
using public_effects::SeedPlace;
constexpr int kMaxx = 23434538, kMultirole = 24010609;

history::CurrentRootSeed seed(int private_code, bool change_entity_allocation = false) {
  history::CurrentRootSeed s;
  auto add = [&](int side, int loc, int seq, int code) {
    mfenv::TrackerSeedCard c;
    c.side = side; c.location = loc; c.sequence = seq; c.position = loc == 2 ? 8 : 1; c.code = code;
    s.tracker.cards.push_back(c);
  };
  if (change_entity_allocation) { add(0, 2, 0, private_code); add(0, 2, 1, 99); }
  add(0, 4, 2, 30);
  add(0, 8, 0, kMultirole);
  add(0, 16, 0, kMaxx);
  if (!change_entity_allocation) { add(0, 2, 0, private_code); add(0, 2, 1, 99); }
  s.status.cards.push_back({{0, 4, 2}, {0, 2, 0}, public_effects::kNegatedUntilEnd, 7});
  s.status.equips.push_back({{0, 8, 0}, {0, 4, 2}});
  s.status.field_origins.push_back({{0, 4, 2}, 1});
  s.status.field_origins.push_back({{0, 8, 0}, 0});
  s.status.effects.push_back({kMaxx, kMaxx * 16LL, 1, 1, 7});
  s.status.effects.push_back({kMultirole, kMultirole * 16LL, 0, 2, 7});
  return s;
}

std::string effects(const history::PublicRootSeed &s) {
  std::ostringstream out;
  auto place = [&](const SeedPlace &p) { for (int x : p) out << x << ','; };
  out << s.turn << ':' << s.turn_player << ':' << s.phase << ';';
  for (const auto &c : s.status.cards) { place(c.place); place(c.source); out << int(c.bits) << ',' << c.turn << ';'; }
  for (const auto &[a, b] : s.status.equips) { place(a); place(b); }
  for (const auto &[p, side] : s.status.field_origins) { place(p); out << side << ';'; }
  for (const auto &e : s.status.effects) for (auto x : e) out << x << ',';
  return out.str();
}

template<class F> void rejected(F &&f, const std::string &reason) {
  bool failed = false;
  try { f(); } catch (const public_effects::PublicRootSeedError &e) {
    failed = true;
    assert(std::string(e.what()).find("current_public_root_seed/" + reason + ":") == 0);
    assert(std::string(e.what()).find("123456789") == std::string::npos);
  }
  assert(failed);
}

int main() {
  mfenv::SemanticTable table({{30, 1}, {99, 2}, {123456789, 3}, {987654321, 4}, {kMaxx, 5}, {kMultirole, 6}}, "public-seed-test");
  const std::vector<SeedPlace> public_places{{0, 4, 2}, {0, 8, 0}, {0, 16, 0}};
  history::History first(&table), second(&table);
  first.InitializeCurrentRoot(0, 7, 1, 4, seed(123456789));
  second.InitializeCurrentRoot(0, 7, 1, 4, seed(987654321, true));
  const auto a = first.CurrentPublicSeed(0, public_places), b = second.CurrentPublicSeed(0, public_places);
  assert(effects(a) == effects(b));  // different private identities and entity allocation, identical common-public DTO
  assert(a.status.cards[0].source[0] == -1 && a.status.cards[0].bits == public_effects::kNegatedUntilEnd);
  assert(a.status.effects.size() == 2 && a.status.effects[0][0] == kMaxx && a.status.effects[1][3] == 2);
  assert((first.CurrentTurnRows(0) == std::array<int64_t, 2>{7, 0}));
  assert(first.FactCount(0) == 0 && first.ClosedTurnRows(0)[0][0] == 0);
  assert(effects(first.CurrentPublicSeed(0, public_places)) == effects(a));
  rejected([&] { first.CurrentPublicSeed(0, {{0, 2, 0}}); }, "private_anchor");
  rejected([&] { first.CurrentPublicSeed(0, {{0, 8, 0}, {0, 16, 0}}); }, "field_status_reference");
  auto unsupported = seed(123456789);
  unsupported.status.effects[0][1] = 0;  // must not silently erase an unexplained active player effect
  first.InitializeCurrentRoot(0, 7, 1, 4, unsupported);
  rejected([&] { first.CurrentPublicSeed(0, public_places); }, "unsupported_player_effect");
  first.InitializeCurrentRoot(0, 7, 1, 4, seed(123456789));

  // The regular live observer retains exact CHAINING metadata in its current public entries. A private choice
  // made afterwards is neither replayed nor included in the getter, and calling it consumes no delivery state.
  history::History streamed(&table);
  streamed.InitializeCurrentRoot(0, 7, 1, 4, seed(123456789));
  std::vector<uint8_t> chaining{70};
  auto u32 = [&](uint32_t value) { for (int i = 0; i < 4; ++i) chaining.push_back((value >> (8 * i)) & 255); };
  u32(kMultirole); u32(0 | (8 << 8) | (1 << 24));
  chaining.insert(chaining.end(), {0, 8, 0});
  u32(kMultirole * 16 + 1); chaining.push_back(1);
  streamed.ConsumeBuffer(chaining.data(), chaining.size());
  const auto public_before = streamed.CurrentPublicSeed(0, public_places);
  history::OwnChoice private_choice;
  private_choice.has_place = true; private_choice.location = 2; private_choice.sequence = 0;
  private_choice.card_row = 3; private_choice.msg = 15;
  streamed.ConsumeOwnChoice(0, private_choice);
  const auto rows_before = streamed.CurrentTurnRows(0);
  const auto facts_before = streamed.FactCount(0);
  const auto public_after = streamed.CurrentPublicSeed(0, public_places);
  assert(effects(public_before) == effects(public_after));
  assert(public_after.chain[0].first.code == kMultirole && public_after.chain[0].first.desc == kMultirole * 16 + 1);
  assert(streamed.CurrentTurnRows(0) == rows_before && streamed.FactCount(0) == facts_before);

  // A private draw within Multirole's link cannot match its GRAVE->SZONE rule. It is proven irrelevant, not copied.
  auto s = seed(123456789);
  public_effects::SeedLink link;
  link.number = 1; link.code = kMultirole; link.desc = kMultirole * 16LL + 1;
  link.origin = {0, 8, 0}; link.source = {0, 8, 0};
  link.moved.push_back({{0, 2, 0}, 1, false, {0, 2, 0}});
  s.chain.emplace_back(link, history::kResolving);
  s.status.links.push_back(link);
  s.status.resolving = 1;
  s.guard_chain = {1, 12, 12};
  s.carry.chain_link = s.carry.settlement_link = 1;
  first.InitializeCurrentRoot(0, 7, 1, 4, s);
  auto current = first.CurrentPublicSeed(0, public_places);
  assert(current.chain.size() == 1 && current.chain[0].first.moved.empty());
  assert(current.guard_chain.resolution == 12 && current.carry.settlement_link == 1);

  // A relevant earlier GRAVE->SZONE move whose entity is now private cannot be dropped on the assumption it stays
  // private: it can return to that destination before resolution. Refuse with a typed reason and no private code.
  s.status.links[0].moved[0] = {{0, 2, 0}, 16, false, {0, 8, 1}};
  first.InitializeCurrentRoot(0, 7, 1, 4, s);
  rejected([&] { first.CurrentPublicSeed(0, public_places); }, "private_chain_move");
  auto negated = s;
  negated.status.links[0].negated = true;
  negated.chain[0].second = history::kNegated;
  first.InitializeCurrentRoot(0, 7, 1, 4, negated);
  assert(first.CurrentPublicSeed(0, public_places).chain[0].first.moved.empty());

  // With a public current anchor, preserve BOTH current place and original destination, even when different.
  s.status.links[0].moved[0] = {{0, 4, 2}, 16, false, {0, 8, 1}};
  first.InitializeCurrentRoot(0, 7, 1, 4, s);
  current = first.CurrentPublicSeed(0, public_places);
  assert((current.chain[0].first.moved[0].place == SeedPlace{0, 4, 2}));
  assert((current.chain[0].first.moved[0].destination == SeedPlace{0, 8, 1}));
  mfenv::PublicTracker tracker;
  tracker.SeedCurrent(7, s.tracker);
  public_effects::Tracker projected;
  projected.SeedCurrent(7, current.status, tracker);
  mfenv::HistoryTokenRecord move;
  move.kind = static_cast<int>(mfenv::HistoryTokenKind::STATE_DELTA);
  move.subtype = static_cast<int>(mfenv::HistorySubtype::MOVE);
  move.card_code = 30;
  move.from_controller_relative = move.to_controller_relative = 0;
  move.from_location = 4; move.from_sequence = 2; move.from_position = 1;
  move.to_location = 8; move.to_sequence = 1; move.to_position = 8; move.turn = 7;
  auto note = tracker.Consume(move);
  projected.Update(move, note, tracker, 30);
  mfenv::HistoryTokenRecord resolved;
  resolved.kind = static_cast<int>(mfenv::HistoryTokenKind::STATE_DELTA);
  resolved.subtype = static_cast<int>(mfenv::HistorySubtype::CHAIN);
  resolved.value = mfenv::CHAIN_SOLVED; resolved.link = 1;
  projected.Update(resolved, {}, tracker, 0);
  public_effects::PublicPlaces after;
  for (const SeedPlace p : {SeedPlace{0,8,0}, SeedPlace{0,8,1}, SeedPlace{0,16,0}})
    after[tracker.EntityAt(p[0],p[1],p[2])] = p;
  auto final = projected.CurrentPublicSeed(tracker, after, {});
  const auto applied = std::find_if(final.cards.begin(), final.cards.end(), [](const auto &c) { return c.place == SeedPlace{0,8,1}; });
  assert(applied != final.cards.end() && (applied->bits & public_effects::kBanishOnLeave));
  std::cout << "common-public current-root projection checks passed\n";
}
