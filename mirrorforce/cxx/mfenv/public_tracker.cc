#include "public_tracker.h"

#include <algorithm>
#include <string>

#include "constants.h"
#include "duel_error.h"

namespace mfenv {
namespace {

constexpr int64_t kCardQuestion = 38723936;
constexpr int64_t kDescAdd = 6, kDescRemove = 7;  /* CHINT_DESC_* and PHINT_DESC_* */
enum Arrival { ARRIVAL_UNKNOWN = 0, ARRIVAL_NORMAL, ARRIVAL_SPECIAL, ARRIVAL_FLIP, ARRIVAL_SET, ARRIVAL_MOVED };
enum Link { LINK_NONE = 0, LINK_EXACT, LINK_GROUP, LINK_ABSENT };
constexpr int kOnField = LOCATION_MZONE | LOCATION_SZONE;

bool IsList(int location) {
  return location == LOCATION_HAND || location == LOCATION_GRAVE || location == LOCATION_REMOVED;
}
bool IsFixed(int location) { return location == LOCATION_MZONE || location == LOCATION_SZONE; }

void Fail(const std::string& what) { throw DuelError("public tracker: " + what); }

}  // namespace

void PublicTracker::SeedCurrent(int turn, const TrackerSeed& seed) {
  PublicTracker fresh;
  fresh.turn_ = turn;
  fresh.counts_ = seed.counts;
  fresh.hints_ = seed.hints;
  std::set<std::tuple<int, int, int, int>> places;
  for (const auto& card : seed.cards) {
    const int position = card.location & LOCATION_OVERLAY ? card.position : 0;
    if (card.side < 0 || card.side > 1 || card.sequence < 0 || card.position < 0 || card.code < 0 ||
        card.arrival_turn < -1 || card.arrival_turn > turn ||
        !places.emplace(card.side, card.location, card.sequence, position).second)
      Fail("invalid current-root card seed");
    if (!(IsList(card.location) || IsFixed(card.location) || (card.location & LOCATION_OVERLAY))) continue;
    const int64_t entity = fresh.New();
    fresh.Put(entity, card.side, card.location, card.sequence, card.position);
    fresh.Learn(entity, card.code);
    if (!(card.location & LOCATION_OVERLAY) && (card.position & POS_FACEDOWN)) fresh.facedown_.insert(entity);
    auto& info = fresh.Info(entity);
    info.arrival_turn = card.arrival_turn;
    info.arrival_kind = card.arrival_kind;
    info.arrived_from = card.arrived_from;
    info.activations = card.activations;
    info.attacks = card.attacks;
    info.counters = card.counters;
    info.desc_hints = card.desc_hints;
    info.hint_kind = card.hint_kind;
    info.hint_value = card.hint_value;
  }
  *this = std::move(fresh);
}

int64_t PublicTracker::At(int side, int location, int sequence, int position) const {
  if (location & LOCATION_OVERLAY) {
    const auto hit = overlays_.find(std::make_tuple(side, location & 0x7F, sequence));
    if (hit == overlays_.end() || position < 0 || position >= static_cast<int>(hit->second.size())) return 0;
    return hit->second[static_cast<size_t>(position)];
  }
  if (IsList(location)) {
    const auto hit = lists_.find({side, location});
    if (hit == lists_.end() || sequence < 0 || sequence >= static_cast<int>(hit->second.size())) return 0;
    return hit->second[static_cast<size_t>(sequence)];
  }
  if (IsFixed(location)) {
    const auto hit = fixed_.find({side, location});
    if (hit == fixed_.end()) return 0;
    const auto slot = hit->second.find(sequence);
    return slot == hit->second.end() ? 0 : slot->second;
  }
  return 0;
}

int64_t PublicTracker::From(const HistoryTokenRecord& r) const {
  return At(r.from_controller_relative, r.from_location, r.from_sequence, r.from_position);
}

int64_t PublicTracker::To(const HistoryTokenRecord& r) const {
  return At(r.to_controller_relative, r.to_location, r.to_sequence, r.to_position);
}

int64_t PublicTracker::Take(int side, int location, int sequence, int position) {
  if (location & LOCATION_OVERLAY) {
    auto hit = overlays_.find(std::make_tuple(side, location & 0x7F, sequence));
    if (hit == overlays_.end() || position < 0 || position >= static_cast<int>(hit->second.size())) return 0;
    const int64_t entity = hit->second[static_cast<size_t>(position)];
    hit->second.erase(hit->second.begin() + position);
    return entity;
  }
  if (IsList(location)) {
    auto hit = lists_.find({side, location});
    if (hit == lists_.end() || sequence < 0 || sequence >= static_cast<int>(hit->second.size())) return 0;
    const int64_t entity = hit->second[static_cast<size_t>(sequence)];
    hit->second.erase(hit->second.begin() + sequence);
    return entity;
  }
  if (IsFixed(location)) {
    auto hit = fixed_.find({side, location});
    if (hit == fixed_.end()) return 0;
    const auto slot = hit->second.find(sequence);
    if (slot == hit->second.end()) return 0;
    const int64_t entity = slot->second;
    hit->second.erase(slot);
    return entity;
  }
  return 0;
}

void PublicTracker::Put(int64_t entity, int side, int location, int sequence, int position) {
  if (location & LOCATION_OVERLAY) {
    auto& materials = overlays_[std::make_tuple(side, location & 0x7F, sequence)];
    const size_t at = static_cast<size_t>(std::min<int64_t>(std::max(0, position),
                                                            static_cast<int64_t>(materials.size())));
    materials.insert(materials.begin() + static_cast<std::ptrdiff_t>(at), entity);
  } else if (IsList(location)) {
    auto& cards = lists_[{side, location}];
    /* A place the tracker never saw filled: keep the list aligned with anonymous entities . */
    while (static_cast<int>(cards.size()) < sequence) cards.push_back(New());
    cards.insert(cards.begin() + std::min<int64_t>(std::max(0, sequence), static_cast<int64_t>(cards.size())),
                 entity);
  } else if (IsFixed(location)) {
    fixed_[{side, location}][sequence] = entity;
  } else {
    codes_.erase(entity);
    facedown_.erase(entity);
  }
}

int64_t PublicTracker::KnownCode(int side, int location, int sequence) const {
  const int64_t entity = At(side, location, sequence, 0);
  if (!entity) return 0;
  const auto hit = codes_.find(entity);
  return hit == codes_.end() ? 0 : hit->second;
}

int64_t PublicTracker::CodeOf(int64_t entity) const {
  const auto hit = codes_.find(entity);
  return hit == codes_.end() ? 0 : hit->second;
}

int PublicTracker::ArrivedFrom(int side, int location, int sequence) const {
  const int64_t entity = At(side, location, sequence, 0);
  if (!entity) return 0;
  const auto hit = info_.find(entity);
  return hit == info_.end() ? 0 : hit->second.arrived_from;
}

int PublicTracker::ArrivedFromEntity(int64_t entity) const {
  const auto hit = info_.find(entity);
  return hit == info_.end() ? 0 : hit->second.arrived_from;
}

std::vector<int64_t> PublicTracker::RowEntities(const TrackerView& view) const {
  std::vector<int64_t> out(view.tokens.size(), 0);
  for (size_t i = 0; i < view.tokens.size(); ++i) {
    const TrackerToken& t = view.tokens[i];
    if (t.kind == TrackerToken::OVERLAY_MATERIAL)
      out[i] = At(t.side, LOCATION_OVERLAY | t.location, t.sequence, t.overlay_index);
    else if (t.kind == TrackerToken::CARD &&
             (IsFixed(t.location) || t.location == LOCATION_GRAVE || t.location == LOCATION_REMOVED ||
              (t.location == LOCATION_HAND && t.side == 0)))
      out[i] = At(t.side, t.location, t.sequence, 0);
  }
  return out;
}

std::vector<std::pair<int64_t, int64_t>> PublicTracker::ListEntities(int side, int location) const {
  std::vector<std::pair<int64_t, int64_t>> out;
  const auto hit = lists_.find({side, location});
  if (hit == lists_.end()) return out;
  out.reserve(hit->second.size());
  for (int64_t entity : hit->second) {
    const auto code = codes_.find(entity);
    out.emplace_back(entity, code == codes_.end() ? 0 : code->second);
  }
  return out;
}

void PublicTracker::Learn(int64_t entity, int64_t code) {
  if (entity && code) codes_[entity] = code;
}

void PublicTracker::Forget(int64_t entity) {
  codes_.erase(entity);
  facedown_.erase(entity);
  info_.erase(entity);
}

void PublicTracker::Swap(int side_a, int location_a, int sequence_a, int side_b, int location_b, int sequence_b) {
  if (!IsFixed(location_a) || !IsFixed(location_b)) Fail("a swap names a place outside the fixed field zones");
  auto& zone_a = fixed_[{side_a, location_a}];
  auto& zone_b = fixed_[{side_b, location_b}];
  int64_t a = 0, b = 0;
  if (auto hit = zone_a.find(sequence_a); hit != zone_a.end()) { a = hit->second; zone_a.erase(hit); }
  if (auto hit = zone_b.find(sequence_b); hit != zone_b.end()) { b = hit->second; zone_b.erase(hit); }
  if (b) zone_a[sequence_a] = b;
  if (a) zone_b[sequence_b] = a;
  const auto key_a = std::make_tuple(side_a, location_a, sequence_a);
  const auto key_b = std::make_tuple(side_b, location_b, sequence_b);
  std::vector<int64_t> materials_a, materials_b;
  bool has_a = false, has_b = false;
  if (auto hit = overlays_.find(key_a); hit != overlays_.end()) {
    materials_a = std::move(hit->second); has_a = true; overlays_.erase(hit);
  }
  if (auto hit = overlays_.find(key_b); hit != overlays_.end()) {
    materials_b = std::move(hit->second); has_b = true; overlays_.erase(hit);
  }
  if (has_a) overlays_[key_b] = std::move(materials_a);
  if (has_b) overlays_[key_a] = std::move(materials_b);
}

void PublicTracker::NewTurn(int64_t turn) {
  turn_ = turn;
  counts_ = {};
  for (auto& entry : info_) entry.second.activations = entry.second.attacks = 0;
  for (auto& entry : activations_) entry.second[0] = entry.second[1] = 0;
}

void PublicTracker::Count(int side, int column) {
  if (side != 0 && side != 1) Fail("a summon names no controller");
  ++counts_[static_cast<size_t>(side)][static_cast<size_t>(column)];
}

TrackerRowAnnotation PublicTracker::ConsumeOwnChoice(const HistoryTokenRecord& record, const TrackerToken* token) {
  TrackerRowAnnotation out;
  if (token == nullptr) {
    out.subject = record.from_location ? From(record) : To(record);
    return out;
  }
  if (token->kind == TrackerToken::OVERLAY_MATERIAL) {
    out.subject = At(token->side, LOCATION_OVERLAY | token->location, token->sequence, token->overlay_index);
  } else if (token->side == 1 &&
             (token->location == LOCATION_HAND || token->location == LOCATION_DECK ||
              token->location == LOCATION_EXTRA)) {
    out.subject = 0;  /* canonical, not real, coordinates */
  } else {
    out.subject = At(token->side, token->location, token->sequence, 0);
  }
  return out;
}

TrackerRowAnnotation PublicTracker::Consume(const HistoryTokenRecord& r) {
  TrackerRowAnnotation out;
  const auto kind = static_cast<HistoryTokenKind>(r.kind);
  const auto subtype = static_cast<HistorySubtype>(r.subtype);
  if (kind == HistoryTokenKind::BOUNDARY && subtype == HistorySubtype::NEW_TURN) {
    NewTurn(r.turn);
    return out;
  }
  if (kind == HistoryTokenKind::ACTION || kind == HistoryTokenKind::PRIVATE_CHOICE) {
    if (r.from_location) {
      if (r.from_controller_relative == 1 &&
          (r.from_location == LOCATION_HAND || r.from_location == LOCATION_DECK ||
           r.from_location == LOCATION_EXTRA))
        return out;
      out.subject = From(r);
    } else {
      out.subject = To(r);
    }
    return out;
  }
  if (kind == HistoryTokenKind::PUBLIC_EVENT) {
    out.subject = Public(r);
    return out;
  }
  if (kind != HistoryTokenKind::STATE_DELTA) return out;
  switch (subtype) {
    case HistorySubtype::MOVE:
      out.subject = Move(r);
      return out;
    case HistorySubtype::DRAW: {
      const int side = r.player_relative;
      auto& cards = lists_[{side, LOCATION_HAND}];
      std::vector<int64_t> revealed;
      if (auto hit = reveals_.find(side); hit != reveals_.end()) {
        revealed = std::move(hit->second);
        reveals_.erase(hit);
      }
      std::vector<int64_t> drawn;
      for (int64_t index = 0; index < r.count; ++index) {
        const int64_t entity = New();
        if (index < static_cast<int64_t>(revealed.size())) Learn(entity, revealed[static_cast<size_t>(index)]);
        EntityInfo& info = Info(entity);
        info.arrival_turn = r.turn;
        info.arrival_kind = ARRIVAL_MOVED;
        info.arrived_from = LOCATION_DECK;
        cards.push_back(entity);
        drawn.push_back(entity);
      }
      out.subject = drawn.size() == 1 ? drawn[0] : 0;
      return out;
    }
    case HistorySubtype::POSITION: {
      const int64_t entity = To(r);
      Learn(entity, r.card_code);
      if (entity) {
        if (r.to_position & POS_FACEDOWN) facedown_.insert(entity);
        else facedown_.erase(entity);
        if ((r.from_position & POS_FACEUP) && (r.to_position & POS_FACEDOWN)) Info(entity).counters.clear();
      }
      out.subject = entity;
      return out;
    }
    case HistorySubtype::CHAIN: {
      const int64_t entity = From(r);
      Learn(entity, r.card_code);
      out.subject = entity;
      if (r.value != CHAIN_CHAINING) return out;
      const int side = r.player_relative;
      if (side != 0 && side != 1) Fail("a chain activation names no player");
      ++counts_[static_cast<size_t>(side)][4];
      if (entity) ++Info(entity).activations;
      const int64_t code = r.card_code, desc = r.detail;
      const std::pair<int64_t, int64_t> key = desc >= 10000 ? std::make_pair(desc >> 4, desc)
                                                            : std::make_pair(code, desc);
      auto& row = activations_[key];
      ++row[static_cast<size_t>(side)];
      ++row[static_cast<size_t>(2 + side)];
      row[4] = r.turn;
      row[5] = ++order_;
      out.effect_present = true;
      out.effect_code = code;
      out.effect_desc = desc;
      return out;
    }
    case HistorySubtype::TARGET:
      out.subject = To(r);
      return out;
    case HistorySubtype::ATTACK: {
      const int64_t attacker = From(r);
      const int side = r.from_controller_relative;
      if (side == 0 || side == 1) ++counts_[static_cast<size_t>(side)][5];
      if (attacker) ++Info(attacker).attacks;
      out.subject = attacker;
      out.counterpart = r.to_location ? To(r) : 0;
      return out;
    }
    case HistorySubtype::NEGATED:
      out.subject = From(r);
      return out;
    default:
      return out;
  }
}

int64_t PublicTracker::Move(const HistoryTokenRecord& r) {
  const int from_side = r.from_controller_relative, from_location = r.from_location,
            from_sequence = r.from_sequence, from_position = r.from_position;
  const int to_side = r.to_controller_relative, to_location = r.to_location, to_sequence = r.to_sequence,
            to_position = r.to_position;
  int64_t entity = from_location ? Take(from_side, from_location, from_sequence, from_position) : 0;
  if (!entity) entity = New();
  if (!(from_location & LOCATION_OVERLAY) && !(to_location & LOCATION_OVERLAY)) {
    if (auto hit = overlays_.find(std::make_tuple(from_side, from_location, from_sequence)); hit != overlays_.end()) {
      std::vector<int64_t> materials = std::move(hit->second);
      overlays_.erase(hit);
      overlays_[std::make_tuple(to_side, to_location, to_sequence)] = std::move(materials);
    }
  }
  if (!to_location || to_location == LOCATION_DECK || to_location == LOCATION_EXTRA) {
    Forget(entity);
    return entity;
  }
  Learn(entity, r.card_code);
  if (IsFixed(to_location) && (to_position & POS_FACEDOWN)) facedown_.insert(entity);
  else facedown_.erase(entity);
  EntityInfo& info = Info(entity);
  const int base_from = from_location & 0x7F, base_to = to_location & 0x7F;
  if (((from_location & kOnField) && to_location != from_location) || (to_location & LOCATION_OVERLAY))
    info.counters.clear();
  if (!(from_location & LOCATION_OVERLAY) && !(to_location & LOCATION_OVERLAY))
    info.hint_kind = info.hint_value = 0;
  if (base_from != base_to || !from_location) info.activations = info.attacks = 0;
  info.arrival_turn = r.turn;
  info.arrival_kind = ARRIVAL_MOVED;
  info.arrived_from = base_from;
  Put(entity, to_side, to_location, to_sequence, to_position);
  return entity;
}

int64_t PublicTracker::Public(const HistoryTokenRecord& r) {
  const int event = r.public_event_kind;
  if (event == PUB_MISSED_EFFECT) return 0;
  const int64_t named = r.from_location ? From(r) : 0;
  const int side = r.from_controller_relative;
  if (event == PUB_CONFIRM || event == PUB_SUMMONING || event == PUB_SPSUMMONING || event == PUB_FLIPSUMMONING)
    Learn(named, r.card_code);
  switch (event) {
    case PUB_DRAW_REVEAL:
      reveals_[r.player_relative].push_back(r.card_code);
      break;
    case PUB_SHUFFLE_HAND: {
      const Zone key{r.player_relative, LOCATION_HAND};
      auto hit = lists_.find(key);
      if (hit != lists_.end()) {
        std::vector<int64_t> fresh;
        for (int64_t entity : hit->second) {
          const auto code = codes_.find(entity);
          if (code != codes_.end() && code->second)
            absorbed_[entity] = std::make_tuple(key.first, key.second, code->second);
          Forget(entity);
          fresh.push_back(New());
        }
        hit->second = std::move(fresh);
      }
      return 0;
    }
    case PUB_SHUFFLE_SET:
      for (auto& zone : fixed_)
        for (auto& slot : zone.second)
          if (facedown_.count(slot.second)) {
            Forget(slot.second);
            slot.second = New();
            facedown_.insert(slot.second);
          }
      return 0;
    case PUB_SWAP_GRAVE_DECK: {
      const Zone key{r.player_relative, LOCATION_GRAVE};
      if (auto hit = lists_.find(key); hit != lists_.end()) {
        for (int64_t entity : hit->second) Forget(entity);
        lists_.erase(hit);
      }
      resync_.insert(r.player_relative);
      return 0;
    }
    case PUB_SUMMONING:
      Count(side, 0);
      if (named) Info(named).arrival_kind = ARRIVAL_NORMAL;
      break;
    case PUB_SPSUMMONING:
      Count(side, 2);
      if (named && Info(named).arrived_from == LOCATION_EXTRA) Count(side, 3);
      if (named) Info(named).arrival_kind = ARRIVAL_SPECIAL;
      break;
    case PUB_FLIPSUMMONING:
      Count(side, 1);
      if (named) Info(named).arrival_kind = ARRIVAL_FLIP;
      break;
    case PUB_SET:
      if (r.from_location == LOCATION_MZONE) Count(side, 0);
      if (named) Info(named).arrival_kind = ARRIVAL_SET;
      break;
    case PUB_ADD_COUNTER:
    case PUB_REMOVE_COUNTER:
      if (named) {
        auto& counters = Info(named).counters;
        const int64_t value = counters[r.detail] + (event == PUB_ADD_COUNTER ? r.value : -r.value);
        if (value > 0) counters[r.detail] = value;
        else counters.erase(r.detail);
      } else if (IsFixed(r.from_location & 0x7F) || IsList(r.from_location & 0x7F)) {
        Fail("a counter names an empty public place");
      }
      break;
    case PUB_CARD_HINT:
      if (named) {
        EntityInfo& info = Info(named);
        if (r.detail == kDescAdd) {
          ++info.desc_hints[r.value];
        } else if (r.detail == kDescRemove) {
          if (--info.desc_hints[r.value] == 0) info.desc_hints.erase(r.value);
        } else {
          info.hint_kind = r.detail;
          info.hint_value = r.value;
        }
      }
      break;
    case PUB_PLAYER_HINT: {
      const int owner = r.player_relative;
      if (owner != 0 && owner != 1) Fail("a player hint names no player");
      auto& hints = hints_[static_cast<size_t>(owner)];
      if (r.value == kCardQuestion && owner == 0) {
        if (r.detail == kDescAdd) cant_check_grave_ = true;
        else if (r.detail == kDescRemove) cant_check_grave_ = false;
      } else if (r.detail == kDescAdd) {
        ++hints[r.value];
      } else if (r.detail == kDescRemove) {
        if (--hints[r.value] == 0) hints.erase(r.value);
      }
      break;
    }
    default:
      break;
  }
  return named;
}

std::array<int64_t, kTrackerCardWidth> PublicTracker::CardRow(int64_t entity, int turn) const {
  static const EntityInfo empty;
  const auto hit = info_.find(entity);
  const EntityInfo& info = hit == info_.end() ? empty : hit->second;
  int64_t bucket = 0;
  if (info.arrival_turn >= 0 && info.arrival_kind != ARRIVAL_UNKNOWN)
    bucket = 1 + std::min<int64_t>(2, std::max<int64_t>(0, turn - info.arrival_turn));
  int64_t total = 0;
  for (const auto& entry : info.counters) total += entry.second;
  std::array<int64_t, 4> pairs{};
  size_t used = 0;
  for (const auto& entry : info.counters) {
    if (used >= 4) break;
    pairs[used++] = entry.first;
    pairs[used++] = std::min<int64_t>(63, entry.second);
  }
  int64_t hint_count = 0, first_hint = 0;
  for (const auto& entry : info.desc_hints)
    if (entry.second > 0) {
      if (!hint_count) first_hint = entry.first;
      ++hint_count;
    }
  return {1, bucket, info.arrival_kind, std::min<int64_t>(3, info.activations), std::min<int64_t>(3, info.attacks), 1,
          std::min<int64_t>(63, total), std::min<int64_t>(3, static_cast<int64_t>(info.counters.size())),
          pairs[0], pairs[1], pairs[2], pairs[3], std::min<int64_t>(3, hint_count), first_hint,
          info.hint_kind, info.hint_value};
}

void PublicTracker::Check(const TrackerView& view) {
  for (int side = 0; side < 2; ++side) {
    for (int location : {LOCATION_HAND, LOCATION_GRAVE, LOCATION_REMOVED}) {
      auto& cards = lists_[{side, location}];
      const auto hit = view.zone_counts.find({side, location});
      const size_t count = hit == view.zone_counts.end() ? 0 : static_cast<size_t>(hit->second);
      if (location == LOCATION_GRAVE && resync_.count(side) && cards.size() < count)
        while (cards.size() < count) cards.push_back(New());
      if (cards.size() != count)
        Fail("tracked " + std::to_string(side) + "/" + std::to_string(location) + " holds " +
             std::to_string(cards.size()) + " cards, public count " + std::to_string(count));
    }
    resync_.erase(side);
    for (int location : {LOCATION_MZONE, LOCATION_SZONE}) {
      std::set<int> occupied, tracked;
      for (const TrackerToken& t : view.tokens)
        if (t.kind == TrackerToken::CARD && t.side == side && t.location == location)
          occupied.insert(t.sequence);
      if (auto hit = fixed_.find({side, location}); hit != fixed_.end())
        for (const auto& slot : hit->second) tracked.insert(slot.first);
      if (occupied != tracked)
        Fail("tracked " + std::to_string(side) + "/" + std::to_string(location) +
             " slots differ from the public field");
    }
    std::map<std::pair<int, int>, int64_t> materials, tracked;
    for (const TrackerToken& t : view.tokens)
      if (t.kind == TrackerToken::OVERLAY_MATERIAL && t.side == side)
        ++materials[{t.location, t.sequence}];
    for (const auto& entry : overlays_)
      if (std::get<0>(entry.first) == side && !entry.second.empty())
        tracked[{std::get<1>(entry.first), std::get<2>(entry.first)}] = static_cast<int64_t>(entry.second.size());
    if (materials != tracked)
      Fail("tracked materials of side " + std::to_string(side) + " differ from the public field");
  }
}

TrackerExport PublicTracker::Export(const TrackerView& view, int turn, const std::vector<TrackerRowAnnotation>& rows) {
  Check(view);
  const auto& tokens = view.tokens;
  std::map<std::tuple<int, int, int, int>, int64_t> places;  /* (side, location, sequence, index) -> token */
  for (size_t i = 0; i < tokens.size(); ++i) {
    const TrackerToken& t = tokens[i];
    if (t.kind == TrackerToken::CARD)
      places.emplace(std::make_tuple(t.side, t.location, t.sequence, -1), static_cast<int64_t>(i));
    else if (t.kind == TrackerToken::OVERLAY_MATERIAL)
      places[std::make_tuple(t.side, LOCATION_OVERLAY | t.location, t.sequence, t.overlay_index)] =
          static_cast<int64_t>(i);
  }
  std::map<int64_t, std::tuple<int, int, int, int>> where;
  for (const auto& zone : lists_)
    for (size_t i = 0; i < zone.second.size(); ++i)
      where[zone.second[i]] = std::make_tuple(zone.first.first, zone.first.second, static_cast<int>(i), -1);
  for (const auto& zone : fixed_)
    for (const auto& slot : zone.second)
      where[slot.second] = std::make_tuple(zone.first.first, zone.first.second, slot.first, -1);
  for (const auto& stack : overlays_)
    for (size_t i = 0; i < stack.second.size(); ++i)
      where[stack.second[i]] = std::make_tuple(std::get<0>(stack.first), LOCATION_OVERLAY | std::get<1>(stack.first),
                                               std::get<2>(stack.first), static_cast<int>(i));
  auto group = [&](int side, int location, int64_t code) -> int64_t {
    for (size_t i = 0; i < tokens.size(); ++i) {
      const TrackerToken& t = tokens[i];
      if ((t.kind == TrackerToken::CARD || t.kind == TrackerToken::KNOWN_UNPOSITIONED) && t.side == side &&
          t.location == location && t.code == code)
        return static_cast<int64_t>(i);
    }
    return -1;
  };
  auto resolve = [&](int64_t entity) -> std::pair<int64_t, int64_t> {
    if (!entity) return {-1, LINK_NONE};
    const auto hit = where.find(entity);
    if (hit == where.end()) {
      const auto key = absorbed_.find(entity);
      if (key == absorbed_.end()) return {-1, LINK_ABSENT};
      const int64_t index = group(std::get<0>(key->second), std::get<1>(key->second), std::get<2>(key->second));
      return index >= 0 ? std::make_pair(index, int64_t(LINK_GROUP)) : std::make_pair(int64_t(-1), int64_t(LINK_ABSENT));
    }
    const auto& place = hit->second;
    if (std::get<0>(place) == 1 && std::get<1>(place) == LOCATION_HAND) {
      const auto code = codes_.find(entity);
      const int64_t index = code != codes_.end() && code->second ? group(1, LOCATION_HAND, code->second) : -1;
      return index >= 0 ? std::make_pair(index, int64_t(LINK_GROUP)) : std::make_pair(int64_t(-1), int64_t(LINK_ABSENT));
    }
    const auto token = places.find(place);
    if (token == places.end()) Fail("a tracked entity has no public token at its place");
    return {token->second, LINK_EXACT};
  };

  TrackerExport out;
  out.cards.assign(tokens.size(), {});
  for (const auto& entry : places) {
    const auto& [side, location, sequence, index] = entry.first;
    int64_t entity = 0;
    if (index >= 0) {
      entity = At(side, location, sequence, index);
    } else if (IsFixed(location) || location == LOCATION_GRAVE || location == LOCATION_REMOVED ||
               (location == LOCATION_HAND && side == 0)) {
      entity = At(side, location, sequence, 0);
    } else {
      continue;
    }
    if (!entity) Fail("a public token has no tracked entity at its place");
    out.cards[static_cast<size_t>(entry.second)] = CardRow(entity, turn);
  }
  out.links.reserve(rows.size());
  for (const TrackerRowAnnotation& row : rows) {
    const auto subject = resolve(row.subject);
    const auto counterpart = resolve(row.counterpart);
    out.links.push_back({subject.first, subject.second, counterpart.first, counterpart.second,
                         row.effect_present ? 1 : 0, row.effect_code, row.effect_desc});
  }
  for (int side = 0; side < 2; ++side) {
    const size_t base = static_cast<size_t>(side) * (kTrackerTurnCounts + 1);
    for (int i = 0; i < kTrackerTurnCounts; ++i)
      out.ledger[base + static_cast<size_t>(i)] = counts_[static_cast<size_t>(side)][static_cast<size_t>(i)];
    int64_t total = 0;
    for (const auto& entry : hints_[static_cast<size_t>(side)])
      if (entry.second > 0) total += entry.second;
    out.ledger[base + kTrackerTurnCounts] = total;
  }
  std::vector<std::pair<const std::pair<int64_t, int64_t>*, const std::array<int64_t, 6>*>> ordered;
  for (const auto& entry : activations_) ordered.emplace_back(&entry.first, &entry.second);
  std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return (*a.second)[5] > (*b.second)[5]; });
  std::vector<std::array<int64_t, kTrackerHintWidth>> hints;
  for (int side = 0; side < 2; ++side)
    for (const auto& entry : hints_[static_cast<size_t>(side)])
      if (entry.second > 0) hints.push_back({side, entry.first, entry.second});
  out.ledger[kTrackerLedgerWidth - 3] = std::max<int64_t>(0, static_cast<int64_t>(ordered.size()) -
                                                             static_cast<int64_t>(kTrackerActivationLimit));
  out.ledger[kTrackerLedgerWidth - 2] = std::max<int64_t>(0, static_cast<int64_t>(hints.size()) -
                                                             static_cast<int64_t>(kTrackerHintLimit));
  out.ledger[kTrackerLedgerWidth - 1] = cant_check_grave_ ? 1 : 0;
  for (size_t i = 0; i < ordered.size() && i < kTrackerActivationLimit; ++i) {
    const auto& row = *ordered[i].second;
    out.activations.push_back({ordered[i].first->first, ordered[i].first->second, row[0], row[1], row[2], row[3],
                               std::min<int64_t>(15, std::max<int64_t>(0, turn - row[4]))});
  }
  if (hints.size() > kTrackerHintLimit) hints.resize(kTrackerHintLimit);
  out.hints = std::move(hints);
  return out;
}

}  // namespace mfenv
