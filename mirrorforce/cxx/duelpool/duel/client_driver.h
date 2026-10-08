// Client mode of the env: one seat's observation from the messages that seat receives in a network room.
//
// A client has no engine. ClientDuel runs the env's own message handlers, public history, guards and observation
// writer (duel_env.h) on the seat's received stream, one message per Feed; every card query the env would make of
// its engine is answered from the client's card view, which the caller keeps (mirrorforce/netduel/agent_client.py:
// the shadow board over the server's refreshes and moves) and sets per zone before each message (SetCards). A
// prompt with one legal row is answered inside, as in the env, and Feed returns the response bytes the client must
// send; a prompt with more rows waits for Step, which returns the bytes (or nothing while a multi-card selection
// still collects its sub-choices, exactly as the env's sub-decisions). Messages only the server writes
// (MSG_START, MSG_WAITING, MSG_UPDATE_DATA, MSG_UPDATE_CARD) belong to the card view, never to Feed; MSG_RETRY (the
// server refused a response) is an error.
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "duel/scripted_driver.h"
#include "duel/public_world.h"

namespace duelenv {

class ClientDuel : public ScriptedDuel {
 public:
  struct CurrentRootView {
    std::string root_id, root_hash, hypothesis_hash;
    int viewer = -1, turn = 0, turn_player = -1, phase = 0;
    std::array<int32_t, 2> lp{};
    std::vector<uint32_t> main, extra, opponent_main, opponent_extra;
    std::map<std::pair<int, int>, std::vector<std::optional<ClientCard>>> cards;
    history::CurrentRootSeed history;
    std::vector<std::array<uint32_t, 4>> reveals;  // absolute player, location, sequence, code
  };
  ClientDuel(const DuelEnvSpec &spec, int seat, std::vector<uint32_t> main, std::vector<uint32_t> extra,
             std::optional<std::vector<uint32_t>> opponent_main = std::nullopt,
             std::optional<std::vector<uint32_t>> opponent_extra = std::nullopt)
      : ScriptedDuel(spec, DealFor(seat, std::move(main), std::move(extra), std::move(opponent_main),
                                   std::move(opponent_extra), spec.config["public_opponent_recipe"_])), seat_(seat) {
    // a client's card view does not follow face-up Extra Deck cards (pendulums): refused, not approximated
    for (const auto *deck : {&deal_.deck_orders[seat], &deal_.extra[seat]})
      for (uint32_t code : *deck)
        if (c_get_card(code).type() & TYPE_PENDULUM)
          throw std::runtime_error("client mode: pendulum card " + std::to_string(code) + " in the deck (face-up "
                                   "Extra Deck cards are not followed)");
    if (spec.config["export_both_seats"_])
      throw std::runtime_error("client mode: export_both_seats needs the other seat's view, which a client has not");
    allow_unreviewed_ = spec.config["allow_unreviewed_public_effects"_];
    RequireReviewed(deal_.deck_orders[seat], allow_unreviewed_, "client deck");
    RequireReviewed(deal_.extra[seat], allow_unreviewed_, "client extra deck");
    RequireReviewed(deal_.deck_orders[1 - seat], allow_unreviewed_, "declared public opponent deck");
    RequireReviewed(deal_.extra[1 - seat], allow_unreviewed_, "declared public opponent extra deck");
    client_mode_ = true;
    const ScriptedDeal &deal = deal_;
    scripted_duel_ = [deal] {
      MDuel duel{0, 0};
      duel.main_deck0 = deal.deck_orders[0];
      duel.extra_deck0 = deal.extra[0];
      duel.deck_name0 = "client0";
      duel.main_deck1 = deal.deck_orders[1];
      duel.extra_deck1 = deal.extra[1];
      duel.deck_name1 = "client1";
      return duel;
    };
    response_log_ = nullptr;
    history_.Keep(false);
    Start();
    history_.Passive(1 - seat);
  }

  int seat() const { return seat_; }

  // Static database types for legal named rows; used by ShadowBoard's later grave/deck swap handling.
  std::map<uint32_t, uint32_t> CurrentCardTypes() const {
    std::map<uint32_t, uint32_t> out;
    for (const auto &[place, zone] : client_cards_)
      for (const auto &card : zone) {
        if (!card) continue;
        const uint32_t code = ShownCode(*card);
        if (code) out[code] = c_get_card(code).type();
        for (uint32_t code : card->overlay) out[code] = c_get_card(code).type();
      }
    return out;
  }

  history::PublicRootSeed CurrentPublicRootSeed() const {
    if (!pending())
      throw public_effects::PublicRootSeedError("not_pending", "export requires this client's current pending decision");
    std::vector<public_effects::SeedPlace> places;
    for (int player = 0; player < 2; ++player)
      for (int location : {LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED})
        for (const auto &card : PublicZone(player, location)) {
          if (!card || (location == LOCATION_REMOVED && (card->position & POS_FACEDOWN))) continue;
          places.push_back({player == seat_ ? 0 : 1, location, card->sequence});
        }
    auto out = history_.CurrentPublicSeed(seat_, places);
    if (out.turn != turn_count_ || out.turn_player != tp_ || out.phase != current_phase_)
      throw public_effects::PublicRootSeedError("observer_clock", "history and client current clock disagree");
    return out;
  }

  void InitializeCurrentRoot(const CurrentRootView &view) {
    if (fed_ || root_initialized_ || pending() || done_ || forced_ != 0)
      throw std::runtime_error("current-root initialization requires a fresh client with no fed messages or decision");
    if (view.viewer != seat_ || view.root_id.empty() || view.root_hash.size() != 64 ||
        view.hypothesis_hash.size() != 64 || view.turn < 1 || view.turn_player < 0 || view.turn_player > 1)
      throw std::runtime_error("invalid current-root identity, viewer or clock");
    auto same = [](std::vector<uint32_t> a, std::vector<uint32_t> b) {
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      return a == b;
    };
    if (!spec_.config["public_opponent_recipe"_] || !same(view.main, deal_.deck_orders[seat_]) ||
        !same(view.extra, deal_.extra[seat_]) || !same(view.opponent_main, deal_.deck_orders[1 - seat_]) ||
        !same(view.opponent_extra, deal_.extra[1 - seat_]))
      throw std::runtime_error("current-root recipes do not match the explicitly declared client recipes");
    for (int player = 0; player < 2; ++player)
      for (int location : {LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE,
                           LOCATION_REMOVED, LOCATION_EXTRA}) {
        const auto hit = view.cards.find({player, location});
        if (hit == view.cards.end()) throw std::runtime_error("current-root view is missing a zone");
        for (size_t i = 0; i < hit->second.size(); ++i) {
          const auto &card = hit->second[i];
          if (!card) {
            if (location != LOCATION_MZONE && location != LOCATION_SZONE)
              throw std::runtime_error("current-root list zone has a hole");
            continue;
          }
          if (card->controller != player || card->location != location || card->sequence != i || card->owner > 1)
            throw std::runtime_error("current-root card has inconsistent coordinates or owner");
          RequireReviewed(std::array<uint32_t, 1>{card->code}, allow_unreviewed_, "current-root card");
          RequireReviewed(card->overlay, allow_unreviewed_, "current-root material");
        }
      }
    // Build the potentially failing seed before changing this client. The owned hypothetical engine is elsewhere.
    auto seeded_history = history_;
    seeded_history.InitializeCurrentRoot(seat_, view.turn, view.turn_player, view.phase, view.history);
    client_cards_ = view.cards;
    history_ = std::move(seeded_history);
    for (int player = 0; player < 2; ++player) lp_[player] = view.lp[player];
    turn_count_ = view.turn;
    tp_ = view.turn_player;
    current_phase_ = view.phase;
    to_play_ = seat_;
    revealed_.clear();
    for (const auto &r : view.reveals)
      revealed_[{static_cast<uint8_t>(r[0]), static_cast<uint8_t>(r[1]), static_cast<uint8_t>(r[2])}] =
          Reveal{static_cast<uint8_t>(1 << seat_), r[3]};
    root_initialized_ = true;
    observed_ = false;
  }

  // The client's cards in one zone, by slot (nullopt for an empty field slot).
  void SetCards(int player, int location, std::vector<std::optional<ClientCard>> cards) {
    if (player != 0 && player != 1) throw std::runtime_error("a player is 0 or 1");
    // the opponent's cards as they come into view (public_effects/v1 is reviewed card by card)
    for (const auto &card : cards)
      if (card) {
        RequireReviewed(std::array<uint32_t, 1>{card->code}, allow_unreviewed_, "client card in view");
        RequireReviewed(card->overlay, allow_unreviewed_, "client material in view");
      }
    client_cards_[{player, location}] = std::move(cards);
  }

  // One received message: the response bytes when the env answered a prompt with one legal row, else nothing.
  std::optional<std::vector<uint8_t>> Feed(int msg, const std::vector<uint8_t> &payload) {
    // MSG_WAITING 3, MSG_START 4, MSG_UPDATE_DATA 6, MSG_UPDATE_CARD 7 (the server's; not in the core's header)
    if (msg == 3 || msg == 4 || msg == 6 || msg == 7)
      throw std::runtime_error("client mode: message " + std::to_string(msg) + " is the server's, not the duel's");
    if (msg == MSG_RETRY) throw std::runtime_error("client mode: the server refused the last response (MSG_RETRY)");
    if (done_) throw std::runtime_error("client mode: a message after the duel's end");
    if (!legal_actions_.empty())
      throw std::runtime_error("client mode: a message arrived while a decision is pending");
    if (payload.size() + 1 > sizeof(data_)) throw std::runtime_error("client mode: a message over the buffer");
    fed_ = true;
    data_[0] = static_cast<uint8_t>(msg);
    std::copy(payload.begin(), payload.end(), data_ + 1);
    dl_ = static_cast<int>(payload.size() + 1);
    const size_t before = repro_responses_.size();
    legal_actions_.clear();
    const bool pending = consume_buffer();
    observed_ = false;
    if (msg == MSG_WIN) done_ = true;
    if (repro_responses_.size() > before + 1)
      throw std::runtime_error("client mode: one message produced more than one response");
    if (pending) {
      if (repro_responses_.size() != before)
        throw std::runtime_error("client mode: a prompt both answered and pending");
      return std::nullopt;
    }
    legal_actions_.clear();
    if (repro_responses_.size() == before) return std::nullopt;
    ++forced_;
    return repro_responses_.back();
  }

  // The policy's choice at the pending prompt: the response bytes, or nothing while a selection collects.
  std::optional<std::vector<uint8_t>> Decide(int index) {
    if (legal_actions_.empty()) throw std::runtime_error("client mode: no decision is pending");
    const size_t before = repro_responses_.size();
    Step(index);
    if (repro_responses_.size() == before) return std::nullopt;  // a multi-card selection's sub-choice
    if (repro_responses_.size() != before + 1) throw std::runtime_error("client mode: one decision, several responses");
    if (ms_idx_ != -1) throw std::runtime_error("client mode: a response sent while a selection still collects");
    legal_actions_.clear();
    return repro_responses_.back();
  }

  // The pending decision answered by its response bytes (a recorded game's player, as its client sent them): the menu
  // path whose response is exactly ``data`` -- one row, or a multi-card selection's sub-choices in the order the
  // bytes list them -- applied as Decide applies it, so the seat's own records are the ones its menu steps would
  // have made. Throws when no path gives ``data``, or when more than one does (the bytes would not determine the
  // seat's own records).
  std::vector<uint8_t> DecideResponse(const std::vector<uint8_t> &data) {
    if (!pending()) throw std::runtime_error("client mode: no decision is pending");
    if (data.empty()) throw std::runtime_error("client mode: an empty response");
    const EnvState root = save_env();
    const bool root_observed = observed_;
    std::vector<std::vector<int>> found;
    std::vector<int> path;
    size_t tries = 0;
    SearchResponse(data, path, found, tries);
    load_env(root);
    observed_ = root_observed;
    if (found.empty())
      throw std::runtime_error("client mode: no menu path gives the response " + Hex(data) + " (msg " +
                               std::to_string(msg_) + ")");
    if (found.size() > 1)
      throw std::runtime_error("client mode: " + std::to_string(found.size()) + " menu paths give the response " +
                               Hex(data) + " (msg " + std::to_string(msg_) + ")");
    std::optional<std::vector<uint8_t>> out;
    for (int index : found[0]) out = Decide(index);
    if (!out || *out != data) throw std::runtime_error("client mode: the found path did not repeat its response");
    return *out;
  }

  // Read-only response decoding for replaying network memory at EVERY sub-decision.
  // SearchResponse observes and steps its receiver; use an independent clone so
  // neither successful decoding nor an exception consumes the original's history.
  std::vector<int> ResponsePath(const std::vector<uint8_t> &data) const {
    if (!pending()) throw std::runtime_error("client mode: no decision is pending");
    if (data.empty()) throw std::runtime_error("client mode: an empty response");
    auto scratch = Clone();
    std::vector<std::vector<int>> found;
    std::vector<int> path;
    size_t tries = 0;
    scratch->SearchResponse(data, path, found, tries);
    if (found.empty())
      throw std::runtime_error("client mode: no menu path gives the response " + Hex(data));
    if (found.size() != 1)
      throw std::runtime_error("client mode: multiple menu paths give the response " + Hex(data));
    return found.front();
  }

  // A decision is pending: a prompt with more than one row, or a multi-card selection's next sub-choice (shown even
  // with one row left, as the env shows it).
  bool pending() const { return !done_ && !legal_actions_.empty(); }
  int forced_count() const { return forced_; }

  // Engine-free, read-only belief context for this client's own pending decision. Material ownership is supplied
  // from the client's received public material slots, not inferred from the host monster's controller/owner.
  // Each row is (controller, location, sequence, material index, visible code, public owner); all visible materials
  // must be covered exactly. Run the extraction on a clone so query caches/rebinding cannot touch this client.
  public_belief::World PublicWorld(const std::vector<std::array<uint32_t, 6>> &material_owners) const {
    if (!spec_.config["public_opponent_recipe"_])
      throw std::runtime_error("client public_world requires an explicitly declared public opponent decklist");
    if (!pending()) throw std::runtime_error("client public_world requires this observer's pending decision");
    auto scratch = Clone();
    return scratch->BuildPublicWorld(material_owners);
  }

  // An independent copy at this point of the stream (a search branches per rollout): the env state, the card view
  // and the counters. The pending prompt's menu is rebuilt in the copy (a menu's callback belongs to its duel).
  std::unique_ptr<ClientDuel> Clone() const {
    const bool public_recipe = spec_.config["public_opponent_recipe"_];
    auto out = std::make_unique<ClientDuel>(spec_, seat_, deal_.deck_orders[seat_], deal_.extra[seat_],
        public_recipe ? std::optional<std::vector<uint32_t>>(deal_.deck_orders[1 - seat_]) : std::nullopt,
        public_recipe ? std::optional<std::vector<uint32_t>>(deal_.extra[1 - seat_]) : std::nullopt);
    out->load_env(save_env());
    out->client_cards_ = client_cards_;
    out->observed_ = false;
    out->forced_ = forced_;
    out->fed_ = fed_;
    out->root_initialized_ = root_initialized_;
    out->Rebind();
    return out;
  }

 private:
  bool fed_ = false, root_initialized_ = false;
  static constexpr size_t kMaxResponseTries = 50000;

  const std::vector<std::optional<ClientCard>> &PublicZone(int player, int location) const {
    static const std::vector<std::optional<ClientCard>> empty;
    const auto hit = client_cards_.find({player, location});
    return hit == client_cards_.end() ? empty : hit->second;
  }

  uint32_t ShownCode(const ClientCard &raw) const {
    Card card = raw.code ? c_get_card(raw.code) : Card();
    card.set_location(raw.controller | (static_cast<uint32_t>(raw.location) << 8) |
                      (static_cast<uint32_t>(raw.sequence) << 16) | (static_cast<uint32_t>(raw.position) << 24));
    return visible_code(seat_, card);
  }

  public_belief::World BuildPublicWorld(const std::vector<std::array<uint32_t, 6>> &material_owners) {
    const int opponent = 1 - seat_;
    public_belief::Input input;
    for (uint8_t location : {LOCATION_HAND, LOCATION_DECK, LOCATION_MZONE, LOCATION_SZONE, LOCATION_REMOVED,
                             LOCATION_EXTRA})
      for (const auto &item : PublicZone(opponent, location)) {
        if (!item) continue;
        const ClientCard &card = *item;
        const uint32_t shown = ShownCode(card);
        const bool extra = shown ? (c_get_card(shown).type() & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK)) != 0
                                 : history_.ArrivedFrom(seat_, 1, location, card.sequence) == LOCATION_EXTRA;
        input.opponent.push_back({location, card.sequence, shown, card.position, extra});
      }
    auto add = [&](uint32_t code) {
      const Card &data = c_get_card(code);
      if (!(data.type() & TYPE_TOKEN))
        input.public_owned.emplace_back(code, (data.type() & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK)) != 0);
    };
    for (int player = 0; player < 2; ++player)
      for (uint8_t location : {LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE,
                               LOCATION_REMOVED, LOCATION_EXTRA}) {
        const auto &zone = PublicZone(player, location);
        for (size_t sequence = 0; sequence < zone.size(); ++sequence) {
          if (!zone[sequence]) continue;
          const ClientCard &raw = *zone[sequence];
          if (raw.controller != player || raw.location != location || raw.sequence != sequence || raw.owner > 1)
            throw std::runtime_error("client public_world: card view has inconsistent public coordinates/owner");
          const uint32_t shown = ShownCode(raw);
          if (raw.owner == static_cast<uint32_t>(opponent) && shown) add(shown);
          if (location != LOCATION_MZONE && !raw.overlay.empty())
            throw std::runtime_error("client public_world: material ownership outside monster zones is unsupported");
        }
      }
    using MaterialKey = std::tuple<uint32_t, uint32_t, uint32_t>;
    std::map<MaterialKey, std::pair<uint32_t, uint32_t>> owners;
    for (const auto &row : material_owners) {
      if (row[0] > 1 || row[1] != LOCATION_MZONE || row[2] >= 7 || row[4] == 0 || row[5] > 1 ||
          !owners.emplace(MaterialKey{row[0], row[2], row[3]}, std::make_pair(row[4], row[5])).second)
        throw std::runtime_error("client public_world: invalid or duplicate public material owner row");
    }
    // Keep the established SearchDuel/query_overlay_owners ordering: side, monster slot, material index.
    for (int player = 0; player < 2; ++player) {
      const auto &zone = PublicZone(player, LOCATION_MZONE);
      for (size_t sequence = 0; sequence < zone.size(); ++sequence) {
        if (!zone[sequence]) continue;
        for (size_t i = 0; i < zone[sequence]->overlay.size(); ++i) {
          const auto hit = owners.find(MaterialKey{player, sequence, i});
          if (hit == owners.end() || hit->second.first != zone[sequence]->overlay[i])
            throw std::runtime_error("client public_world: missing/mismatched public material ownership");
          if (hit->second.second == static_cast<uint32_t>(opponent)) add(hit->second.first);
          owners.erase(hit);
        }
      }
    }
    if (!owners.empty()) throw std::runtime_error("client public_world: owner rows name absent materials");
    input.unpositioned = history_.Unpositioned(seat_);
    input.hand_group = history_.HandGroup(seat_);
    for (const auto &item : PublicZone(seat_, LOCATION_DECK)) {
      if (!item || !item->code) throw std::runtime_error("client public_world: own remaining deck multiset is incomplete");
      input.own_deck.push_back(item->code);
      if (revealed_to(seat_, seat_, LOCATION_DECK, item->sequence))
        input.own_deck_fixed.emplace_back(item->sequence, revealed_.at({seat_, LOCATION_DECK, item->sequence}).code);
    }
    return public_belief::Build(std::move(input), true, deal_.deck_orders[opponent], deal_.extra[opponent],
                                [](uint32_t code) { return c_get_card(code).type(); });
  }

  static std::string Hex(const std::vector<uint8_t> &bytes) {
    static const char *digits = "0123456789abcdef";
    std::string out;
    for (uint8_t b : bytes) {
      out += digits[b >> 4];
      out += digits[b & 0xF];
    }
    return out;
  }

  // Whether a multi-card selection's sub-choices so far are the first indices ``data`` lists (the env's response is
  // the count, the must-select zeros of a sum selection, then the chosen indices in order).
  bool ResponsePrefix(const std::vector<uint8_t> &data) const {
    if (ms_idx_ == -1) return true;  // not a card selection: no prefix to check
    const size_t offset = 1 + (ms_mode_ == 0 ? 0 : static_cast<size_t>(ms_must_));
    if (data.size() < offset + ms_r_idxs_.size()) return false;
    for (size_t k = 0; k < ms_r_idxs_.size(); ++k)
      if (data[offset + k] != static_cast<uint8_t>(ms_r_idxs_[k])) return false;
    return true;
  }

  // Depth-first over the rows of the pending prompt and its sub-choices, from this state; the env state is the one it
  // started from when it returns.
  void SearchResponse(const std::vector<uint8_t> &data, std::vector<int> &path, std::vector<std::vector<int>> &found,
                      size_t &tries) {
    if (!observed_) Observe();  // the guards decide the shown menu when the observation is written
    const EnvState saved = save_env();
    const size_t shown = visible_.empty() ? legal_actions_.size() : visible_.size();
    for (size_t i = 0; i < shown && found.size() < 2; ++i) {
      if (++tries > kMaxResponseTries)
        throw std::runtime_error("client mode: more than " + std::to_string(kMaxResponseTries) +
                                 " menu steps tried for one response");
      load_env(saved);
      observed_ = true;
      const auto out = Decide(static_cast<int>(i));
      path.push_back(static_cast<int>(i));
      if (out) {
        if (*out == data) found.push_back(path);
      } else if (pending() && ResponsePrefix(data)) {
        SearchResponse(data, path, found, tries);
      }
      path.pop_back();
    }
    load_env(saved);
    observed_ = true;
  }

  void Rebind() {
    if (done_ || legal_actions_.empty()) return;
    if (ms_idx_ != -1) {
      handle_multi_select();  // a selection's sub-choice: its menu and callback from the selection state
      return;
    }
    if (message_starts_.size() != 2 || dp_ != dl_)
      throw std::runtime_error("client mode: a pending prompt is the buffer's one message");
    dp_ = 0;
    legal_actions_.clear();
    handle_message();
    if (dp_ != dl_ || legal_actions_.empty()) throw std::runtime_error("client mode: the prompt did not rebuild");
  }

 public:

 private:
  static ScriptedDeal DealFor(int seat, std::vector<uint32_t> main, std::vector<uint32_t> extra,
                              std::optional<std::vector<uint32_t>> opponent_main,
                              std::optional<std::vector<uint32_t>> opponent_extra, bool public_recipe) {
    if (seat != 0 && seat != 1) throw std::runtime_error("a seat is 0 or 1");
    if (public_recipe) {
      if (!opponent_main || opponent_main->empty() || !opponent_extra)
        throw std::runtime_error("client mode: public opponent recipe must be explicitly declared");
      if (opponent_main->size() > 60 || opponent_extra->size() > 15)
        throw std::runtime_error("client mode: invalid public opponent recipe size");
    } else if (opponent_main || opponent_extra) {
      throw std::runtime_error("client mode: opponent recipe supplied in closed-decklist mode");
    }
    ScriptedDeal deal;
    deal.seed_words = {0};
    deal.deck_orders[seat] = std::move(main);
    deal.extra[seat] = std::move(extra);
    if (public_recipe) {
      deal.deck_orders[1 - seat] = std::move(*opponent_main);
      deal.extra[1 - seat] = std::move(*opponent_extra);
      std::sort(deal.deck_orders[1 - seat].begin(), deal.deck_orders[1 - seat].end());
      std::sort(deal.extra[1 - seat].begin(), deal.extra[1 - seat].end());
    }
    deal.start_lp = 8000;
    deal.start_hand = 5;
    deal.draw_count = 1;
    deal.duel_options = 5 << 16;
    return deal;
  }

  int seat_;
  int forced_ = 0;
  bool allow_unreviewed_ = false;  // allow_unreviewed_public_effects: tests and audits of other pools
};

}  // namespace duelenv
