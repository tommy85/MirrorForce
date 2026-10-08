// Scripted duels on the env: one DuelEnvImpl driven through an explicit deal, for probes and acceptance tests.
//
// The deal is mfenv's (mirrorforce/cxx/mfenv/instance.h ``Deal``): eight core seed words for create_duel_v2, each
// main deck in new_card order (drawn from the end, so opening hands are chosen), each extra deck in .ydk order (added
// back to front as the env does), and the player settings, which must be the env's own. The duel then runs through
// the env's normal message handling, menus and observation writer; prompts with one legal row are answered inside the
// env as always. The driver keeps what tests compare: the core message stream, each viewer's public facts
// (history_obs.h), every response the core was given as the bytes a network client sends, and observations of the
// player to move. mirrorforce/probes/scripted.py runs named steps against it (probes/agent_adapter.py).
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "duel/duel_env.h"

namespace duelenv {

struct ScriptedDeal {
  std::vector<uint32_t> seed_words;
  std::array<std::vector<uint32_t>, 2> deck_orders;
  std::array<std::vector<uint32_t>, 2> extra;
  int32_t start_lp = 0, start_hand = 0, draw_count = 0;
  uint32_t duel_options = 0;
};

class ScriptedDuel : public DuelEnvImpl {
 public:
  using DuelEnvImpl::State;

  ScriptedDuel(const DuelEnvSpec &spec, ScriptedDeal deal) : DuelEnvImpl(spec, 0), deal_(std::move(deal)) {
    if (std::string(spec.config["play_mode"_]) != "self")
      throw std::runtime_error("a scripted duel is self-play: the caller answers both players");
    if (spec.config["record"_] || spec.config["oppo_info"_] || spec.config["verbose"_])
      throw std::runtime_error("a scripted duel writes no replay, shows no hidden cards and prints nothing");
    if (deal_.seed_words.size() != 8 && deal_.seed_words.size() != 1)
      throw std::runtime_error("a scripted deal has eight core seed words, or one outer seed (an env repro record)");
    if (deal_.start_lp != init_lp_ || deal_.start_hand != startcount_ || deal_.draw_count != drawcount_)
      throw std::runtime_error("a scripted deal must use the env's player settings (8000 LP, 5 cards, 1 draw)");
    if (deal_.duel_options != static_cast<uint32_t>(duel_options_))
      throw std::runtime_error("a scripted deal must use the env's duel options " + std::to_string(duel_options_));
    for (const auto *decks : {&deal_.deck_orders, &deal_.extra})
      for (const auto &deck : *decks)
        for (uint32_t code : deck)
          if (cards_data_.find(code) == cards_data_.end())
            throw std::runtime_error("scripted deck card " + std::to_string(code) + " is not in the card table");
    scripted_duel_ = [this] { return Create(); };
    response_log_ = &responses_;
    history_.Keep(true);
  }
  ScriptedDuel(const ScriptedDuel &) = delete;
  ScriptedDuel &operator=(const ScriptedDuel &) = delete;
  // The core duel is ended with its owner: mid-game, or kept after the game's end (a search duel).
  ~ScriptedDuel() {
    if (pduel_ != 0 && (duel_started_ || keep_core_after_end_)) end_duel(pduel_);
  }

  void Start() {
    if (started_) throw std::runtime_error("a scripted duel starts once");
    started_ = true;
    reset();
  }

  bool finished() const { return done_; }
  int player() const { return to_play_; }
  // The player to move's own view is withheld after its hidden identities were replaced (search_api.h ReplaceHidden):
  // its private history still holds the truth's identities.
  void RequireOwnView() const {
    if (!done_ && replaced_[to_play_])
      throw std::runtime_error("observation: the player to move's hidden identities were replaced; its private history "
                               "is the truth's (an S3 particle must carry it); restore first");
  }
  int prompt_msg() const { return msg_; }
  int winner() const { return winner_; }
  int win_reason() const { return win_reason_; }
  // The menu the player to move is shown (the full menu less rows the guards withhold; guards.h).
  std::vector<LegalAction> actions() const {
    if (visible_.empty()) return legal_actions_;
    std::vector<LegalAction> shown;
    for (int row : visible_) shown.push_back(legal_actions_[row]);
    return shown;
  }
  const std::vector<std::vector<uint8_t>> &responses() const { return responses_; }
  const std::vector<mfenv::Message> &messages() const { return history_.Messages(); }
  const std::vector<mfenv::HistoryTokenRecord> &facts(int viewer) const { return history_.Facts(viewer); }
  const std::array<std::set<uint32_t>, 2> &seen(int viewer) const { return history_.Seen(viewer); }
  std::map<std::pair<int, int64_t>, int64_t> unpositioned(int viewer) const { return history_.Unpositioned(viewer); }
  std::vector<int> hand_group(int viewer) const { return history_.HandGroup(viewer); }

  // The observation of the player to move (every state key, batch dimensions of one), as training writes it.
  State Observe() {
    Require();
    std::vector<Array> arrays;
    for (ShapeSpec shape : spec_.state_spec.template AllValues<ShapeSpec>()) {
      for (int &dim : shape.shape)
        if (dim == -1) dim = 1;
      arrays.emplace_back(shape);
    }
    State state(arrays);
    WriteState(state);
    observed_ = true;
    return state;
  }

  VisibleAudit AuditVisible() {
    Require();
    return audit_visible(to_play_);
  }
  std::vector<std::array<uint32_t, 4>> EngineEquips() {
    Require();
    return engine_equips();
  }

  void Step(int index) {
    Require();
    if (done_) throw std::runtime_error("the scripted duel is over");
    if (!observed_) Observe();  // the guards decide the shown menu when the observation is written
    const size_t shown = visible_.empty() ? legal_actions_.size() : visible_.size();
    if (index < 0 || static_cast<size_t>(index) >= shown)
      throw std::runtime_error("menu row " + std::to_string(index) + " of " + std::to_string(shown));
    observed_ = false;
    step(index);
  }

  // Raw response bytes for the current prompt (duel_env.h respond_raw): a recorded game's player as its client
  // answered, bypassing the menu.
  void Respond(const std::vector<uint8_t> &bytes) {
    Require();
    observed_ = false;
    respond_raw(bytes);
  }

 protected:
  void Require() const {
    if (!started_) throw std::runtime_error("start the scripted duel first");
  }

  MDuel Create() {
    intptr_t pduel = 0;
    if (deal_.seed_words.size() == 1) {
      pduel = create_duel_from_outer_seed(deal_.seed_words[0]);  // the env's own creation (repro records)
    } else {
      std::array<uint32_t, 8> seeds{};
      std::copy(deal_.seed_words.begin(), deal_.seed_words.end(), seeds.begin());
      pduel = create_duel_v2(seeds.data());
    }
    checked_duel(pduel);
    dormant::CreateIn(pduel);  // the registered table's dormant identities, before the decks (dormant_law.h)
    for (int player = 0; player < 2; ++player) {
      set_player_info(pduel, player, deal_.start_lp, deal_.start_hand, deal_.draw_count);
      for (uint32_t code : deal_.deck_orders[player])
        new_card(pduel, code, player, player, LOCATION_DECK, 0, POS_FACEDOWN_DEFENSE);
      for (auto it = deal_.extra[player].rbegin(); it != deal_.extra[player].rend(); ++it)
        new_card(pduel, *it, player, player, LOCATION_EXTRA, 0, POS_FACEDOWN_DEFENSE);
    }
    start_duel(pduel, static_cast<int32_t>(deal_.duel_options));
    MDuel duel{pduel, deal_.seed_words.size() == 1 ? deal_.seed_words[0] : 0};
    duel.main_deck0 = deal_.deck_orders[0];
    duel.extra_deck0 = deal_.extra[0];
    duel.deck_name0 = "scripted0";
    duel.main_deck1 = deal_.deck_orders[1];
    duel.extra_deck1 = deal_.extra[1];
    duel.deck_name1 = "scripted1";
    return duel;
  }

  ScriptedDeal deal_;
  std::vector<std::vector<uint8_t>> responses_;
  bool started_ = false;
  bool observed_ = false;
};

}  // namespace duelenv
