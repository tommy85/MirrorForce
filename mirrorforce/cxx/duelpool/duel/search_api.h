// Engine search primitives of the env (search plan section 2, S0), on a ScriptedDuel.
//
// - Take / Restore: the duel's whole state at a decision point. The core part is the R4 arena (duel_snapshot /
//   duel_rollback: every allocation of the duel, Lua heap included, copied back to the same addresses); the env part
//   is every member of DuelEnvImpl (and ScriptedDuel) that the rest of the duel or its observations depend on,
//   listed in DUEL_ENV_FIELDS (duel_env.h). A member missing from the list shows up as a difference in the restore-continuity
//   test (tests/test_search_api.py), which compares every observation key and the message stream.
// - Step: ScriptedDuel::Step (one decision, single-option prompts answered inside the env).
// - PermuteHidden(player, viewer, target): rewrites player's hidden cards (hand, deck order, face-down field cards,
//   face-down extra deck) through the core's Debug.PermuteHidden. The assignment must keep every identity viewer's
//   observation shows (visible_code: own cards, public and confirmed cards, cards the viewer's tracker places; a
//   deck's order is hidden even from its owner), the opponent's hand must still hold the identities the viewer knows
//   without their positions (obs:unpositioned_, unpositioned.h), and the core requires the same cards;
//   anything else is refused, never corrected, and the duel is left unchanged.
// - ReshuffleFuture(seed): new deck orders for both players and a new core random sequence, so a rollout does not
//   inherit the real future (draws, coin tosses, random selections).
// - Layout / privileged card rows: the true hidden state, for critics, belief labels and the realizer only; the
//   observation the policy and search read stays the obs: keys (mirrorforce/agent/env/privileged.py refuses priv:).
// - PublicWorld(viewer): what a particle sampler may read (S1, mirrorforce/agent/search/particles.py): the opponent's
//   hidden places with what the viewer sees there, the identities it knows without positions, its own deck's cards
//   (not their order), and -- only when the opponent's decklist is declared public (public_opponent_recipe) -- the
//   pool of cards the hidden places hold: that decklist minus every card of the opponent the viewer has seen. A pool
//   that contradicts the public cards, or does not fill the hidden places exactly, throws.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "duel/dormant_law.h"
#include "duel/public_world.h"
#include "duel/scripted_driver.h"

extern "C" {
int32_t query_effect_info(intptr_t pduel, uint8_t *buf, int32_t buf_size);
int32_t query_overlay_owners(intptr_t pduel, uint8_t playerid, uint32_t out[], int32_t cap);
}

namespace duelenv {

class SearchDuel : public ScriptedDuel {
 public:
  // A decision point's env state: the env's own (DuelEnvImpl::EnvState, DUEL_ENV_FIELDS) and the scripted driver's
  // response log and observation flag.
  struct EnvState {
    DuelEnvImpl::EnvState env;
    decltype(SearchDuel::responses_) responses_;
    decltype(SearchDuel::observed_) observed_;
  };

  // A decision point: the core arena copy and the env state. Freed with the last reference.
  class Snapshot {
   public:
    Snapshot(void *core, int64_t extent, EnvState env) : core_(core), extent_(extent), env_(std::move(env)) {}
    ~Snapshot() { duel_snapshot_free(core_); }
    Snapshot(const Snapshot &) = delete;
    Snapshot &operator=(const Snapshot &) = delete;
    void *core() const { return core_; }
    int64_t extent() const { return extent_; }
    const EnvState &env() const { return env_; }
    const SearchDuel *owner = nullptr;

   private:
    void *core_;
    int64_t extent_;
    EnvState env_;
  };

  // The true hidden cards of one player (privileged): hand and deck in sequence order, face-down field and banished
  // cards as (location, sequence, code), face-down extra deck cards in sequence order.
  struct Layout {
    std::vector<uint32_t> hand, deck, extra;
    std::vector<std::array<uint32_t, 3>> facedown;
  };

  using World = public_belief::World;

  SearchDuel(const DuelEnvSpec &spec, ScriptedDeal deal, bool keep_history) : ScriptedDuel(spec, std::move(deal)) {
    history_.Keep(keep_history);
    keep_core_after_end_ = true;  // a finished rollout stays restorable
  }

  std::shared_ptr<Snapshot> Take() {
    Require();
    void *core = duel_snapshot(pduel_);
    if (core == nullptr) throw std::runtime_error("duel_snapshot refused: the duel's arena is empty or over its reservation");
    EnvState env{save_env(), responses_, observed_};
    auto snapshot = std::make_shared<Snapshot>(core, duel_arena_extent(pduel_), std::move(env));
    snapshot->owner = this;
    return snapshot;
  }

  void Restore(const Snapshot &snapshot) {
    Require();
    if (snapshot.owner != this) throw std::runtime_error("a snapshot restores only into the duel it was taken from");
    const int32_t rc = duel_rollback(pduel_, snapshot.core());
    if (rc != 0) throw std::runtime_error("duel_rollback returned " + std::to_string(rc));
    load_env(snapshot.env().env);
    responses_ = snapshot.env().responses_;
    observed_ = snapshot.env().observed_;
  }

  Layout HiddenLayout(int player) {
    Require();
    if (player != 0 && player != 1) throw std::runtime_error("a player is 0 or 1");
    Layout out;
    for (const Card &c : get_cards_in_location(player, LOCATION_HAND)) out.hand.push_back(c.code_);
    for (const Card &c : get_cards_in_location(player, LOCATION_DECK)) out.deck.push_back(c.code_);
    for (const Card &c : get_cards_in_location(player, LOCATION_EXTRA))
      if (c.position_ & POS_FACEDOWN) out.extra.push_back(c.code_);
    for (uint8_t location : {LOCATION_MZONE, LOCATION_SZONE, LOCATION_REMOVED})
      for (const Card &c : get_cards_in_location(player, location))
        if (!(c.location_ & LOCATION_OVERLAY) && (c.position_ & POS_FACEDOWN))
          out.facedown.push_back({static_cast<uint32_t>(c.location_), static_cast<uint32_t>(c.sequence_), c.code_});
    return out;
  }

  // Rewrites player's hidden cards to ``target`` (every list of HiddenLayout(player), in its order; the face-down
  // field slots named exactly as HiddenLayout names them; ``with_extra`` includes the face-down extra deck).
  // The checks both rewrites make before touching the duel: the assignment names every hidden place of ``player``;
  // whatever ``viewer``'s observation shows there stays (hand, revealed deck, known face-down and extra deck
  // identities); the opponent's identities the viewer knows without their positions stay among its unshown cards;
  // and no place the prompt's menu names changes its card, except the menu owner's own deck (true: rebind after).
  bool CheckAssignment(const char *what, int player, int viewer, const Layout &target, bool with_extra) {
    Require();
    if (viewer != 0 && viewer != 1) throw std::runtime_error("a viewer is 0 or 1");
    const Layout now = HiddenLayout(player);
    if (target.hand.size() != now.hand.size() || target.deck.size() != now.deck.size() ||
        target.facedown.size() != now.facedown.size() || (with_extra && target.extra.size() != now.extra.size()))
      throw std::runtime_error(std::string(what) + ": the assignment must name every hidden place of the player");
    // what the viewer's observation shows must stay
    const std::vector<Card> hand = get_cards_in_location(player, LOCATION_HAND);
    for (size_t i = 0; i < hand.size(); ++i) {
      const CardCode shown = visible_code(viewer, hand[i]);
      if (shown != 0 && target.hand[i] != shown)
        throw std::runtime_error(std::string(what) + ": hand " + std::to_string(i) + " shows " + std::to_string(shown) +
                                 " to the viewer");
    }
    const std::vector<Card> deck = get_cards_in_location(player, LOCATION_DECK);
    for (size_t i = 0; i < deck.size(); ++i)
      if (revealed_to(viewer, deck[i].controler_, deck[i].location_, deck[i].sequence_) && target.deck[i] != deck[i].code_)
        throw std::runtime_error(std::string(what) + ": deck " + std::to_string(i) + " was shown to the viewer");
    for (size_t i = 0; i < now.facedown.size(); ++i) {
      if (target.facedown[i][0] != now.facedown[i][0] || target.facedown[i][1] != now.facedown[i][1])
        throw std::runtime_error(std::string(what) + ": face-down slots must be named as the hidden layout names them");
      const Card c = get_card(player, static_cast<uint8_t>(now.facedown[i][0]), static_cast<uint8_t>(now.facedown[i][1]));
      const CardCode shown = visible_code(viewer, c);
      if (shown != 0 && target.facedown[i][2] != shown)
        throw std::runtime_error(std::string(what) + ": a face-down card the viewer knows must stay");
    }
    std::vector<Card> extra;
    if (with_extra) {
      for (const Card &c : get_cards_in_location(player, LOCATION_EXTRA))
        if (c.position_ & POS_FACEDOWN) extra.push_back(c);
      for (size_t i = 0; i < extra.size(); ++i) {
        const CardCode shown = visible_code(viewer, extra[i]);
        if (shown != 0 && target.extra[i] != shown)
          throw std::runtime_error(std::string(what) + ": an extra deck card the viewer knows must stay");
      }
    }
    if (player != viewer) {
      // the opponent's identities the viewer knows without their positions stay among the unshown cards of their
      // location (obs:unpositioned_); the extra deck is checked when it is assigned
      // the hand's known identities are among the cards of the viewer's shuffled group (not among the cards that came
      // to the hand since); the deck's and the extra deck's among all their unshown cards
      std::map<std::pair<int, int64_t>, int64_t> unshown;
      for (int i : history_.HandGroup(viewer))
        if (visible_code(viewer, hand.at(static_cast<size_t>(i))) == 0) ++unshown[{LOCATION_HAND, target.hand.at(i)}];
      for (size_t i = 0; i < deck.size(); ++i)
        if (visible_code(viewer, deck[i]) == 0) ++unshown[{LOCATION_DECK, target.deck[i]}];
      if (with_extra)
        for (size_t i = 0; i < extra.size(); ++i)
          if (visible_code(viewer, extra[i]) == 0) ++unshown[{LOCATION_EXTRA, target.extra[i]}];
      for (const auto &[key, copies] : history_.Unpositioned(viewer)) {
        if (key.first == LOCATION_EXTRA && !with_extra) continue;  // not reassigned
        if (key.first != LOCATION_HAND && key.first != LOCATION_DECK && key.first != LOCATION_EXTRA)
          throw std::runtime_error(std::string(what) + ": no check for identities without positions at location " +
                                   std::to_string(key.first));
        if (unshown[key] < copies)
          throw std::runtime_error(std::string(what) + ": the viewer knows the opponent's location " +
                                   std::to_string(key.first) + " holds " + std::to_string(copies) + " of " +
                                   std::to_string(key.second) + " (positions unknown)");
      }
    }
    // the prompt's menu names places by position: one whose identity the assignment changes would then name another
    // card (the core's pending selection holds the card itself) -- refused, except the menu owner's own deck, whose
    // places are bound by identity (bind_own_deck_selections) and are bound again below
    bool rebind = false;
    for (const LegalAction &action : legal_actions_) {
      if (action.spec_.empty() || std::isalpha(static_cast<unsigned char>(action.spec_.back()))) continue;
      const auto [controller, location, sequence, position] = spec_to_ls(to_play_, action.spec_);
      if (controller != player) continue;
      bool changed = false;
      if (location == LOCATION_HAND) changed = target.hand.at(sequence) != now.hand.at(sequence);
      else if (location == LOCATION_DECK) changed = target.deck.at(sequence) != now.deck.at(sequence);
      else if (location == LOCATION_EXTRA)  // face-down extra cards come first in the extra deck's order
        changed = with_extra && sequence < now.extra.size() && target.extra.at(sequence) != now.extra.at(sequence);
      else
        for (size_t i = 0; i < now.facedown.size(); ++i)
          if (now.facedown[i][0] == location && now.facedown[i][1] == sequence)
            changed = target.facedown[i][2] != now.facedown[i][2];
      if (!changed) continue;
      if (location == LOCATION_DECK && player == to_play_) {
        rebind = true;
        continue;
      }
      throw std::runtime_error(std::string(what) + ": the prompt names " + action.spec_ + ", whose card the assignment changes");
    }
    return rebind;
  }

  void PermuteHidden(int player, int viewer, const Layout &target, bool with_extra) {
    const Layout now = HiddenLayout(player);
    const bool rebind = CheckAssignment("permute_hidden", player, viewer, target, with_extra);
    std::string source = "mf_permute_ok = Debug.PermuteHidden(" + std::to_string(player) + "," + LuaList(target.hand) +
                         "," + LuaList(target.deck) + ",";
    std::vector<uint32_t> flat;
    for (const auto &slot : target.facedown) flat.insert(flat.end(), slot.begin(), slot.end());
    source += LuaList(flat);
    if (with_extra) source += "," + LuaList(target.extra);
    source += ")\n";
    RunChunk("./script/mf_search_permute.lua", source);
    // the core refuses a different multiset (or a slot that is not face-down) and leaves the duel unchanged
    const Layout after = HiddenLayout(player);
    if (after.hand != target.hand || after.deck != target.deck || after.facedown != target.facedown ||
        (with_extra && after.extra != target.extra))
      throw std::runtime_error("permute_hidden: the core refused the assignment (other cards than the hidden ones)");
    if (rebind) RebindOwnDeck(now.deck);
  }

  // Gives ``player``'s hidden places the identities of ``target`` in place (duel_replace_hidden: each card object
  // keeps its place, position and history, its identity's effects and script state are replaced; the multiset may
  // differ from the truth's), under the same checks as PermuteHidden, and installs the particle's ``recipe`` (main and
  // extra deck lists) as ``player``'s: the cards it owns afterwards must be exactly the recipe (and, with an open
  // decklist, the recipe the declared one), so its recipe rows and remaining counts are the particle's. Needs the
  // dormant identities of a registered table (dormant_law.h). A refusal -- by these checks or by the core -- leaves
  // the duel byte-identical and is counted by its reason (RefusalCounts: frequent refusals bias which positions can
  // be searched). After a replacement ``player``'s own view (observation, public world) is refused until a restore:
  // its private history -- its own draws, searches and seen cards -- still holds the truth's identities (S3 requirement,
  // search plan: the particle must carry that private past).
  void ReplaceHidden(int player, int viewer, const Layout &target, bool with_extra, const std::vector<uint32_t> &main,
                     const std::vector<uint32_t> &extra) {
    try {
      CheckAssignment("replace_hidden", player, viewer, target, with_extra);
    } catch (const std::runtime_error &) {
      ++refusals_["view"];
      throw;
    }
    const std::map<uint32_t, int> recipe = CheckRecipe(player, viewer, target, with_extra, main, extra);
    std::vector<uint32_t> flat;
    for (const auto &slot : target.facedown) flat.insert(flat.end(), slot.begin(), slot.end());
    // with verify_refusals (tests): the core arena right around the call, so a refusal can be shown to leave it
    // byte-identical (queries in between would rewrite card caches and evaluate scripts)
    const uint64_t before = verify_refusals_ ? ArenaDigest() : 0;
    const int32_t rc = duel_replace_hidden(
        pduel_, static_cast<uint8_t>(player), target.hand.data(), static_cast<int32_t>(target.hand.size()),
        target.deck.data(), static_cast<int32_t>(target.deck.size()), flat.data(), static_cast<int32_t>(flat.size()),
        target.extra.data(), with_extra ? static_cast<int32_t>(target.extra.size()) : -1);
    if (rc != 0 && verify_refusals_) refusal_digests_ = {before, ArenaDigest()};
    if (rc == -4) {
      // what holds which place: counted by each reason (duel_hidden_blockers' bits)
      static const std::array<const char *, 13> names = {
          "counters", "relations", "battle",      "attachments",   "foreign",      "own_effect",    "operation",
          "unique",   "chain",     "selection",   "operation_group", "turn_counter", "indestructible"};
      uint32_t all = 0;
      for (uint32_t bits : HiddenBlockers(player)) all |= bits;
      std::string named;
      for (size_t b = 0; b < names.size(); ++b)
        if (all & (1u << b)) {
          ++refusals_[std::string("held:") + names[b]];
          named += (named.empty() ? "" : ",") + std::string(names[b]);
        }
      ++refusals_["held"];
      throw std::runtime_error("replace_hidden: the core refused the assignment (held: " + named + ")");
    }
    if (rc != 0) {
      static const std::map<int32_t, const char *> reasons = {
          {-1, "boundary"}, {-2, "places"}, {-3, "identity"}, {-4, "held"}, {-5, "snapshot"}, {-6, "script"},
          {-7, "rollback"}};
      const auto it = reasons.find(rc);
      const std::string reason = it == reasons.end() ? "unknown" : it->second;
      ++refusals_[reason];
      if (rc == -7) throw std::runtime_error("replace_hidden: the core could not roll back; discard this duel");
      throw std::runtime_error("replace_hidden: the core refused the assignment (" + reason + ", " +
                               std::to_string(rc) + ")");
    }
    const Layout after = HiddenLayout(player);
    if (after.hand != target.hand || after.deck != target.deck || after.facedown != target.facedown ||
        (with_extra && after.extra != target.extra))
      throw std::runtime_error("replace_hidden: the core reported success but the layout differs");
    if (OwnedCards(player) != recipe)
      throw std::runtime_error("replace_hidden: the cards the player owns differ from the installed recipe; discard "
                               "this duel");
    (player == 0 ? main_deck0_ : main_deck1_) = main;
    (player == 0 ? extra_deck0_ : extra_deck1_) = extra;
    own_recipe_rows_[player] = recipe_rows(main, extra);
    replaced_[player] = true;
  }

  // Refuses (counted under "recipe" or "foreign") a particle recipe that is not what ``player`` would own after the
  // replacement -- its cards now, less the replaced identities, plus the target's -- or, with an open decklist,
  // differs from the declared one; or a replacement of a card ``player`` does not own (its owner's recipe would
  // change). Returns the recipe as a multiset.
  std::map<uint32_t, int> CheckRecipe(int player, int viewer, const Layout &target, bool with_extra,
                                      const std::vector<uint32_t> &main, const std::vector<uint32_t> &extra) {
    auto refuse = [&](const char *reason, const std::string &why) {
      ++refusals_[reason];
      throw std::runtime_error("replace_hidden: " + why);
    };
    std::map<uint32_t, int> recipe;
    for (const auto *part : {&main, &extra})
      for (uint32_t code : *part) {
        const Card &data = c_get_card(code);
        const bool kind = (data.type_ & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK)) != 0;
        if ((data.type_ & TYPE_TOKEN) || kind != (part == &extra))
          refuse("recipe", "recipe card " + std::to_string(code) + " is in the wrong part (or a token)");
        ++recipe[code];
      }
    if (spec_.config["public_opponent_recipe"_] && player == 1 - viewer) {
      std::map<uint32_t, int> declared;
      for (uint32_t code : player == 0 ? main_deck0_ : main_deck1_) ++declared[code];
      for (uint32_t code : player == 0 ? extra_deck0_ : extra_deck1_) ++declared[code];
      if (declared != recipe) refuse("recipe", "the recipe differs from the declared decklist");
    }
    const Layout now = HiddenLayout(player), owners = HiddenOwners(player);
    std::map<uint32_t, int> after = OwnedCards(player);
    auto change = [&](uint32_t from, uint32_t to, uint32_t owner) {
      if (from == to) return;
      if (static_cast<int>(owner) != player)
        refuse("foreign", "a replaced place holds a card " + std::to_string(player) + " does not own");
      if (--after[from] == 0) after.erase(from);
      ++after[to];
    };
    for (size_t i = 0; i < now.hand.size(); ++i) change(now.hand[i], target.hand[i], owners.hand[i]);
    for (size_t i = 0; i < now.deck.size(); ++i) change(now.deck[i], target.deck[i], owners.deck[i]);
    for (size_t i = 0; i < now.facedown.size(); ++i)
      change(now.facedown[i][2], target.facedown[i][2], owners.facedown[i][2]);
    if (with_extra)
      for (size_t i = 0; i < now.extra.size(); ++i) change(now.extra[i], target.extra[i], owners.extra[i]);
    if (after != recipe) refuse("recipe", "the recipe is not what the player would own after the replacement");
    return recipe;
  }

  // Every card ``player`` owns, wherever it is (both sides, every location, Xyz materials), tokens excepted: its
  // recipe, as a multiset (privileged: hidden cards included).
  std::map<uint32_t, int> OwnedCards(int player) {
    std::map<uint32_t, int> out;
    for (int side = 0; side < 2; ++side) {
      for (uint8_t location : {LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE,
                               LOCATION_REMOVED, LOCATION_EXTRA})
        for (const auto &r : OwnerRecords(side, location))
          if (static_cast<int>(r[5]) == player && !(c_get_card(r[0]).type_ & TYPE_TOKEN)) ++out[r[0]];
      std::vector<uint32_t> words(4 * 64);
      const int32_t n = query_overlay_owners(pduel_, static_cast<uint8_t>(side), words.data(),
                                             static_cast<int32_t>(words.size()));
      if (n < 0) throw std::runtime_error("query_overlay_owners returned " + std::to_string(n));
      for (int32_t i = 0; i < n; ++i)
        if (static_cast<int>(words[4 * i + 3]) == player) ++out[words[4 * i + 2]];
    }
    return out;
  }

  // The owners of ``player``'s hidden places, arranged as HiddenLayout arranges their codes (each code replaced by
  // its card's owner; the face-down entries keep location and sequence).
  Layout HiddenOwners(int player) {
    Layout out;
    for (const auto &r : OwnerRecords(player, LOCATION_HAND)) out.hand.push_back(r[5]);
    for (const auto &r : OwnerRecords(player, LOCATION_DECK)) out.deck.push_back(r[5]);
    for (const auto &r : OwnerRecords(player, LOCATION_EXTRA))
      if (r[4] & POS_FACEDOWN) out.extra.push_back(r[5]);
    for (uint8_t location : {LOCATION_MZONE, LOCATION_SZONE, LOCATION_REMOVED})
      for (const auto &r : OwnerRecords(player, location))
        if (r[4] & POS_FACEDOWN) out.facedown.push_back({r[2], r[3], r[5]});
    return out;
  }

  // One location's cards as the engine lists them (the order get_cards_in_location reads): code, controller,
  // location, sequence, position, owner.
  std::vector<std::array<uint32_t, 6>> OwnerRecords(int player, uint8_t location) {
    const int32_t length = OCG_QueryFieldCard(pduel_, static_cast<uint8_t>(player), location,
                                              QUERY_CODE | QUERY_POSITION | QUERY_OWNER, query_buf_);
    std::vector<std::array<uint32_t, 6>> out;
    int32_t at = 0;
    while (at < length) {
      uint32_t size = 0, code = 0;
      int32_t owner = 0;
      std::memcpy(&size, query_buf_ + at, 4);
      if (size == LEN_EMPTY) {
        at += 4;
        continue;
      }
      if (size != 20 || at + 20 > length)
        throw std::runtime_error("malformed owner query record of " + std::to_string(size) + " bytes");
      std::memcpy(&code, query_buf_ + at + 8, 4);
      std::memcpy(&owner, query_buf_ + at + 16, 4);
      out.push_back({code, query_buf_[at + 12], query_buf_[at + 13], query_buf_[at + 14], query_buf_[at + 15],
                     static_cast<uint32_t>(owner)});
      at += 20;
    }
    return out;
  }

  // Why each of a player's hidden cards could not change identity in place (duel_hidden_blockers: the hand, the deck,
  // the face-down monster, spell/trap and banished cards, the face-down extra deck; 0 when it could).
  std::vector<uint32_t> HiddenBlockers(int player) {
    Require();
    std::vector<uint32_t> out(256);
    const int32_t n = duel_hidden_blockers(pduel_, static_cast<uint8_t>(player), out.data(),
                                           static_cast<int32_t>(out.size()));
    if (n < 0) throw std::runtime_error("duel_hidden_blockers returned " + std::to_string(n));
    out.resize(static_cast<size_t>(n));
    return out;
  }

  // The viewer's own cards hidden in the opponent's control that its stream does not show (duel_env.h).
  std::array<std::map<CardCode, int>, 2> UnseenOwn(int viewer) {
    Require();
    if (viewer != 0 && viewer != 1) throw std::runtime_error("a viewer is 0 or 1");
    return unseen_own_cards(static_cast<PlayerId>(viewer));
  }

  // Refusals of ReplaceHidden by reason since the last reset (search bookkeeping, not duel state).
  std::map<std::string, int64_t> RefusalCounts(bool reset) {
    std::map<std::string, int64_t> out = refusals_;
    if (reset) refusals_.clear();
    return out;
  }

  // A full collection of the duel's Lua heap (stress lanes: replacement and rollback under GC pressure).
  void CollectGarbage() {
    Require();
    if (duel_collect_garbage(pduel_) != 0) throw std::runtime_error("duel_collect_garbage refused");
  }

  uint64_t ArenaDigest() {
    Require();
    uint64_t digest = 0;
    if (duel_arena_digest(pduel_, &digest) != 0) throw std::runtime_error("duel_arena_digest refused");
    return digest;
  }

  // The core's registered-effect dump (query_effect_info): duel-level effects, runtime card effects, the
  // once-per-turn tables and activity counters (privileged: audit and acceptance tests only).
  std::vector<uint8_t> EffectInfo() {
    Require();
    std::vector<uint8_t> buffer(1 << 20);
    const int32_t length = query_effect_info(pduel_, buffer.data(), static_cast<int32_t>(buffer.size()));
    if (length < 0) throw std::runtime_error("query_effect_info returned " + std::to_string(length));
    buffer.resize(static_cast<size_t>(length));
    return buffer;
  }

  // After the menu owner's deck was reordered: each menu place in it names the card of the same code as before, the
  // k-th unused copy in the deck's new order (the binding bind_own_deck_selections makes, in menu order), so the
  // menu's rows and the cards they show stay what they were.
  void RebindOwnDeck(const std::vector<uint32_t> &before) {
    const Layout now = HiddenLayout(to_play_);
    std::map<std::string, std::string> renamed;
    std::set<size_t> used;
    auto rebind = [&](const std::string &spec) -> std::string {
      if (spec.empty() || std::isalpha(static_cast<unsigned char>(spec.back()))) return spec;
      if (const auto hit = renamed.find(spec); hit != renamed.end()) return hit->second;
      const auto [controller, location, sequence, position] = spec_to_ls(to_play_, spec);
      if (controller != to_play_ || location != LOCATION_DECK) return spec;
      const uint32_t code = before.at(sequence);
      for (size_t i = 0; i < now.deck.size(); ++i)
        if (now.deck[i] == code && !used.count(i)) {
          used.insert(i);
          return renamed[spec] = ls_to_spec(LOCATION_DECK, static_cast<uint8_t>(i), 0, false);
        }
      throw std::runtime_error("permute_hidden: no deck card left to bind " + spec + " to");
    };
    for (std::string &spec : ms_specs_) spec = rebind(spec);
    decltype(ms_spec2idx_) remaining;
    for (const auto &[spec, index] : ms_spec2idx_) remaining[rebind(spec)] = index;
    ms_spec2idx_ = std::move(remaining);
    for (LegalAction &action : legal_actions_) action.spec_ = rebind(action.spec_);
  }

  // New deck orders for both players (uniform permutations from ``seed``) and a new core random sequence.
  void ReshuffleFuture(uint64_t seed) {
    Require();
    std::mt19937_64 rng(seed);
    for (int player = 0; player < 2; ++player) {
      Layout target = HiddenLayout(player);
      // a deck card a confirm showed keeps its place (revealed_ lasts until the chain resolves); the rest are redrawn
      const std::vector<Card> deck = get_cards_in_location(player, LOCATION_DECK);
      std::vector<size_t> free;
      for (size_t i = 0; i < deck.size(); ++i)
        if (!revealed_.count({deck[i].controler_, deck[i].location_, deck[i].sequence_})) free.push_back(i);
      std::vector<uint32_t> codes;
      for (size_t i : free) codes.push_back(target.deck[i]);
      std::shuffle(codes.begin(), codes.end(), rng);
      for (size_t j = 0; j < free.size(); ++j) target.deck[free[j]] = codes[j];
      PermuteHidden(player, player, target, false);
    }
    std::array<uint32_t, 8> words{};
    for (auto &w : words) w = static_cast<uint32_t>(rng());
    if (duel_set_future_seed(pduel_, words.data()) != 0) throw std::runtime_error("duel_set_future_seed refused");
    gen_.seed(static_cast<uint32_t>(rng()));  // the env's own draws in steps (the end phase's random discard)
  }

  // Play continues from ``text``, an env generator's state (DuelEnvImpl::play_gen_ of the game being cloned).
  void SetGenerator(const std::string &text) {
    std::istringstream in(text);
    in >> gen_;
    if (!in) throw std::runtime_error("set_generator: malformed generator state");
  }

  World PublicWorld(int viewer) {
    Require();
    if (viewer != 0 && viewer != 1) throw std::runtime_error("a viewer is 0 or 1");
    if (replaced_[viewer])
      throw std::runtime_error("public_world: this viewer's hidden identities were replaced; its private history is "
                               "the truth's (restore first)");
    const int opponent = 1 - viewer;
    public_belief::Input input;
    for (uint8_t location : {LOCATION_HAND, LOCATION_DECK, LOCATION_MZONE, LOCATION_SZONE, LOCATION_REMOVED,
                             LOCATION_EXTRA})
      for (const Card &c : get_cards_in_location(opponent, location))
        if (!((location == LOCATION_MZONE || location == LOCATION_SZONE || location == LOCATION_REMOVED) &&
              (c.location_ & LOCATION_OVERLAY))) {
          const CardCode shown = visible_code(viewer, c);
          const bool extra = shown ? (c_get_card(shown).type_ & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK)) != 0
                                   : history_.ArrivedFrom(viewer, 1, location, c.sequence_) == LOCATION_EXTRA;
          input.opponent.push_back({location, c.sequence_, shown, c.position_, extra});
        }
    input.public_owned = OpponentPublicCards(viewer);
    input.unpositioned = history_.Unpositioned(viewer);
    input.hand_group = history_.HandGroup(viewer);
    for (const Card &c : get_cards_in_location(viewer, LOCATION_DECK)) {
      input.own_deck.push_back(c.code_);
      if (revealed_to(viewer, c.controler_, c.location_, c.sequence_)) input.own_deck_fixed.emplace_back(c.sequence_, c.code_);
    }
    return public_belief::Build(std::move(input), spec_.config["public_opponent_recipe"_],
        opponent == 0 ? main_deck0_ : main_deck1_, opponent == 0 ? extra_deck0_ : extra_deck1_,
        [](uint32_t code) { return c_get_card(code).type_; });
  }

  // A player's obs:own_recipe_ rows as its observation would carry them (tests: the installed recipe of a replaced
  // player, whose observation itself is withheld).
  std::vector<uint8_t> OwnRecipeRows(int player) {
    Require();
    Array array{ShapeSpec(sizeof(uint8_t), {kRecipeRows, kOwnRecipeWidth})};
    TArray<uint8_t> rows(array);
    rows.Zero();
    _set_obs_own_recipe(rows, static_cast<PlayerId>(player));
    const auto *data = static_cast<const uint8_t *>(array.Data());
    return std::vector<uint8_t>(data, data + array.size);
  }

  // The god view of the board (every card with its identity), privileged: rows as the oppo_info writer makes them.
  std::vector<uint8_t> PrivilegedCards(std::vector<size_t> *shape) {
    Require();
    const ShapeSpec spec = spec_.state_spec["obs:cards_"_];
    std::vector<size_t> dims(spec.shape.begin(), spec.shape.end());
    Array array{ShapeSpec(spec.element_size, spec.shape)};
    TArray<uint8_t> cards(array);
    _set_obs_g_cards(cards, to_play_);
    *shape = dims;
    const auto *data = static_cast<const uint8_t *>(array.Data());
    return std::vector<uint8_t>(data, data + array.size);
  }

 private:
  // Every card of the opponent the viewer sees, wherever it is (either side, any location, Xyz materials), as (code,
  // whether it is an extra deck card); tokens are no deck cards. Ownership is public (QUERY_OWNER; a material's from
  // query_overlay_owners -- the stream shows where each material came from).
  std::vector<std::pair<uint32_t, bool>> OpponentPublicCards(int viewer) {
    const int opponent = 1 - viewer;
    std::vector<std::pair<uint32_t, bool>> out;
    auto add = [&](uint32_t code) {
      const Card &data = c_get_card(code);
      if (data.type_ & TYPE_TOKEN) return;
      out.emplace_back(code, (data.type_ & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK)) != 0);
    };
    for (int player = 0; player < 2; ++player)
      for (uint8_t location : {LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE,
                               LOCATION_REMOVED, LOCATION_EXTRA}) {
        const int32_t length = OCG_QueryFieldCard(pduel_, player, location,
                                                  QUERY_CODE | QUERY_POSITION | QUERY_OVERLAY_CARD | QUERY_OWNER,
                                                  query_buf_);
        int32_t at = 0;
        while (at < length) {
          uint32_t size = 0;
          std::memcpy(&size, query_buf_ + at, 4);
          if (size == LEN_EMPTY) {
            at += 4;
            continue;
          }
          if (size < 24 || at + static_cast<int32_t>(size) > length)
            throw std::runtime_error("malformed card query record of " + std::to_string(size) + " bytes");
          uint32_t code = 0, materials = 0;
          int32_t owner = 0;
          std::memcpy(&code, query_buf_ + at + 8, 4);
          Card c = c_get_card(code);
          c.controler_ = query_buf_[at + 12];
          c.location_ = query_buf_[at + 13];
          c.sequence_ = query_buf_[at + 14];
          c.position_ = query_buf_[at + 15];
          std::memcpy(&materials, query_buf_ + at + 16, 4);
          if (24 + 4 * materials != size)
            throw std::runtime_error("card query record of " + std::to_string(size) + " bytes for " +
                                     std::to_string(materials) + " materials");
          std::memcpy(&owner, query_buf_ + at + 20 + 4 * materials, 4);
          if (owner == opponent && visible_code(viewer, c) != 0) add(code);
          at += static_cast<int32_t>(size);
        }
      }
    for (int player = 0; player < 2; ++player) {
      std::vector<uint32_t> words(4 * 64);
      const int32_t n = query_overlay_owners(pduel_, static_cast<uint8_t>(player), words.data(),
                                             static_cast<int32_t>(words.size()));
      if (n < 0) throw std::runtime_error("query_overlay_owners returned " + std::to_string(n));
      for (int32_t i = 0; i < n; ++i)
        if (static_cast<int>(words[4 * i + 3]) == opponent) add(words[4 * i + 2]);
    }
    return out;
  }

  std::map<std::string, int64_t> refusals_;  // ReplaceHidden refusals by reason (RefusalCounts)
 public:
  bool verify_refusals_ = false;                       // tests: digest the arena around each core call
  std::pair<uint64_t, uint64_t> refusal_digests_{0, 0};  // the last refused call's arena before and after
 private:

  static std::string LuaList(const std::vector<uint32_t> &values) {
    std::string out = "{";
    for (size_t i = 0; i < values.size(); ++i) out += (i ? "," : "") + std::to_string(values[i]);
    return out + "}";
  }
  // Runs a generated chunk in the duel's interpreter (the env's script reader serves it under ``name`` for this
  // call); a core log line is an error.
  void RunChunk(const std::string &name, const std::string &source) {
    core_script_errors().clear();
    script_override() = {name.c_str(), &source};
    const int32_t ok = preload_script(pduel_, name.c_str());
    script_override() = {};
    std::vector<std::string> errors;
    errors.swap(core_script_errors());
    if (!ok || !errors.empty())
      throw std::runtime_error("search chunk " + name + " failed" + (errors.empty() ? "" : ": " + errors.front()));
  }
};

}  // namespace duelenv
