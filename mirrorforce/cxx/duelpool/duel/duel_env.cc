#include "duel/duel_env.h"
#include "duel/announce_law.h"
#include "duel/scripted_driver.h"
#include "duel/client_driver.h"
#include "duel/search_api.h"
#include "duel/rollout_pool.h"
#include "envpool/py_envpool.h"

namespace py = pybind11;

namespace {

// The announce law over the module's card table (init_module must have run): the same function the env calls at
// MSG_ANNOUNCE_CARD, exported so the evaluation path, the deployment client and the oracle test call it directly.
duelenv::announce::Tables TablesOf(const std::vector<std::pair<uint32_t, int64_t>> &staples,
                                   const std::vector<py::dict> &library) {
  duelenv::announce::Tables tables;
  for (const auto &[code, count] : staples) {
    if (!tables.staples.emplace(code, count).second)
      throw std::runtime_error("the staple table lists a card twice");
  }
  for (const py::dict &row : library) {
    duelenv::announce::Recipe recipe;
    recipe.main = row["main"].cast<std::vector<uint32_t>>();
    recipe.extra = row["extra"].cast<std::vector<uint32_t>>();
    recipe.cluster = row["cluster"].cast<std::string>();
    recipe.format = row["format"].cast<std::string>();
    tables.library.push_back(std::move(recipe));
  }
  return tables;
}

py::dict LawIdentity(const duelenv::announce::Registration &law) {
  py::dict out;
  out["law"] = duelenv::announce::kLaw;
  out["belief_law"] = duelenv::announce::kBeliefLaw;
  out["cap"] = law.cap;
  out["room_format"] = law.room_format ? py::object(py::str(*law.room_format)) : py::object(py::none());
  out["staple_threshold"] = law.staple_threshold;
  out["tables_sha256"] = law.tables_sha256;
  return out;
}

// The process's announce law (announce_law.h Register): the env's announce prompts use it; a different second
// registration is refused. Returns its identity, which runs record in their checkpoint receipts.
py::dict RegisterAnnounceLaw(const std::vector<py::dict> &library,
                             const std::vector<std::pair<uint32_t, int64_t>> &staples, double staple_threshold,
                             int64_t cap, const std::optional<std::string> &room_format) {
  duelenv::announce::Registration law;
  law.tables = TablesOf(staples, library);
  law.cap = cap;
  law.room_format = room_format;
  law.staple_threshold = staple_threshold;
  return LawIdentity(duelenv::announce::Register(std::move(law)));
}

py::dict AnnounceCandidates(const std::vector<uint32_t> &opcodes, const std::vector<uint32_t> &seen_opponent,
                            const std::vector<uint32_t> &seen_own, const std::vector<uint32_t> &own_main,
                            const std::vector<uint32_t> &own_extra,
                            const std::vector<std::pair<uint32_t, int64_t>> &staples,
                            const std::vector<py::dict> &library, const std::optional<std::string> &room_format,
                            int64_t cap) {
  if (duelenv::card_ids_.empty()) throw std::runtime_error("announce_candidates needs init_module first");
  const duelenv::announce::Tables tables = TablesOf(staples, library);
  std::vector<uint32_t> card_table;
  card_table.reserve(duelenv::card_ids_.size());
  for (const auto &[code, id] : duelenv::card_ids_) card_table.push_back(code);
  auto card_id = [](uint32_t code) -> int64_t {
    const auto it = duelenv::card_ids_.find(code);
    return it == duelenv::card_ids_.end() ? 0 : it->second;
  };
  auto declarable = [&opcodes](uint32_t code) {
    const auto it = duelenv::cards_data_.find(code);
    return it != duelenv::cards_data_.end() && duelenv::evaluate_announce_card_filter(it->second, opcodes);
  };
  const auto result = duelenv::announce::Candidates(opcodes, seen_opponent, seen_own, own_main, own_extra, tables,
                                                    room_format, cap, card_table, card_id, declarable);
  py::dict tiers, truncated, out;
  const char *names[] = {"literal", "seen", "own_recipe", "staple", "belief"};
  for (size_t t = 0; t < 5; ++t) tiers[names[t]] = result.tiers[t];
  truncated["belief"] = result.truncated_belief;
  truncated[duelenv::announce::kEmptyUnion] = result.truncated_empty_union;
  out["law"] = duelenv::announce::kLaw;
  out["belief_law"] = duelenv::announce::kBeliefLaw;
  out["candidates"] = result.candidates;
  out["branch"] = result.empty_union ? duelenv::announce::kEmptyUnion : "union";
  out["truncated"] = truncated;
  out["tiers"] = tiers;
  return out;
}

// ---- scripted duels (scripted_driver.h) ----

duelenv::ScriptedDeal DealOf(const py::dict &deal) {
  static const std::set<std::string> keys = {"seed_words", "deck_orders", "extra", "start_lp", "start_hand",
                                             "draw_count", "duel_options"};
  for (const auto &item : deal)
    if (!keys.count(item.first.cast<std::string>()))
      throw std::runtime_error("unknown scripted deal field " + item.first.cast<std::string>());
  duelenv::ScriptedDeal out;
  out.seed_words = deal["seed_words"].cast<std::vector<uint32_t>>();
  const auto orders = deal["deck_orders"].cast<std::vector<std::vector<uint32_t>>>();
  const auto extra = deal["extra"].cast<std::vector<std::vector<uint32_t>>>();
  if (orders.size() != 2 || extra.size() != 2) throw std::runtime_error("a scripted deal has two decks");
  for (int p = 0; p < 2; ++p) {
    out.deck_orders[p] = orders[p];
    out.extra[p] = extra[p];
  }
  out.start_lp = deal["start_lp"].cast<int32_t>();
  out.start_hand = deal["start_hand"].cast<int32_t>();
  out.draw_count = deal["draw_count"].cast<int32_t>();
  out.duel_options = deal["duel_options"].cast<uint32_t>();
  return out;
}

// The env spec of a scripted duel: the defaults, self-play, and the observation sizes training uses.
duelenv::DuelEnvSpec ScriptedSpec(const py::dict &config) {
  auto conf = duelenv::DuelEnvSpec::kDefaultConfig;
  conf["play_mode"_] = std::string("self");
  for (const auto &item : config) {
    const auto key = item.first.cast<std::string>();
    const int value = item.second.cast<int>();
    if (key == "max_options") conf["max_options"_] = value;
    else if (key == "max_cards") conf["max_cards"_] = value;
    else if (key == "n_history_actions") conf["n_history_actions"_] = value;
    else if (key == "max_steps") conf["max_steps"_] = value;
    else if (key == "belief_labels") conf["belief_labels"_] = value != 0;
    else if (key == "public_opponent_recipe") conf["public_opponent_recipe"_] = value != 0;
    else if (key == "allow_unreviewed_public_effects") conf["allow_unreviewed_public_effects"_] = value != 0;
    else if (key == "export_both_seats") conf["export_both_seats"_] = value != 0;
    else if (key == "room_format") conf["room_format"_] = value;
    else if (key == "room_era") conf["room_era"_] = value;
    else if (key == "history_window") conf["history_window"_] = value;
    else if (key == "history_chunk") conf["history_chunk"_] = value;
    else if (key == "history_chunk_cap") conf["history_chunk_cap"_] = value;
    else if (key == "history_chunk_slots") conf["history_chunk_slots"_] = value;
    else throw std::runtime_error("unsupported scripted duel config key " + key);
  }
  return duelenv::DuelEnvSpec(conf.AllValues());
}

template <typename D>
py::array ToNumpy(const TArray<D> &array) {
  return ArrayToNumpyHelper<D>::Convert(array);
}

// The state keys of the player to move: every key but the privileged label: ones (``labels`` false), or only those.
py::dict StateDict(const duelenv::ScriptedDuel::State &state, bool labels) {
  const auto keys = duelenv::DuelEnvSpec::StateSpec::AllKeys();
  py::dict out;
  size_t i = 0;
  auto add = [&](const auto &array) {
    const std::string key = keys[i++];
    // privileged keys (label:, priv:) never join the observation
    const bool privileged = key.rfind("label:", 0) == 0 || key.rfind("priv:", 0) == 0;
    if (privileged == labels) out[py::str(key)] = ToNumpy(array);
  };
  std::apply([&](const auto &...array) { (add(array), ...); }, state.AllValues());
  return out;
}

py::dict StateKeys(duelenv::ScriptedDuel &duel, bool labels) {
  duel.RequireOwnView();
  return StateDict(duel.Observe(), labels);
}

py::dict Observation(duelenv::ScriptedDuel &duel) { return StateKeys(duel, false); }

// One zone of a ClientDuel's card view from Python rows (client_driver.h).
std::vector<std::optional<duelenv::DuelEnvImpl::ClientCard>> ClientCards(const py::sequence &rows) {
  std::vector<std::optional<duelenv::DuelEnvImpl::ClientCard>> cards;
  for (const auto &item : rows) {
    if (item.is_none()) {
      cards.emplace_back();
      continue;
    }
    const auto t = item.cast<py::sequence>();
    if (t.size() != 19) throw std::runtime_error("a client card row has 19 fields");
    duelenv::DuelEnvImpl::ClientCard c;
    c.code = t[0].cast<uint32_t>();
    c.controller = t[1].cast<uint8_t>();
    c.location = t[2].cast<uint8_t>();
    c.sequence = t[3].cast<uint8_t>();
    c.position = t[4].cast<uint8_t>();
    c.level = t[5].cast<uint32_t>();
    c.rank = t[6].cast<uint32_t>();
    c.attack = t[7].cast<int32_t>();
    c.defense = t[8].cast<int32_t>();
    c.equip = t[9].cast<uint32_t>();
    c.overlay = t[10].cast<std::vector<uint32_t>>();
    c.counters = t[11].cast<std::vector<std::pair<uint32_t, uint32_t>>>();
    c.owner = t[12].cast<uint32_t>();
    c.status = t[13].cast<uint32_t>();
    c.lscale = t[14].cast<uint32_t>();
    c.rscale = t[15].cast<uint32_t>();
    c.link = t[16].cast<uint32_t>();
    c.link_marker = t[17].cast<uint32_t>();
    c.stats_known = t[18].cast<bool>();
    cards.push_back(std::move(c));
  }
  return cards;
}

void ClientSetCards(duelenv::ClientDuel &duel, int player, int location, const py::list &rows) {
  duel.SetCards(player, location, ClientCards(rows));
}

// The explicit search-only boundary shares the Python wire validator. Default ctor/feed/set_cards never import it.
std::map<uint32_t, uint32_t> ClientInitializeCurrentRoot(duelenv::ClientDuel &duel, const py::dict &input) {
  const py::dict value = py::module_::import("mirrorforce.netduel.current_root_view")
                            .attr("CurrentRootView").attr("from_dict")(input).attr("to_dict")().cast<py::dict>();
  duelenv::ClientDuel::CurrentRootView view;
  view.root_id = value["root_id"].cast<std::string>();
  view.root_hash = value["root_hash"].cast<std::string>();
  view.hypothesis_hash = value["hypothesis_hash"].cast<std::string>();
  view.viewer = value["viewer"].cast<int>();
  view.lp = value["lp"].cast<std::array<int32_t, 2>>();
  view.turn = value["turn"].cast<int>();
  view.turn_player = value["turn_player"].cast<int>();
  view.phase = value["phase"].cast<int>();
  view.main = value["main"].cast<std::vector<uint32_t>>();
  view.extra = value["extra"].cast<std::vector<uint32_t>>();
  view.opponent_main = value["opponent_main"].cast<std::vector<uint32_t>>();
  view.opponent_extra = value["opponent_extra"].cast<std::vector<uint32_t>>();
  const auto cards = value["cards"].cast<py::list>();
  const std::array<int, 7> locations{LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE,
                                      LOCATION_REMOVED, LOCATION_EXTRA};
  const auto seed = value["public_seed"].cast<py::dict>();
  auto side = [&](int player) { return player == view.viewer ? 0 : 1; };
  auto place = [&](const py::handle &item) -> duelenv::public_effects::SeedPlace {
    if (item.is_none()) return {-1, 0, 0};
    auto p = item.cast<std::array<int, 3>>();
    p[0] = side(p[0]);
    return p;
  };
  std::map<std::tuple<int, int, int>, uint32_t> positioned;
  for (const auto &item : seed["positioned"].cast<py::list>()) {
    const auto r = item.cast<std::array<uint32_t, 4>>();
    positioned[{r[0], r[1], r[2]}] = r[3];
  }
  for (const auto &item : seed["reveals"].cast<py::list>()) {
    const auto r = item.cast<std::array<uint32_t, 4>>();
    if (r[1] != LOCATION_DECK && r[1] != LOCATION_EXTRA) positioned[{r[0], r[1], r[2]}] = r[3];
  }
  for (int player = 0; player < 2; ++player) {
    const auto zones = cards[player].cast<py::list>();
    for (size_t z = 0; z < locations.size(); ++z) {
      auto &zone = view.cards[{player, locations[z]}];
      zone = ClientCards(zones[z].cast<py::list>());
      for (const auto &card : zone) {
        if (!card) continue;
        mfenv::TrackerSeedCard t;
        t.side = side(player);
        t.location = locations[z];
        t.sequence = card->sequence;
        t.position = card->position;
        const auto known = positioned.find({player, t.location, t.sequence});
        t.code = known == positioned.end() ? card->code : known->second;
        for (const auto &[kind, count] : card->counters) t.counters[kind] = count;
        view.history.tracker.cards.push_back(t);
        for (size_t i = 0; i < card->overlay.size(); ++i) {
          mfenv::TrackerSeedCard material;
          material.side = t.side;
          material.location = t.location | LOCATION_OVERLAY;
          material.sequence = t.sequence;
          material.position = static_cast<int>(i);
          material.code = card->overlay[i];
          view.history.tracker.cards.push_back(material);
        }
        if (card->equip) {
          const uint32_t target = card->equip;
          view.history.status.equips.push_back({{t.side, t.location, t.sequence},
                                                {side(target & 255), static_cast<int>((target >> 8) & 255),
                                                 static_cast<int>((target >> 16) & 255)}});
        }
      }
    }
  }
  for (const auto &item : seed["card_metadata"].cast<py::list>()) {
    const auto r = item.cast<py::dict>();
    const auto at = place(r["place"]);
    auto found = std::find_if(view.history.tracker.cards.begin(), view.history.tracker.cards.end(),
                             [&](const auto &c) { return c.side == at[0] && c.location == at[1] && c.sequence == at[2]; });
    if (found == view.history.tracker.cards.end()) throw std::runtime_error("current card metadata has no place");
    found->arrival_turn = r["arrival_turn"].cast<int64_t>();
    found->arrival_kind = r["arrival_kind"].cast<int>();
    found->arrived_from = r["arrived_from"].cast<int>();
    found->activations = r["activations"].cast<int64_t>();
    found->attacks = r["attacks"].cast<int64_t>();
    found->hint_kind = r["hint_kind"].cast<int64_t>();
    found->hint_value = r["hint_value"].cast<int64_t>();
    for (const auto &[desc, count] : r["desc_hints"].cast<std::vector<std::pair<int64_t, int64_t>>>())
      found->desc_hints[desc] = count;
  }
  const auto counts = seed["turn_counts"].cast<std::array<std::array<int64_t, 6>, 2>>();
  const auto ledger = seed["turn_ledger"].cast<std::array<std::array<int64_t, 8>, 2>>();
  for (int p = 0; p < 2; ++p) {
    view.history.tracker.counts[side(p)] = counts[p];
    view.history.ledger[side(p)] = ledger[p];
  }
  for (const auto &item : seed["player_hints"].cast<py::list>()) {
    const auto r = item.cast<std::array<int64_t, 3>>();
    view.history.tracker.hints[side(static_cast<int>(r[0]))][r[1]] = r[2];
  }
  if (seed.contains("field_groups") && py::len(seed["field_groups"]) != 0)
    throw std::runtime_error("native root importer cannot consume unprojected field knowledge");
  for (const auto &item : seed["unpositioned"].cast<py::list>()) {
    const auto r = item.cast<std::array<int64_t, 3>>();
    if (r[0] == LOCATION_HAND) view.history.hand[r[1]] = r[2];
    else if (r[0] == LOCATION_DECK) view.history.deck[r[1]] = r[2];
    else if (r[0] == LOCATION_EXTRA) view.history.extra[r[1]] = r[2];
    else throw std::runtime_error("unsupported native root unpositioned location");
  }
  view.history.hand_group = seed["hand_group"].cast<std::vector<int>>();
  for (const auto &item : seed["reveals"].cast<py::list>()) {
    const auto r = item.cast<std::array<uint32_t, 4>>();
    view.reveals.push_back(r);
    if (r[0] != static_cast<uint32_t>(view.viewer)) {
      if (r[1] == LOCATION_DECK) view.history.shown_deck[r[2]] = r[3];
      if (r[1] == LOCATION_EXTRA) view.history.shown_extra[r[2]] = r[3];
    }
  }
  int64_t order = 0;
  for (const auto &item : seed["activations"].cast<py::list>()) {
    const auto r = item.cast<py::dict>();
    duelenv::history::Activation a;
    const auto turn = r["turn"].cast<std::array<int64_t, 2>>();
    const auto resolved = r["resolved"].cast<std::array<int64_t, 2>>();
    const auto duel = r["duel"].cast<std::array<int64_t, 2>>();
    for (int p = 0; p < 2; ++p) a.turn[side(p)] = turn[p], a.resolved[side(p)] = resolved[p], a.duel[side(p)] = duel[p];
    a.negated = r["negated"].cast<int64_t>();
    a.last_turn = r["last_turn"].cast<int64_t>();
    a.order = ++order;
    view.history.activations[{r["code"].cast<int64_t>(), r["desc"].cast<int64_t>()}] = a;
  }
  for (const auto &item : seed["chain"].cast<py::list>()) {
    const auto r = item.cast<py::dict>();
    duelenv::public_effects::SeedLink link;
    link.number = r["link"].cast<int>();
    link.side = side(r["player"].cast<int>());
    link.code = r["code"].cast<int64_t>();
    link.desc = r["desc"].cast<int64_t>();
    link.origin = place(r["origin"]);
    link.source = place(r["source"]);
    link.negated = r["negated"].cast<bool>();
    link.left = r["left"].cast<bool>();
    for (const auto &target : r["targets"].cast<py::list>()) link.targets.push_back(place(target));
    for (const auto &item : r["moved"].cast<py::list>()) {
      const auto moved = item.cast<py::dict>();
      link.moved.push_back({place(moved["place"]), moved["from_location"].cast<int>(), moved["control"].cast<bool>(),
                            place(moved["destination"])});
    }
    view.history.chain.emplace_back(link, r["state"].cast<int>());
    link.code = duelenv::history::ArtworkBase(static_cast<uint32_t>(link.code));
    view.history.status.links.push_back(std::move(link));
  }
  const auto ctx = seed["chain_context"].cast<py::dict>();
  view.history.guard_chain.links = ctx["links"].cast<int>();
  view.history.guard_chain.solving = ctx["solving"].cast<int64_t>();
  view.history.guard_chain.resolution = ctx["resolution"].cast<int64_t>();
  view.history.carry.chain_link = ctx["chain_link"].cast<int>();
  view.history.carry.settlement_link = ctx["settlement_link"].cast<int>();
  view.history.carry.in_damage_step = ctx["in_damage_step"].cast<bool>();
  view.history.status.resolving = view.history.guard_chain.resolution ? view.history.carry.chain_link : 0;
  for (const auto &item : seed["card_status"].cast<py::list>()) {
    const auto r = item.cast<py::dict>();
    view.history.status.cards.push_back({place(r["place"]), place(r["source"]), r["bits"].cast<uint8_t>(),
                                         r["turn"].cast<int>()});
  }
  for (const auto &item : seed["public_effects"].cast<py::list>()) {
    auto r = item.cast<std::array<int64_t, 5>>();
    r[0] = duelenv::history::ArtworkBase(static_cast<uint32_t>(r[0]));
    r[2] = side(static_cast<int>(r[2]));
    view.history.status.effects.push_back(r);
  }
  for (const auto &item : seed["field_origins"].cast<py::list>()) {
    const auto r = item.cast<std::array<int, 4>>();
    view.history.status.field_origins.push_back({{side(r[0]), r[1], r[2]}, side(r[3])});
  }
  duel.InitializeCurrentRoot(view);
  return duel.CurrentCardTypes();
}

py::dict ClientCurrentPublicRootSeed(const duelenv::ClientDuel &duel) {
  auto current = duel.CurrentPublicRootSeed();
  auto absolute = [&](int side) { return side == 0 ? duel.seat() : 1 - duel.seat(); };
  auto coordinates = [&](duelenv::public_effects::SeedPlace p) {
    if (p[0] >= 0) p[0] = absolute(p[0]);
    return p;
  };
  auto place = [&](const duelenv::public_effects::SeedPlace &p) -> py::object {
    return p[0] < 0 ? py::object(py::none()) : py::cast(coordinates(p));
  };
  py::list cards, effects, links;
  auto &status_cards = current.status.cards;
  std::sort(status_cards.begin(), status_cards.end(),
            [&](const auto &a, const auto &b) { return coordinates(a.place) < coordinates(b.place); });
  for (const auto &card : status_cards) {
    py::dict row;
    row["place"] = place(card.place);
    row["source"] = place(card.source);
    row["bits"] = card.bits;
    row["turn"] = card.turn;
    cards.append(row);
  }
  std::vector<std::array<duelenv::public_effects::SeedPlace, 2>> equips;
  for (const auto &[equip, host] : current.status.equips) equips.push_back({coordinates(equip), coordinates(host)});
  std::sort(equips.begin(), equips.end());
  std::vector<std::array<int, 4>> origins;
  for (const auto &[p, side] : current.status.field_origins) {
    const auto at = coordinates(p);
    origins.push_back({at[0], at[1], at[2], absolute(side)});
  }
  std::sort(origins.begin(), origins.end());
  for (auto effect : current.status.effects) {
    effect[2] = absolute(static_cast<int>(effect[2]));
    effects.append(py::cast(effect));
  }
  for (const auto &[link, state] : current.chain) {
    py::dict row;
    row["link"] = link.number;
    row["player"] = absolute(link.side);
    row["code"] = link.code;
    row["desc"] = link.desc;
    row["origin"] = place(link.origin);  // the broadcast CHAINING coordinate, not a current private slot
    row["source"] = place(link.source);
    row["state"] = state;
    row["negated"] = link.negated;
    row["left"] = link.left;
    py::list targets, moved;
    for (const auto &target : link.targets) targets.append(place(target));
    for (const auto &m : link.moved) {
      py::dict record;
      record["place"] = place(m.place);
      record["destination"] = place(m.destination);
      record["from_location"] = m.from_location;
      record["control"] = m.control;
      moved.append(record);
    }
    row["targets"] = targets;
    row["moved"] = moved;
    links.append(row);
  }
  py::dict context;
  context["links"] = current.guard_chain.links;
  context["solving"] = current.guard_chain.solving;
  context["resolution"] = current.guard_chain.resolution;
  context["chain_link"] = current.carry.chain_link;
  context["settlement_link"] = current.carry.settlement_link;
  context["in_damage_step"] = current.carry.in_damage_step;
  py::dict seed;
  seed["card_status"] = cards;
  seed["equips"] = py::cast(equips);
  seed["field_origins"] = py::cast(origins);
  seed["public_effects"] = effects;
  seed["chain"] = links;
  seed["chain_context"] = context;
  py::dict out;
  out["schema"] = "mirrorforce_current_public_root_seed/v1";
  out["source_viewer"] = duel.seat();
  out["turn"] = current.turn;
  out["turn_player"] = current.turn_player;
  out["phase"] = current.phase;
  out["public_seed"] = seed;
  return out;
}

// The rollout pool's live slots and their observations (rollout_pool.h): (meta arrays, one obs dict per slot).
py::tuple RolloutObserve(duelenv::RolloutPool &pool) {
  std::vector<duelenv::RolloutPool::State> states;
  std::vector<duelenv::RolloutPool::Live> live;
  {
    py::gil_scoped_release release;
    live = pool.Observe(&states);
  }
  const auto n = static_cast<py::ssize_t>(live.size());
  py::array_t<int32_t> slot(n), continuation(n), root_action(n), to_play(n), own_index(n);
  py::array_t<int64_t> root(n);
  py::array_t<bool> own(n), fresh(n), cut(n);
  for (py::ssize_t i = 0; i < n; ++i) {
    slot.mutable_at(i) = live[i].slot;
    root.mutable_at(i) = live[i].root;
    continuation.mutable_at(i) = live[i].continuation;
    root_action.mutable_at(i) = live[i].root_action;
    to_play.mutable_at(i) = live[i].to_play;
    own.mutable_at(i) = live[i].own;
    own_index.mutable_at(i) = live[i].own_index;
    fresh.mutable_at(i) = live[i].fresh;
    cut.mutable_at(i) = live[i].cut;
  }
  py::dict meta;
  meta["slot"] = slot;
  meta["root"] = root;
  meta["continuation"] = continuation;
  meta["root_action"] = root_action;
  meta["to_play"] = to_play;
  meta["own"] = own;
  meta["own_index"] = own_index;
  meta["fresh"] = fresh;
  meta["cut"] = cut;
  py::list obs;
  for (const auto &state : states) obs.append(StateDict(state, false));
  return py::make_tuple(meta, obs);
}

// The facts a history record carries (the client-stream test's columns; no trace index, which differs per stream).
py::dict FactDict(const mfenv::HistoryTokenRecord &r) {
  py::dict d;
  d["kind"] = r.kind;
  d["subtype"] = r.subtype;
  d["turn"] = r.turn;
  d["player_relative"] = r.player_relative;
  d["from_controller_relative"] = r.from_controller_relative;
  d["from_location"] = r.from_location;
  d["from_sequence"] = r.from_sequence;
  d["to_controller_relative"] = r.to_controller_relative;
  d["to_location"] = r.to_location;
  d["to_sequence"] = r.to_sequence;
  d["phase"] = r.phase;
  d["link"] = r.link;
  d["public_event_kind"] = r.public_event_kind;
  d["value"] = r.value;
  d["detail"] = r.detail;
  d["reason"] = r.reason;
  d["amount"] = r.amount;
  d["count"] = r.count;
  d["from_position"] = r.from_position;
  d["to_position"] = r.to_position;
  d["flags"] = r.flags;
  d["card_code"] = r.card_code;
  d["card_row"] = r.card_row;
  d["effect_row"] = r.effect_row;
  return d;
}

py::list FactList(const std::vector<mfenv::HistoryTokenRecord> &records) {
  py::list out;
  for (const auto &r : records) out.append(FactDict(r));
  return out;
}

// Each viewer's facts from one message stream read as a single interval (a whole duel's core stream, or one
// client's received stream), by the same mfenv code the env's history runs.
py::dict HistoryFacts(const std::vector<std::pair<int, py::bytes>> &stream) {
  if (duelenv::card_ids_.empty()) throw std::runtime_error("history_facts needs init_module first");
  std::vector<mfenv::Message> messages;
  messages.reserve(stream.size());
  for (const auto &[msg, payload] : stream) {
    const std::string bytes = payload;
    messages.push_back(mfenv::Message{msg, std::vector<uint8_t>(bytes.begin(), bytes.end())});
  }
  int turn = 0;
  mfenv::ChainCarry carry;
  const auto interval =
      mfenv::CarriedIntervalTokens(messages, 0, &turn, &carry, duelenv::history::CardRows(duelenv::card_ids_));
  py::dict out;
  for (int viewer = 0; viewer < 2; ++viewer) {
    const auto hit = interval.by_viewer.find(viewer);
    out[py::int_(viewer)] = hit == interval.by_viewer.end() ? py::list() : FactList(hit->second);
  }
  return out;
}

// The guards (guards.h) over a given sequence of prompts and choices, for tests of the law itself: each step's
// exclusions before its choice is recorded.
py::list GuardsSimulate(const std::vector<py::dict> &steps) {
  duelenv::guards::Guards guards;
  py::list out;
  for (const py::dict &step : steps) {
    duelenv::guards::Prompt p;
    p.player = step["player"].cast<int>();
    p.msg = step["msg"].cast<int>();
    p.turn = step["turn"].cast<int>();
    p.phase = step["phase"].cast<int>();
    p.chain.links = step["links"].cast<int>();
    p.chain.resolution = step["resolution"].cast<int64_t>();
    p.facts = step["facts"].cast<int64_t>();
    p.key = step["key"].cast<std::string>();
    p.activate = step["activate"].cast<std::vector<bool>>();
    py::dict row;
    if (duelenv::guards::Guarded(p)) {
      const auto e = guards.Exclusions(p);
      row["no_progress"] = std::vector<int>(e.no_progress.begin(), e.no_progress.end());
      row["cycle"] = std::vector<int>(e.cycle.begin(), e.cycle.end());
      row["kept_menu"] = e.kept_menu;
      guards.Record(p, step["choice"].cast<int>());
    } else {
      row["no_progress"] = std::vector<int>();
      row["cycle"] = std::vector<int>();
      row["kept_menu"] = false;
    }
    row["scope"] = duelenv::guards::Scope(p);
    out.append(row);
  }
  return out;
}

py::dict LayoutDict(const duelenv::SearchDuel::Layout &layout) {
  py::dict out;
  out["hand"] = layout.hand;
  out["deck"] = layout.deck;
  out["extra"] = layout.extra;
  py::list facedown;
  for (const auto &slot : layout.facedown) facedown.append(py::make_tuple(slot[0], slot[1], slot[2]));
  out["facedown"] = facedown;
  return out;
}

void PermuteHidden(duelenv::SearchDuel &duel, int player, int viewer, const std::vector<uint32_t> &hand,
                   const std::vector<uint32_t> &deck, const std::vector<std::array<uint32_t, 3>> &facedown,
                   const std::optional<std::vector<uint32_t>> &extra) {
  duelenv::SearchDuel::Layout target;
  target.hand = hand;
  target.deck = deck;
  target.facedown = facedown;
  if (extra) target.extra = *extra;
  duel.PermuteHidden(player, viewer, target, extra.has_value());
}

// The privileged stream (priv: keys): the god-view card rows and both players' hidden layouts. Never an input of
// the policy or the search (mirrorforce/agent/env/privileged.py refuses these keys there).
// The public world a particle sampler reads (search_api.h PublicWorld).
py::dict WorldDict(const duelenv::public_belief::World &w) {
  py::dict out;
  out["hand"] = w.hand;
  out["deck"] = w.deck;
  out["extra"] = w.extra;
  py::list facedown;
  for (const auto &slot : w.facedown) facedown.append(py::make_tuple(slot[0], slot[1], slot[2], slot[3]));
  out["facedown"] = facedown;
  out["decklist_public"] = w.decklist_public;
  out["pool_main"] = w.pool_main;
  out["pool_extra"] = w.pool_extra;
  py::list public_owned;
  for (const auto &[code, extra] : w.public_owned) public_owned.append(py::make_tuple(code, extra));
  out["public_owned"] = public_owned;
  out["unpositioned"] = w.unpositioned;
  out["hand_group"] = w.hand_group;
  out["own_deck"] = w.own_deck;
  out["own_deck_fixed"] = w.own_deck_fixed;
  out["types"] = w.types;
  return out;
}

py::dict World(duelenv::SearchDuel &duel, int viewer) { return WorldDict(duel.PublicWorld(viewer)); }

py::dict ClientWorld(const duelenv::ClientDuel &duel, const py::list &rows) {
  std::vector<std::array<uint32_t, 6>> owners;
  for (const auto &item : rows) {
    if (!(py::isinstance<py::list>(item) || py::isinstance<py::tuple>(item)))
      throw std::runtime_error("client public_world material owners are six public integers per row");
    const auto values = item.cast<py::sequence>();
    if (values.size() != 6) throw std::runtime_error("client public_world material owner row needs six integers");
    std::array<uint32_t, 6> row{};
    for (size_t i = 0; i < row.size(); ++i) {
      if (!py::isinstance<py::int_>(values[i]) || py::isinstance<py::bool_>(values[i]))
        throw std::runtime_error("client public_world material coordinates/owner must be non-boolean integers");
      row[i] = values[i].cast<uint32_t>();
    }
    owners.push_back(row);
  }
  return WorldDict(duel.PublicWorld(owners));
}

std::vector<std::pair<uint32_t, uint32_t>> DormantScan(const std::vector<uint32_t> &codes) {
  std::array<uint32_t, 8> seeds{1, 2, 3, 4, 5, 6, 7, 8};
  const intptr_t pduel = create_duel_v2(seeds.data());
  if (pduel == 0) throw std::runtime_error("scratch duel creation returned a null handle");
  try {
    const auto report = duelenv::dormant::Scan(pduel, codes);
    end_duel(pduel);
    return report;
  } catch (...) {
    end_duel(pduel);
    throw;
  }
}

void RegisterDormantTable(const std::vector<uint32_t> &codes, const std::string &sha256) {
  duelenv::dormant::Register({codes, sha256});
}

py::object DormantIdentity() {
  const auto law = duelenv::dormant::Registered();
  if (!law) return py::none();
  py::dict out;
  out["law"] = duelenv::dormant::kLaw;
  out["sha256"] = law->sha256;
  out["codes"] = law->codes;
  return out;
}

void ReplaceHidden(duelenv::SearchDuel &duel, int player, int viewer, const std::vector<uint32_t> &hand,
                   const std::vector<uint32_t> &deck, const std::vector<std::array<uint32_t, 3>> &facedown,
                   const std::optional<std::vector<uint32_t>> &extra, const std::vector<uint32_t> &recipe_main,
                   const std::vector<uint32_t> &recipe_extra) {
  duelenv::SearchDuel::Layout target;
  target.hand = hand;
  target.deck = deck;
  target.facedown = facedown;
  if (extra) target.extra = *extra;
  duel.ReplaceHidden(player, viewer, target, extra.has_value(), recipe_main, recipe_extra);
}

py::dict Privileged(duelenv::SearchDuel &duel) {
  std::vector<size_t> shape;
  const std::vector<uint8_t> bytes = duel.PrivilegedCards(&shape);
  py::array_t<uint8_t> cards(shape);
  std::memcpy(cards.mutable_data(), bytes.data(), bytes.size());
  py::dict out;
  out["priv:cards_"] = cards;
  out["priv:layout"] = py::make_tuple(LayoutDict(duel.HiddenLayout(0)), LayoutDict(duel.HiddenLayout(1)));
  return out;
}

py::object Prompt(duelenv::ScriptedDuel &duel) {
  if (duel.finished()) return py::none();
  duel.Observe();  // fills each row's card id, as training's observation does
  std::unordered_map<int64_t, uint32_t> codes;
  for (const auto &[code, id] : duelenv::card_ids_) codes.emplace(id, code);
  py::list rows;
  for (const auto &a : duel.actions()) {
    py::dict row;
    row["act"] = static_cast<int>(a.act_);
    row["phase"] = static_cast<int>(a.phase_);
    row["finish"] = a.finish_;
    if (a.cid_ != 0 && !codes.count(a.cid_)) throw std::runtime_error("a menu row names an unknown card id");
    row["code"] = a.cid_ == 0 ? 0u : codes.at(a.cid_);
    row["effect"] = a.effect_;
    row["position"] = static_cast<int>(a.position_);
    row["number"] = static_cast<int>(a.number_);
    row["place"] = static_cast<int>(a.place_);
    row["attribute"] = static_cast<int>(a.attribute_);
    row["spec"] = a.spec_;
    rows.append(row);
  }
  return py::make_tuple(duel.player(), duel.prompt_msg(), rows);
}

}  // namespace

using DuelEnvSpec = PyEnvSpec<duelenv::DuelEnvSpec>;
using DuelEnvPool = PyEnvPool<duelenv::DuelEnvPool>;

PYBIND11_MODULE(duel_native, m) {
  REGISTER(m, DuelEnvSpec, DuelEnvPool)

  py::register_exception<duelenv::public_effects::PublicRootSeedError>(m, "CurrentPublicRootSeedError", PyExc_RuntimeError);

  m.def("init_module", &duelenv::init_module);
  m.attr("step_limit_law") = "step_limit_timeout_loss/v1";  // the env's step-limit result (duel_env.h step)
  m.attr("guard_laws") = py::make_tuple(duelenv::guards::kNoProgressLaw, duelenv::guards::kCommandCycleLaw,
                                      duelenv::kIllegalActivationLaw);
  m.def("announce_candidates", &AnnounceCandidates, py::arg("opcodes"), py::kw_only(), py::arg("seen_opponent"),
        py::arg("seen_own"), py::arg("own_main"), py::arg("own_extra"), py::arg("staples"), py::arg("library"),
        py::arg("room_format"), py::arg("cap"),
        "The announce-card candidate law (announce_law.h) over the module's card table.");
  m.def("register_announce_law", &RegisterAnnounceLaw, py::kw_only(), py::arg("library"), py::arg("staples"),
        py::arg("staple_threshold"), py::arg("cap"), py::arg("room_format"),
        "Register the process's announce law (announce_law.h); returns its identity.");
  m.def("dormant_scan", &DormantScan, py::arg("codes"),
        "Each code loaded twice in one scratch duel, in code order: both loads' duel-level report bits (dormant_law.h)");
  m.def("register_dormant_table", &RegisterDormantTable, py::kw_only(), py::arg("codes"), py::arg("sha256"),
        "Registers the run's dormant identities (one table per process)");
  m.def("dormant_identity", &DormantIdentity, "The registered dormant table (law, sha256, codes) or None");
  m.def("announce_law_identity", [] { return LawIdentity(duelenv::announce::Registered()); },
        "The registered announce law's identity (refuses when none is registered).");
  m.def("guards_simulate", &GuardsSimulate, py::arg("steps"),
        "The no-progress guards over a sequence of prompts and choices (tests of guards.h).");
  py::class_<duelenv::RolloutPool>(m, "RolloutPool",
                                   "Continuation pool of search: K continuations of collection-game roots, each "
                                   "from a root action, to a leaf depth (rollout_pool.h)")
      .def(py::init([](const py::dict &config, int slots, int threads, uint64_t seed) {
             return std::make_unique<duelenv::RolloutPool>(ScriptedSpec(config), slots, threads, seed);
           }),
           py::arg("config"), py::arg("slots"), py::arg("threads"), py::arg("seed"))
      .def("submit",
           [](duelenv::RolloutPool &pool, uint32_t seed, const std::array<std::vector<uint32_t>, 2> &main_decks,
              const std::array<std::vector<uint32_t>, 2> &extra_decks, const std::vector<int> &actions,
              uint64_t stream_hash, int to_play, const std::string &play_gen, const std::vector<int> &root_actions,
              int depth, int width) {
             duelenv::RolloutPool::Root root;
             root.deal.seed_words = {seed};
             root.deal.deck_orders = main_decks;
             root.deal.extra = extra_decks;
             root.deal.start_lp = 8000;
             root.deal.start_hand = 5;
             root.deal.draw_count = 1;
             root.deal.duel_options = 5u << 16;
             root.actions = actions;
             root.stream_hash = stream_hash;
             root.to_play = to_play;
             root.play_gen = play_gen;
             root.root_actions = root_actions;
             root.depth = depth;
             root.width = width;
             return pool.Submit(std::move(root));
           },
           py::kw_only(), py::arg("seed"), py::arg("main_decks"), py::arg("extra_decks"), py::arg("actions"),
           py::arg("stream_hash"), py::arg("to_play"), py::arg("play_gen"), py::arg("root_actions"), py::arg("depth"),
           py::arg("width"),
           "Queue a root (a collection env's root_record fields) with one root action per continuation (-1 none), "
           "the leaf depth in the root player's decisions (0: the game's end) and the slots it may occupy at once; "
           "returns its id")
      .def("observe", &RolloutObserve, "Fill idle slots, then (meta, observations) of every live slot")
      .def("step",
           [](duelenv::RolloutPool &pool, const std::vector<int> &slots, const std::vector<int> &actions) {
             py::gil_scoped_release release;
             pool.Step(slots, actions);
           },
           py::arg("slots"), py::arg("actions"), "Step live slots by menu index (a cut slot's action is ignored)")
      .def("results",
           [](duelenv::RolloutPool &pool) {
             py::list out;
             for (const auto &r : pool.TakeResults()) {
               py::dict d;
               d["root"] = r.root;
               d["continuation"] = r.continuation;
               d["seed"] = r.seed;
               d["root_action"] = r.root_action;
               d["depth"] = r.depth;
               d["root_player"] = r.root_player;
               d["winner"] = r.winner;
               d["win_reason"] = r.win_reason;
               d["decisions"] = r.decisions;
               d["own_decisions"] = r.own_decisions;
               d["truncated"] = r.truncated;
               d["error"] = r.error;
               out.append(d);
             }
             return out;
           },
           "Finished continuations since the last call")
      .def("root_log",
           [](duelenv::RolloutPool &pool) {
             py::list out;
             for (const auto &r : pool.TakeRootLog()) {
               py::dict d;
               d["root"] = r.root;
               d["replay_steps"] = r.replay_steps;
               d["replay"] = r.replay;
               d["snapshot"] = r.snapshot;
               d["refused"] = r.refused;
               out.append(d);
             }
             return out;
           },
           "Prepared roots since the last call: replay length and costs, or the refusal")
      .def("timings",
           [](const duelenv::RolloutPool &pool) {
             const auto t = pool.timings();
             py::dict d;
             d["replay"] = t.replay;
             d["snapshot"] = t.snapshot;
             d["restore"] = t.restore;
             d["reshuffle"] = t.reshuffle;
             d["root_action"] = t.root_action;
             d["observe"] = t.observe;
             d["step"] = t.step;
             d["clones"] = t.clones;
             d["continuations"] = t.continuations;
             d["steps"] = t.steps;
             d["replay_steps"] = t.replay_steps;
             d["errors"] = t.errors;
             return d;
           })
      .def_property_readonly("queued", &duelenv::RolloutPool::queued)
      .def_property_readonly("live", &duelenv::RolloutPool::live);
  m.def("history_facts", &HistoryFacts, py::arg("messages"),
        "Each viewer's public facts from one message stream (history_obs.h's records), keyed by viewer.");
  m.attr("live_duel_capacity") = duel_live_capacity();
  m.attr("card_view_law") = duelenv::kCardViewLaw;
  m.attr("pending_source_law") = duelenv::kPendingSourceLaw;
  m.attr("placement_ref_law") = duelenv::kPlacementRefLaw;
  m.attr("unpositioned_law") = duelenv::kUnpositionedLaw;
  m.attr("current_root_unpositioned_import_law") = "three-zone-projection-only/v1";
  m.attr("public_effects_law") = duelenv::kPublicEffectsLaw;
  m.attr("public_effects_table_sha256") = duelenv::public_effects::kTableSha256;
  m.attr("public_effects_reviewed") = std::vector<uint32_t>(std::begin(duelenv::public_effects::kReviewed),
                                                            std::end(duelenv::public_effects::kReviewed));
  m.attr("observation_laws") = py::make_tuple(duelenv::kCardViewLaw, duelenv::kPendingSourceLaw,
                                              duelenv::kUnpositionedLaw, duelenv::kPublicEffectsLaw,
                                              duelenv::kPlacementRefLaw);

  m.attr("history_window_rows") = duelenv::history::kEvents;
  m.attr("history_chunk_rows") = duelenv::history::kChunkRows;
  m.attr("history_chunk_cap") = duelenv::history::kChunkCap;
  m.attr("history_chunk_slots") = duelenv::history::kChunkSlots;
  m.def("own_desc_index", &duelenv::history::OwnDescIndex, py::arg("code"), py::arg("desc"),
        "The description index history rows write for a card's description (15 when it is another card's; tests).");
  py::class_<duelenv::ClientDuel>(m, "ClientDuel",
                                  "Client mode: one seat's observation from its received stream (client_driver.h)")
      .def(py::init([](int seat, std::vector<uint32_t> main, std::vector<uint32_t> extra, const py::dict &config,
                      std::optional<std::vector<uint32_t>> opponent_main,
                      std::optional<std::vector<uint32_t>> opponent_extra) {
             return std::make_unique<duelenv::ClientDuel>(ScriptedSpec(config), seat, std::move(main),
                                                          std::move(extra), std::move(opponent_main),
                                                          std::move(opponent_extra));
           }),
           py::arg("seat"), py::arg("main"), py::arg("extra"), py::arg("config") = py::dict(),
           py::arg("opponent_main") = py::none(), py::arg("opponent_extra") = py::none())
      .def("set_cards", &ClientSetCards, py::arg("player"), py::arg("location"), py::arg("cards"),
           "One zone of the client's card view: per slot None or (code, controller, location, sequence, position, "
           "level, rank, attack, defense, equip, overlay codes, [(counter type, count)], owner, status, lscale, "
           "rscale, link, link marker, stats known)")
      .def("initialize_current_root", &ClientInitializeCurrentRoot, py::arg("view"),
           "Initialize a fresh observer from a legal current hypothetical root; no opening/history replay")
      .def("current_public_root_seed", &ClientCurrentPublicRootSeed,
           "Read only current common-public chain/effects/field relations in absolute seats, without private history")
      .def("feed",
           [](duelenv::ClientDuel &d, int msg, const py::bytes &payload) -> py::object {
             const std::string raw = payload;
             const auto out = d.Feed(msg, std::vector<uint8_t>(raw.begin(), raw.end()));
             if (!out) return py::none();
             return py::bytes(reinterpret_cast<const char *>(out->data()), out->size());
           },
           py::arg("msg"), py::arg("payload"), "One received message: the response bytes the env answered, or None")
      .def("prompt",
           [](duelenv::ClientDuel &d) -> py::object { return d.pending() ? Prompt(d) : py::none(); },
           "None, or (player, msg, menu rows) of the pending decision")
      .def("observation", [](duelenv::ClientDuel &d) { return Observation(d); },
           "Every state key for the seat (as ScriptedDuel.observation)")
      .def("step",
           [](duelenv::ClientDuel &d, int index) -> py::object {
             const auto out = d.Decide(index);
             if (!out) return py::none();
             return py::bytes(reinterpret_cast<const char *>(out->data()), out->size());
           },
           py::arg("index"), "The policy's row: the response bytes, or None while a selection collects")
      .def("forced_count", &duelenv::ClientDuel::forced_count)
      .def_property_readonly("pending", &duelenv::ClientDuel::pending)
      .def("public_world", &ClientWorld,
           py::arg("material_owners"),
           "Read-only public belief context of this client's pending decision; exact public material owners required")
      .def("clone", &duelenv::ClientDuel::Clone, "An independent copy at this point of the stream")
      .def("step_response",
           [](duelenv::ClientDuel &d, const py::bytes &raw) {
             const std::string bytes = raw;
             const auto out = d.DecideResponse(std::vector<uint8_t>(bytes.begin(), bytes.end()));
             return py::bytes(reinterpret_cast<const char *>(out.data()), out.size());
           },
           py::arg("response"),
           "The pending decision answered by its response bytes: the one menu path (a row, or a selection's "
           "sub-choices) whose response they are, applied as step would; returns the bytes")
      .def("response_path",
           [](const duelenv::ClientDuel &d, const py::bytes &raw) {
             const std::string bytes = raw;
             return d.ResponsePath(std::vector<uint8_t>(bytes.begin(), bytes.end()));
           },
           py::arg("response"),
           "Read-only: the unique menu-row path for response bytes, found on a clone; no history is consumed")
      .def_property_readonly("seat", &duelenv::ClientDuel::seat);

  py::class_<duelenv::ScriptedDuel>(m, "ScriptedDuel",
                                    "One env driven through an explicit deal (scripted_driver.h).")
      .def(py::init([](const py::dict &deal, const py::dict &config) {
             return std::make_unique<duelenv::ScriptedDuel>(ScriptedSpec(config), DealOf(deal));
           }),
           py::arg("deal"), py::arg("config") = py::dict())
      .def("start", &duelenv::ScriptedDuel::Start)
      .def("prompt", &Prompt, "None when the duel is over, else (player, msg, menu rows) of the prompt")
      .def("step", &duelenv::ScriptedDuel::Step, py::arg("index"))
      .def("respond",
           [](duelenv::ScriptedDuel &d, const py::bytes &raw) {
             const std::string bytes = raw;
             d.Respond(std::vector<uint8_t>(bytes.begin(), bytes.end()));
           },
           py::arg("response"),
           "Raw response bytes for the current prompt, bypassing the menu (a recorded game's player); that "
           "player's own-choice bookkeeping is skipped")
      .def("repro_record",
           [](const duelenv::ScriptedDuel &d, const std::string &error) { return d.repro_json(error.c_str()); },
           py::arg("error"), "The env's fatal repro record (ENV_FATAL_REPRO) for the current game, as JSON text")
      .def("observation", &Observation, "Every state key for the player to move but the privileged label: keys")
      .def("labels", [](duelenv::ScriptedDuel &d) { return StateKeys(d, true); },
           "Privileged: the label: keys for the player to move (filled with the belief_labels config)")
      .def("messages",
           [](const duelenv::ScriptedDuel &d) {
             py::list out;
             for (const auto &m : d.messages())
               out.append(py::make_tuple(m.msg, py::bytes(reinterpret_cast<const char *>(m.payload.data()),
                                                          m.payload.size())));
             return out;
           })
      .def("message_count", [](const duelenv::ScriptedDuel &d) { return d.messages().size(); })
      .def("messages_since",
           [](const duelenv::ScriptedDuel &d, size_t start) {
             py::list out;
             const auto &messages = d.messages();
             for (size_t i = start; i < messages.size(); ++i)
               out.append(py::make_tuple(messages[i].msg,
                                         py::bytes(reinterpret_cast<const char *>(messages[i].payload.data()),
                                                   messages[i].payload.size())));
             return out;
           },
           py::arg("start"), "The core messages from index ``start`` on (lanes that read the stream as it grows)")
      .def("responses",
           [](const duelenv::ScriptedDuel &d) {
             py::list out;
             for (const auto &r : d.responses())
               out.append(py::bytes(reinterpret_cast<const char *>(r.data()), r.size()));
             return out;
           })
      .def("facts", [](const duelenv::ScriptedDuel &d, int viewer) { return FactList(d.facts(viewer)); },
           py::arg("viewer"))
      .def("seen",
           [](const duelenv::ScriptedDuel &d, int viewer) {
             const auto &seen = d.seen(viewer);
             return py::make_tuple(std::vector<uint32_t>(seen[0].begin(), seen[0].end()),
                                   std::vector<uint32_t>(seen[1].begin(), seen[1].end()));
           },
           py::arg("viewer"), "(own, opponent's) card codes the viewer's facts have shown (the announce seen sets)")
      .def("engine_equips", &duelenv::ScriptedDuel::EngineEquips,
           "Audit only (privileged): (controller, location, sequence, target info location) of each equipped card")
      .def("audit_visible",
           [](duelenv::ScriptedDuel &d) {
             const auto a = d.AuditVisible();
             py::dict out;
             out["shown"] = a.shown;
             out["hand_public"] = a.hand_public;
             out["hand_known"] = a.hand_known;
             out["extra_faceup"] = a.extra_faceup;
             out["revealed"] = a.revealed;
             out["unpositioned"] = a.unpositioned;
             out["mismatches"] = a.mismatches;
             return out;
           },
           "Audit only: the opponent cards the player to move is shown, and the identities it knows without their "
           "positions, against the engine's identities")
      .def("hand_group", [](const duelenv::ScriptedDuel &d, int viewer) { return d.hand_group(viewer); },
           py::arg("viewer"), "The opponent's hand places in the viewer's shuffled group (its hand bound holds there)")
      .def("unpositioned", [](const duelenv::ScriptedDuel &d, int viewer) { return d.unpositioned(viewer); },
           py::arg("viewer"), "(location, code) -> copies of the opponent's cards the viewer knows without positions")
      .def_property_readonly("finished", &duelenv::ScriptedDuel::finished)
      .def_property_readonly("winner", &duelenv::ScriptedDuel::winner);
  py::class_<duelenv::SearchDuel::Snapshot, std::shared_ptr<duelenv::SearchDuel::Snapshot>>(
      m, "SearchSnapshot", "A decision point of a SearchDuel: the core arena copy and the env state")
      .def_property_readonly("arena_bytes", &duelenv::SearchDuel::Snapshot::extent);
  py::class_<duelenv::SearchDuel, duelenv::ScriptedDuel>(
      m, "SearchDuel", "A scripted duel with the engine search primitives (search_api.h)")
      .def(py::init([](const py::dict &deal, const py::dict &config, bool keep_history) {
             return std::make_unique<duelenv::SearchDuel>(ScriptedSpec(config), DealOf(deal), keep_history);
           }),
           py::arg("deal"), py::arg("config") = py::dict(), py::arg("keep_history") = false)
      .def("take", &duelenv::SearchDuel::Take, "Snapshot this decision point")
      .def("restore", &duelenv::SearchDuel::Restore, py::arg("snapshot"), "Return to a snapshot of this duel")
      .def("hidden_layout",
           [](duelenv::SearchDuel &d, int player) { return LayoutDict(d.HiddenLayout(player)); }, py::arg("player"),
           "Privileged: a player's true hidden cards")
      .def("permute_hidden", &PermuteHidden, py::arg("player"), py::kw_only(), py::arg("viewer"), py::arg("hand"),
           py::arg("deck"), py::arg("facedown"), py::arg("extra") = py::none(),
           "Rewrite a player's hidden cards; refused unless it keeps what the viewer's observation shows")
      .def("set_generator", &duelenv::SearchDuel::SetGenerator, py::arg("state"),
           "Play continues from an env generator state (a root record's play_gen; replaying a collection game)")
      .def("reshuffle_future", &duelenv::SearchDuel::ReshuffleFuture, py::arg("seed"),
           "New deck orders for both players and a new core random sequence")
      .def("replace_hidden", &ReplaceHidden, py::arg("player"), py::kw_only(), py::arg("viewer"), py::arg("hand"),
           py::arg("deck"), py::arg("facedown"), py::arg("extra") = py::none(), py::arg("recipe_main"),
           py::arg("recipe_extra"),
           "Give a player's hidden places other identities in place and install the particle's recipe (needs a "
           "registered dormant table); refused unless it keeps what the viewer sees and the recipe is what the player "
           "then owns; the player's own view is refused afterwards until a restore")
      .def("own_recipe_rows",
           [](duelenv::SearchDuel &duel, int player) {
             const auto bytes = duel.OwnRecipeRows(player);
             py::array_t<uint8_t> rows({duelenv::kRecipeRows, duelenv::kOwnRecipeWidth});
             std::memcpy(rows.mutable_data(), bytes.data(), bytes.size());
             return rows;
           },
           py::arg("player"), "Tests: a player's obs:own_recipe_ rows (a replaced player's installed recipe)")
      .def("unseen_own", [](duelenv::SearchDuel &duel, int viewer) { return duel.UnseenOwn(viewer); },
           py::arg("viewer"),
           "Tests and census: the viewer's own cards hidden in the opponent's control that its stream does not show, "
           "(main-deck kind, extra-deck kind) code counts")
      .def("refusal_counts", &duelenv::SearchDuel::RefusalCounts, py::arg("reset") = false,
           "replace_hidden refusals by reason")
      .def("arena_digest", &duelenv::SearchDuel::ArenaDigest, "A digest of the duel's whole core arena")
      .def("collect_garbage", &duelenv::SearchDuel::CollectGarbage, "A full collection of the duel's Lua heap")
      .def_readwrite("verify_refusals", &duelenv::SearchDuel::verify_refusals_,
                     "Tests: digest the core arena right around each replace_hidden call")
      .def_readonly("refusal_digests", &duelenv::SearchDuel::refusal_digests_,
                    "The last refused replace_hidden call's arena digests, before and after")
      .def("hidden_blockers", &duelenv::SearchDuel::HiddenBlockers, py::arg("player"),
           "Why each hidden card of a player could not change identity in place (bits; 0 when it could)")
      .def("effect_info",
           [](duelenv::SearchDuel &d) {
             const auto bytes = d.EffectInfo();
             return py::bytes(reinterpret_cast<const char *>(bytes.data()), bytes.size());
           },
           "Privileged: the core's registered-effect dump (query_effect_info)")
      .def("public_world", &World, py::arg("viewer"),
           "What a particle sampler may read: hidden places, known identities, own deck cards, the public pool")
      .def("privileged", &Privileged, "The priv: stream (god-view card rows, hidden layouts)");
}
