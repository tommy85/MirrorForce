#ifndef DUELPOOL_DUEL_DUEL_ENV_H_
#define DUELPOOL_DUEL_DUEL_ENV_H_

// clang-format off
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <memory>
#include <type_traits>
#include <unistd.h>


#include <fmt/core.h>
#include <fmt/ranges.h>
#include <SQLiteCpp/SQLiteCpp.h>
#include <SQLiteCpp/VariadicBind.h>
#include <ankerl/unordered_dense.h>
#include <unordered_set>

#include "envpool/BS_thread_pool.h"

#include "envpool/async_envpool.h"
#include "envpool/env.h"

#include "ocgcore/common.h"
#include "ocgcore/card_data.h"
#include "ocgcore/ocgapi.h"

extern "C" {
// R4 snapshot exports of the core (ocgapi.cpp): the duel's whole arena, copied back to the same addresses
void *duel_snapshot(intptr_t pduel);
int32_t duel_rollback(intptr_t pduel, void *snap);
void duel_snapshot_free(void *snap);
int64_t duel_arena_extent(intptr_t pduel);
}

#include "duel/announce_law.h"
#include "duel/dormant_law.h"
#include "duel/history_obs.h"

// The maintained engine core's common.h dropped the short integer aliases of the
// upstream core (only `byte` remains); the environment code still uses them.
using uint64 = unsigned long long;
using uint32 = unsigned int;
using uint16 = unsigned short;
using uint8 = unsigned char;
using int64 = long long;
using int32 = int;
using int16 = short;
using int8 = signed char;

// clang-format on

namespace duelenv {

inline std::vector<std::vector<int>> combinations(int n, int r) {
  std::vector<std::vector<int>> combs;
  std::vector<bool> m(n);
  std::fill(m.begin(), m.begin() + r, true);

  do {
    std::vector<int> cs;
    cs.reserve(r);
    for (int i = 0; i < n; ++i) {
      if (m[i]) {
        cs.push_back(i);
      }
    }
    combs.push_back(cs);
  } while (std::prev_permutation(m.begin(), m.end()));

  return combs;
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind, int i,
                   int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  if (i == n - 1) {
    return r == 1 || (w[ind[i]] == r);
  }
  return sum_to(w, ind, i + 1, r - 1) || sum_to(w, ind, i + 1, r - w[ind[i]]);
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind,
                   int r) {
  return sum_to(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
combinations_with_weight(const std::vector<int> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

inline std::vector<std::vector<int>> tribute_combinations(
    const std::vector<int> &weights, int minimum_value,
    int maximum_cards) {
  const int count = static_cast<int>(weights.size());
  const int begin = minimum_value <= 0 ? 0 : 1;
  const int end = std::min(count, std::max(0, maximum_cards));
  std::vector<std::vector<int>> results;

  for (int size = begin; size <= end; ++size) {
    for (const auto &combination : combinations(count, size)) {
      int value = 0;
      for (const auto index : combination) {
        value += weights[index];
      }
      if (value >= minimum_value) {
        results.push_back(combination);
      }
    }
  }
  return results;
}

// Maintained-core protocol (upstream engine-core #493): a sum parameter is two
// 16-bit operands, unless the high operand has bit 0x8000 set, in which case
// the low 31 bits are a single operand (field::get_sum_params).
inline void decode_sum_param(uint32_t param, int &value1, int &value2) {
  value1 = param & 0xffff;
  value2 = (param >> 16) & 0xffff;
  if (value2 & 0x8000) {
    value1 = static_cast<int>(param & 0x7fffffff);
    value2 = 0;
  }
}

inline std::vector<int> sum_param_values(uint32_t param) {
  std::vector<int> values;
  int value1, value2;
  decode_sum_param(param, value1, value2);
  if (value1 > 0) {
    values.push_back(value1);
  }
  if (value2 > 0 && value2 != value1) {
    values.push_back(value2);
  }
  return values;
}

inline bool sum_params_exact(
    const std::vector<uint32_t> &params, int index, int remaining) {
  if (index == static_cast<int>(params.size())) {
    return remaining == 0;
  }
  if (remaining <= 0) {
    return false;
  }
  for (const auto value : sum_param_values(params[index])) {
    if (value <= remaining &&
        sum_params_exact(params, index + 1, remaining - value)) {
      return true;
    }
  }
  return false;
}

inline bool sum_params_exact(
    const std::vector<uint32_t> &params, int target) {
  return !params.empty() && sum_params_exact(params, 0, target);
}

inline bool sum_params_at_least_minimal(
    const std::vector<uint32_t> &params, int target) {
  if (params.empty()) {
    return false;
  }
  int64_t minimum_sum = 0;
  int64_t maximum_sum = 0;
  int minimum_value = 0x7fffffff;
  for (const auto param : params) {
    int value1, value2;
    decode_sum_param(param, value1, value2);
    const int current_minimum =
        value2 > 0 && value2 < value1 ? value2 : value1;
    minimum_sum += current_minimum;
    maximum_sum += std::max(value1, value2);
    minimum_value = std::min(minimum_value, current_minimum);
  }
  return maximum_sum >= target &&
         minimum_sum - minimum_value < target;
}

inline std::vector<std::vector<int>> combinations_with_sum_params(
    const std::vector<uint32_t> &must_params,
    const std::vector<uint32_t> &select_params, int target,
    int minimum_count, int maximum_count, bool at_least) {
  const int count = static_cast<int>(select_params.size());
  const int begin = at_least ? 0 : std::max(0, minimum_count);
  const int end = at_least
      ? count
      : std::min(count, std::max(minimum_count, maximum_count));
  std::vector<std::vector<int>> results;

  for (int size = begin; size <= end; ++size) {
    for (const auto &combination : combinations(count, size)) {
      std::vector<uint32_t> params = must_params;
      params.reserve(must_params.size() + combination.size());
      for (const auto index : combination) {
        params.push_back(select_params[index]);
      }
      const bool valid = at_least
          ? sum_params_at_least_minimal(params, target)
          : sum_params_exact(params, target);
      if (valid) {
        results.push_back(combination);
      }
    }
  }
  return results;
}

inline bool evaluate_announce_card_filter(
    const card_data &card, const std::vector<uint32_t> &opcodes) {
  std::vector<int64_t> stack;
  for (const auto opcode : opcodes) {
    auto pop_binary = [&stack]() {
      const auto rhs = stack.back();
      stack.pop_back();
      const auto lhs = stack.back();
      stack.pop_back();
      return std::pair<int64_t, int64_t>{lhs, rhs};
    };
    switch (opcode) {
    case OPCODE_ADD:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        stack.push_back(lhs + rhs);
      }
      break;
    case OPCODE_SUB:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        stack.push_back(lhs - rhs);
      }
      break;
    case OPCODE_MUL:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        stack.push_back(lhs * rhs);
      }
      break;
    case OPCODE_DIV:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        if (rhs == 0) {
          return false;
        }
        stack.push_back(lhs / rhs);
      }
      break;
    case OPCODE_AND:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        stack.push_back(lhs && rhs);
      }
      break;
    case OPCODE_OR:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        stack.push_back(lhs || rhs);
      }
      break;
    case OPCODE_NEG:
      if (!stack.empty()) {
        stack.back() = -stack.back();
      }
      break;
    case OPCODE_NOT:
      if (!stack.empty()) {
        stack.back() = !stack.back();
      }
      break;
    case OPCODE_ISCODE:
      if (!stack.empty()) {
        stack.back() = card.code == static_cast<uint32_t>(stack.back());
      }
      break;
    case OPCODE_ISSETCARD:
      if (!stack.empty()) {
        stack.back() = card.is_setcode(static_cast<uint32_t>(stack.back()));
      }
      break;
    case OPCODE_ISTYPE:
      if (!stack.empty()) {
        stack.back() = card.type & static_cast<uint32_t>(stack.back());
      }
      break;
    case OPCODE_ISRACE:
      if (!stack.empty()) {
        stack.back() = card.race & static_cast<uint32_t>(stack.back());
      }
      break;
    case OPCODE_ISATTRIBUTE:
      if (!stack.empty()) {
        stack.back() = card.attribute & static_cast<uint32_t>(stack.back());
      }
      break;
    default:
      stack.push_back(static_cast<int32_t>(opcode));
      break;
    }
  }
  if (stack.size() != 1 || stack.back() == 0) {
    return false;
  }
  return card.code == 78734254u || card.code == 13857930u ||
         (!card.alias &&
          (card.type & (TYPE_MONSTER | TYPE_TOKEN)) !=
              (TYPE_MONSTER | TYPE_TOKEN));
}

static std::string msg_to_string(int msg) {
  switch (msg) {
  case MSG_RETRY:
    return "retry";
  case MSG_HINT:
    return "hint";
  case MSG_WIN:
    return "win";
  case MSG_SELECT_BATTLECMD:
    return "select_battlecmd";
  case MSG_SELECT_IDLECMD:
    return "select_idlecmd";
  case MSG_SELECT_EFFECTYN:
    return "select_effectyn";
  case MSG_SELECT_YESNO:
    return "select_yesno";
  case MSG_SELECT_OPTION:
    return "select_option";
  case MSG_SELECT_CARD:
    return "select_card";
  case MSG_SELECT_CHAIN:
    return "select_chain";
  case MSG_SELECT_PLACE:
    return "select_place";
  case MSG_SELECT_POSITION:
    return "select_position";
  case MSG_SELECT_TRIBUTE:
    return "select_tribute";
  case MSG_SELECT_COUNTER:
    return "select_counter";
  case MSG_SELECT_SUM:
    return "select_sum";
  case MSG_SELECT_DISFIELD:
    return "select_disfield";
  case MSG_SORT_CARD:
    return "sort_card";
  case MSG_SELECT_UNSELECT_CARD:
    return "select_unselect_card";
  case MSG_CONFIRM_DECKTOP:
    return "confirm_decktop";
  case MSG_CONFIRM_CARDS:
    return "confirm_cards";
  case MSG_SHUFFLE_DECK:
    return "shuffle_deck";
  case MSG_SHUFFLE_HAND:
    return "shuffle_hand";
  case MSG_SWAP_GRAVE_DECK:
    return "swap_grave_deck";
  case MSG_SHUFFLE_SET_CARD:
    return "shuffle_set_card";
  case MSG_REVERSE_DECK:
    return "reverse_deck";
  case MSG_DECK_TOP:
    return "deck_top";
  case MSG_SHUFFLE_EXTRA:
    return "shuffle_extra";
  case MSG_NEW_TURN:
    return "new_turn";
  case MSG_NEW_PHASE:
    return "new_phase";
  case MSG_CONFIRM_EXTRATOP:
    return "confirm_extratop";
  case MSG_MOVE:
    return "move";
  case MSG_POS_CHANGE:
    return "pos_change";
  case MSG_SET:
    return "set";
  case MSG_SWAP:
    return "swap";
  case MSG_FIELD_DISABLED:
    return "field_disabled";
  case MSG_SUMMONING:
    return "summoning";
  case MSG_SUMMONED:
    return "summoned";
  case MSG_SPSUMMONING:
    return "spsummoning";
  case MSG_SPSUMMONED:
    return "spsummoned";
  case MSG_FLIPSUMMONING:
    return "flipsummoning";
  case MSG_FLIPSUMMONED:
    return "flipsummoned";
  case MSG_CHAINING:
    return "chaining";
  case MSG_CHAINED:
    return "chained";
  case MSG_CHAIN_SOLVING:
    return "chain_solving";
  case MSG_CHAIN_SOLVED:
    return "chain_solved";
  case MSG_CHAIN_END:
    return "chain_end";
  case MSG_CHAIN_NEGATED:
    return "chain_negated";
  case MSG_CHAIN_DISABLED:
    return "chain_disabled";
  case MSG_RANDOM_SELECTED:
    return "random_selected";
  case MSG_BECOME_TARGET:
    return "become_target";
  case MSG_DRAW:
    return "draw";
  case MSG_DAMAGE:
    return "damage";
  case MSG_RECOVER:
    return "recover";
  case MSG_EQUIP:
    return "equip";
  case MSG_LPUPDATE:
    return "lpupdate";
  case MSG_CARD_TARGET:
    return "card_target";
  case MSG_CANCEL_TARGET:
    return "cancel_target";
  case MSG_PAY_LPCOST:
    return "pay_lpcost";
  case MSG_ADD_COUNTER:
    return "add_counter";
  case MSG_REMOVE_COUNTER:
    return "remove_counter";
  case MSG_ATTACK:
    return "attack";
  case MSG_BATTLE:
    return "battle";
  case MSG_ATTACK_DISABLED:
    return "attack_disabled";
  case MSG_DAMAGE_STEP_START:
    return "damage_step_start";
  case MSG_DAMAGE_STEP_END:
    return "damage_step_end";
  case MSG_MISSED_EFFECT:
    return "missed_effect";
  case MSG_TOSS_COIN:
    return "toss_coin";
  case MSG_TOSS_DICE:
    return "toss_dice";
  case MSG_ROCK_PAPER_SCISSORS:
    return "rock_paper_scissors";
  case MSG_HAND_RES:
    return "hand_res";
  case MSG_ANNOUNCE_RACE:
    return "announce_race";
  case MSG_ANNOUNCE_ATTRIB:
    return "announce_attrib";
  case MSG_ANNOUNCE_CARD:
    return "announce_card";
  case MSG_ANNOUNCE_NUMBER:
    return "announce_number";
  case MSG_CARD_HINT:
    return "card_hint";
  case MSG_TAG_SWAP:
    return "tag_swap";
  case MSG_RELOAD_FIELD:
    return "reload_field";
  case MSG_AI_NAME:
    return "ai_name";
  case MSG_SHOW_HINT:
    return "show_hint";
  case MSG_PLAYER_HINT:
    return "player_hint";
  case MSG_MATCH_KILL:
    return "match_kill";
  case MSG_CUSTOM_MSG:
    return "custom_msg";
  default:
    return "unknown_msg";
  }
}

// system string
static const std::map<int, std::string> system_strings = {
    // announce type
    {1050, "Monster"},
    {1051, "Spell"},
    {1052, "Trap"},
    {1054, "Normal"},
    {1055, "Effect"},
    {1056, "Fusion"},
    {1057, "Ritual"},
    {1058, "Trap Monsters"},
    {1059, "Spirit"},
    {1060, "Union"},
    {1061, "Gemini"},
    {1062, "Tuner"},
    {1063, "Synchro"},
    {1064, "Token"},
    {1066, "Quick-Play"},
    {1067, "Continuous"},
    {1068, "Equip"},
    {1069, "Field"},
    {1070, "Counter"},
    {1071, "Flip"},
    {1072, "Toon"},
    {1073, "Xyz"},
    {1074, "Pendulum"},
    {1075, "Special Summon"},
    {1076, "Link"},
    {1080, "(N/A)"},
    {1081, "Extra Monster Zone"},
    // announce type end
    // actions
    {1150, "Activate"},
    {1151, "Normal Summon"},
    {1152, "Special Summon"},
    {1153, "Set"},
    {1154, "Flip Summon"},
    {1155, "To Defense"},
    {1156, "To Attack"},
    {1157, "Attack"},
    {1158, "View"},
    {1159, "S/T Set"},
    {1160, "Put in Pendulum Zone"},
    {1161, "Do Effect"},
    {1162, "Reset Effect"},
    {1163, "Pendulum Summon"},
    {1164, "Synchro Summon"},
    {1165, "Xyz Summon"},
    {1166, "Link Summon"},
    {1167, "Tribute Summon"},
    {1168, "Ritual Summon"},
    {1169, "Fusion Summon"},
    {1190, "Add to hand"},
    {1191, "Send to GY"},
    {1192, "Banish"},
    {1193, "Return to Deck"},
    // actions end
    {1, "Normal Summon"},
    {30, "Replay rules apply. Continue this attack?"},
    {31, "Attack directly with this monster?"},
    {80, "Start Step of the Battle Phase."},
    {81, "During the End Phase."},
    {90, "Conduct this Normal Summon without Tributing?"},
    {91, "Use additional Summon?"},
    {92, "Tribute your opponent's monster?"},
    {93, "Continue selecting Materials?"},
    {94, "Activate this card's effect now?"},
    {95, "Use the effect of [%ls]?"},
    {96, "Use the effect of [%ls] to avoid destruction?"},
    {97, "Place [%ls] to a Spell & Trap Zone?"},
    {98, "Tribute a monster(s) your opponent controls?"},
    {200, "From [%ls], activate [%ls]?"},
    {203, "Chain another card or effect?"},
    {210, "Continue selecting?"},
    {218, "Pay LP by Effect of [%ls], instead?"},
    {219, "Detach Xyz material by Effect of [%ls], instead?"},
    {220, "Remove Counter(s) by Effect of [%ls], instead?"},
    {221, "On [%ls], Activate Trigger Effect of [%ls]?"},
    {222, "Activate Trigger Effect?"},
    {221, "On [%ls], Activate Trigger Effect of [%ls]?"},
    {1621, "Attack Negated"},
    {1622, "[%ls] Missed timing"}
};

// Keep the legacy system-string IDs stable for checkpoint compatibility.
// New script-level choices are assigned after the legacy range.
static const std::map<int, std::string> system_string_extensions = {
    {1213, "Yes"},
    {1214, "No"},
};

// Standard descriptions used directly by this script snapshot. Keep these in
// a separate map so adding lower numeric keys cannot renumber 1213/1214.
static const std::map<int, std::string> script_system_string_extensions = {
    {10, "Remove Counter(s)"},
    {66, "Keep revealed"},
    {1104, "Back to Hand"},
    {1623, "Coin landed on:"},
};

// Generic engine prompts below 1000 fit in the remaining byte range and are
// appended after every checkpoint-compatible ID above.
static const std::vector<int> generic_system_string_extensions = {
    2,   3,   4,   5,   6,   7,   11,  12,  20,  21,  22,  23,
    24,  25,  26,  27,  28,  29,  40,  41,  42,  43,  44,  60,
    61,  62,  63,  64,  65,  67,  70,  71,  72,  100, 101, 102,
    103, 201, 202, 204, 205, 206, 207, 208, 209, 211, 212, 213,
    214, 215, 216, 217, 223, 224, 225, 500, 501, 502, 503, 504,
    505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516,
    517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528,
    529, 530, 531, 532, 533, 534, 549, 550, 551, 552, 553, 554,
    555, 556, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,
    570, 571, 572, 573, 574, 575,
};

static std::string get_system_string(int desc) {
  auto it = system_strings.find(desc);
  if (it != system_strings.end()) {
    return it->second;
  }
  auto extension_it = system_string_extensions.find(desc);
  if (extension_it != system_string_extensions.end()) {
    return extension_it->second;
  }
  auto script_extension_it = script_system_string_extensions.find(desc);
  if (script_extension_it != script_system_string_extensions.end()) {
    return script_extension_it->second;
  }
  return "system string " + std::to_string(desc);
}

static std::string ltrim(std::string s) {
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(),
                       std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}


inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos) {
  std::string spec;
  if (loc & LOCATION_HAND) {
    spec += "h";
  } else if (loc & LOCATION_MZONE) {
    spec += "m";
  } else if (loc & LOCATION_SZONE) {
    spec += "s";
  } else if (loc & LOCATION_GRAVE) {
    spec += "g";
  } else if (loc & LOCATION_REMOVED) {
    spec += "r";
  } else if (loc & LOCATION_EXTRA) {
    spec += "x";
  }
  spec += std::to_string(seq + 1);
  if (loc & LOCATION_OVERLAY) {
    spec.push_back('a' + pos);
  }
  return spec;
}

inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos, bool opponent) {
  std::string spec = ls_to_spec(loc, seq, pos);
  if (opponent) {
    spec.insert(0, 1, 'o');
  }
  return spec;
}

inline std::tuple<uint8_t, uint8_t, uint8_t>
spec_to_ls(const std::string spec) {
  uint8_t loc;
  uint8_t seq;
  uint8_t pos = 0;
  int offset = 1;
  if (spec[0] == 'h') {
    loc = LOCATION_HAND;
  } else if (spec[0] == 'm') {
    loc = LOCATION_MZONE;
  } else if (spec[0] == 's') {
    loc = LOCATION_SZONE;
  } else if (spec[0] == 'g') {
    loc = LOCATION_GRAVE;
  } else if (spec[0] == 'r') {
    loc = LOCATION_REMOVED;
  } else if (spec[0] == 'x') {
    loc = LOCATION_EXTRA;
  } else if (std::isdigit(spec[0])) {
    loc = LOCATION_DECK;
    offset = 0;
  } else {
    std::string s = fmt::format("Invalid spec {}", spec);
    throw std::runtime_error(s);
  }
  int end = offset;
  while (end < spec.size() && std::isdigit(spec[end])) {
    end++;
  }
  seq = std::stoi(spec.substr(offset, end - offset)) - 1;
  if (end < spec.size()) {
    pos = spec[end] - 'a';
  }
  return {loc, seq, pos};
}


inline std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>
spec_to_ls(uint8_t player, const std::string spec) {
  uint8_t controller = player;
  int offset = 0;
  if (spec[0] == 'o') {
    controller = 1 - player;
    offset++;
  }
  auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
  return {controller, loc, seq, pos};
}


static std::tuple<std::vector<uint32>, std::vector<uint32>, std::vector<uint32>> read_decks(const std::string &fp) {
  std::ifstream file(fp);
  std::string line;
  std::vector<uint32> main_deck, extra_deck, side_deck;
  bool found_extra = false;

  if (file.is_open()) {
    // Read the main deck
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.find("side") != std::string::npos) {
        break;
      }
      if (line.find("extra") != std::string::npos) {
        found_extra = true;
        break;
      }
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        main_deck.push_back(std::stoul(line));
      }
    }

    if (main_deck.size() < 40) {
      std::string err = fmt::format("Main deck must contain at least 40 cards, found: {}, file: {}", main_deck.size(), fp);
      throw std::runtime_error(err);
    }

    // Read the extra deck
    if (found_extra) {
      while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
          line.pop_back();
        }
        if (line.find("side") != std::string::npos) {
          break;
        }
        // Check if line contains only digits
        if (std::all_of(line.begin(), line.end(), ::isdigit)) {
          extra_deck.push_back(std::stoul(line));
        }
      }
    }

    // Read the side deck
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        side_deck.push_back(std::stoul(line));
      }
    }

    file.close();
  } else {
    throw std::runtime_error(fmt::format("Unable to open deck file: {}", fp));
  }

  return std::make_tuple(main_deck, extra_deck, side_deck);
}

template <class K = uint8_t>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::map<K, std::string> &m, int id_offset = 0,
         int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  int i = 0;
  for (const auto &[k, v] : m) {
    if (i < m_offset) {
      i++;
      continue;
    }
    m2[k] = i - m_offset + id_offset;
    i++;
  }
  return m2;
}

template <class K = char>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::vector<K> &cmds, int id_offset = 0, int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  for (int i = m_offset; i < cmds.size(); i++) {
    m2[cmds[i]] = i - m_offset + id_offset;
  }
  return m2;
}

static std::string reason_to_string(uint8_t reason) {
  // !victory 0x0 Surrendered
  // !victory 0x1 LP reached 0
  // !victory 0x2 Cards can't be drawn
  // !victory 0x3 Time limit up
  // !victory 0x4 Lost connection
  switch (reason) {
  case 0x0:
    return "Surrendered";
  case 0x1:
    return "LP reached 0";
  case 0x2:
    return "Cards can't be drawn";
  case 0x3:
    return "Time limit up";
  case 0x4:
    return "Lost connection";
  default:
    return "Unknown";
  }
}

#define DEFINE_X_TO_ID_FUN(name, x_map) \
inline uint8_t name(decltype(x_map)::key_type x) { \
  auto it = x_map.find(x); \
  if (it != x_map.end()) { \
    return it->second; \
  } \
  throw std::runtime_error( \
    fmt::format("[" #name "] cannot find id: {}", x)); \
}

#define DEFINE_X_TO_STRING_FUN(name, x_map) \
inline std::string name(decltype(x_map)::key_type x) { \
  auto it = x_map.find(x); \
  if (it != x_map.end()) { \
    return it->second; \
  } \
  return "unknown"; \
}

static ankerl::unordered_dense::map<int, uint8_t> make_system_string_ids() {
  auto ids = make_ids(system_strings, 16);
  int next_id = 16 + static_cast<int>(system_strings.size());
  for (const auto &entry : system_string_extensions) {
    ids[entry.first] = static_cast<uint8_t>(next_id++);
  }
  for (const auto &entry : script_system_string_extensions) {
    ids[entry.first] = static_cast<uint8_t>(next_id++);
  }
  for (const auto desc : generic_system_string_extensions) {
    if (next_id >= 255) {
      throw std::runtime_error("System string feature IDs exceed byte capacity");
    }
    ids[desc] = static_cast<uint8_t>(next_id++);
  }
  return ids;
}

static const ankerl::unordered_dense::map<int, uint8_t> system_string2id =
    make_system_string_ids();
inline uint8_t system_string_to_id(int desc) {
  auto it = system_string2id.find(desc);
  if (it != system_string2id.end()) {
    return it->second;
  }
  return 255;
}


static const std::map<uint8_t, std::string> location2str = {
    {LOCATION_DECK, "Deck"},
    {LOCATION_HAND, "Hand"},
    {LOCATION_MZONE, "Main Monster Zone"},
    {LOCATION_SZONE, "Spell & Trap Zone"},
    {LOCATION_GRAVE, "Graveyard"},
    {LOCATION_REMOVED, "Banished"},
    {LOCATION_EXTRA, "Extra Deck"},
};

static const ankerl::unordered_dense::map<uint8_t, uint8_t> location2id =
    make_ids(location2str, 1);
DEFINE_X_TO_ID_FUN(location_to_id, location2id)


#define POS_NONE 0x0 // xyz materials (overlay) ???

static const std::map<uint8_t, std::string> position2str = {
    {POS_NONE, "none"},
    {POS_FACEUP_ATTACK, "face-up attack"},
    {POS_FACEDOWN_ATTACK, "face-down attack"},
    {POS_ATTACK, "attack"},
    {POS_FACEUP_DEFENSE, "face-up defense"},
    {POS_FACEUP, "face-up"},
    {POS_FACEDOWN_DEFENSE, "face-down defense"},
    {POS_FACEDOWN, "face-down"},
    {POS_DEFENSE, "defense"},
};
DEFINE_X_TO_STRING_FUN(position_to_string, position2str)

static const ankerl::unordered_dense::map<uint8_t, uint8_t> position2id =
    make_ids(position2str);
DEFINE_X_TO_ID_FUN(position_to_id_exact, position2id)
// Maintained-core protocol (upstream #841): POS_REVEAL (0x80) marks a face-down
// card revealed on the field; the position id keeps the base position.
#ifndef POS_REVEAL
#define POS_REVEAL 0x80
#endif
inline uint8_t position_to_id(uint8_t x) { return position_to_id_exact(x & ~POS_REVEAL); }


#define ATTRIBUTE_NONE 0x0 // token

static const std::map<uint8_t, std::string> attribute2str = {
    {ATTRIBUTE_NONE, "None"},   {ATTRIBUTE_EARTH, "Earth"},
    {ATTRIBUTE_WATER, "Water"}, {ATTRIBUTE_FIRE, "Fire"},
    {ATTRIBUTE_WIND, "Wind"},   {ATTRIBUTE_LIGHT, "Light"},
    {ATTRIBUTE_DARK, "Dark"},   {ATTRIBUTE_DEVINE, "Divine"},
};
DEFINE_X_TO_STRING_FUN(attribute_to_string, attribute2str)

static const ankerl::unordered_dense::map<uint8_t, uint8_t> attribute2id =
    make_ids(attribute2str);
DEFINE_X_TO_ID_FUN(attribute_to_id, attribute2id)


#define RACE_NONE 0x0 // token

static const std::map<uint32_t, std::string> race2str = {
    {RACE_NONE, "None"},
    {RACE_WARRIOR, "Warrior"},
    {RACE_SPELLCASTER, "Spellcaster"},
    {RACE_FAIRY, "Fairy"},
    {RACE_FIEND, "Fiend"},
    {RACE_ZOMBIE, "Zombie"},
    {RACE_MACHINE, "Machine"},
    {RACE_AQUA, "Aqua"},
    {RACE_PYRO, "Pyro"},
    {RACE_ROCK, "Rock"},
    {RACE_WINDBEAST, "Windbeast"},
    {RACE_PLANT, "Plant"},
    {RACE_INSECT, "Insect"},
    {RACE_THUNDER, "Thunder"},
    {RACE_DRAGON, "Dragon"},
    {RACE_BEAST, "Beast"},
    {RACE_BEASTWARRIOR, "Beast Warrior"},
    {RACE_DINOSAUR, "Dinosaur"},
    {RACE_FISH, "Fish"},
    {RACE_SEASERPENT, "Sea Serpent"},
    {RACE_REPTILE, "Reptile"},
    {RACE_PSYCHO, "Psycho"},
    {RACE_DEVINE, "Divine"},
    {RACE_CREATORGOD, "Creator God"},
    {RACE_WYRM, "Wyrm"},
    {RACE_CYBERSE, "Cyberse"},
    {RACE_ILLUSION, "Illusion"}};

static const ankerl::unordered_dense::map<uint32_t, uint8_t> race2id =
    make_ids(race2str);
DEFINE_X_TO_ID_FUN(race_to_id, race2id)


static const std::map<uint32_t, std::string> type2str = {
    {TYPE_MONSTER, "Monster"},
    {TYPE_SPELL, "Spell"},
    {TYPE_TRAP, "Trap"},
    {TYPE_NORMAL, "Normal"},
    {TYPE_EFFECT, "Effect"},
    {TYPE_FUSION, "Fusion"},
    {TYPE_RITUAL, "Ritual"},
    {TYPE_TRAPMONSTER, "Trap Monster"},
    {TYPE_SPIRIT, "Spirit"},
    {TYPE_UNION, "Union"},
    {TYPE_DUAL, "Dual"},
    {TYPE_TUNER, "Tuner"},
    {TYPE_SYNCHRO, "Synchro"},
    {TYPE_TOKEN, "Token"},
    {TYPE_QUICKPLAY, "Quick-play"},
    {TYPE_CONTINUOUS, "Continuous"},
    {TYPE_EQUIP, "Equip"},
    {TYPE_FIELD, "Field"},
    {TYPE_COUNTER, "Counter"},
    {TYPE_FLIP, "Flip"},
    {TYPE_TOON, "Toon"},
    {TYPE_XYZ, "XYZ"},
    {TYPE_PENDULUM, "Pendulum"},
    {TYPE_SPSUMMON, "Special"},
    {TYPE_LINK, "Link"},
};

inline std::vector<uint8_t> type_to_ids(uint32_t type) {
  std::vector<uint8_t> ids;
  ids.reserve(type2str.size());
  for (const auto &[k, v] : type2str) {
    ids.push_back(std::min(1u, type & k));
  }
  return ids;
}

static const std::map<int, std::string> phase2str = {
    {PHASE_DRAW, "draw phase"},
    {PHASE_STANDBY, "standby phase"},
    {PHASE_MAIN1, "main1 phase"},
    {PHASE_BATTLE_START, "battle start phase"},
    {PHASE_BATTLE_STEP, "battle step phase"},
    {PHASE_DAMAGE, "damage phase"},
    {PHASE_DAMAGE_CAL, "damage calculation phase"},
    {PHASE_BATTLE, "battle phase"},
    {PHASE_MAIN2, "main2 phase"},
    {PHASE_END, "end phase"},
};
DEFINE_X_TO_STRING_FUN(phase_to_string, phase2str)

static const ankerl::unordered_dense::map<int, uint8_t> phase2id =
    make_ids(phase2str);
DEFINE_X_TO_ID_FUN(phase_to_id, phase2id)


static const std::vector<int> _msgs = {
    MSG_SELECT_IDLECMD,  MSG_SELECT_CHAIN,     MSG_SELECT_CARD,
    MSG_SELECT_TRIBUTE,  MSG_SELECT_POSITION,  MSG_SELECT_EFFECTYN,
    MSG_SELECT_YESNO,    MSG_SELECT_BATTLECMD, MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_OPTION,   MSG_SELECT_PLACE,     MSG_SELECT_SUM,
    MSG_SELECT_DISFIELD, MSG_ANNOUNCE_ATTRIB,  MSG_ANNOUNCE_NUMBER,
    MSG_ANNOUNCE_CARD,   MSG_ANNOUNCE_RACE,    MSG_TOSS_COIN,
    MSG_TOSS_DICE,       MSG_SWAP_GRAVE_DECK,
};

static const ankerl::unordered_dense::map<int, uint8_t> msg2id =
    make_ids(_msgs, 1);
DEFINE_X_TO_ID_FUN(msg_to_id, msg2id)


enum class ActionAct {
  None,
  Set,
  Repo,
  SpSummon,
  Summon,
  MSet,
  Attack,
  DirectAttack,
  Activate,
  Cancel,
};

inline std::string action_act_to_string(ActionAct act) {
  switch (act) {
  case ActionAct::None:
    return "None";
  case ActionAct::Set:
    return "Set";
  case ActionAct::Repo:
    return "Repo";
  case ActionAct::SpSummon:
    return "SpSummon";
  case ActionAct::Summon:
    return "Summon";
  case ActionAct::MSet:
    return "MSet";
  case ActionAct::Attack:
    return "Attack";
  case ActionAct::DirectAttack:
    return "DirectAttack";
  case ActionAct::Activate:
    return "Activate";
  case ActionAct::Cancel:
    return "Cancel";
  default:
    return "Unknown";
  }
}

enum class ActionPhase {
  None,
  Battle,
  Main2,
  End,
};

inline std::string action_phase_to_string(ActionPhase phase) {
  switch (phase) {
  case ActionPhase::None:
    return "None";
  case ActionPhase::Battle:
    return "Battle";
  case ActionPhase::Main2:
    return "Main2";
  case ActionPhase::End:
    return "End";
  default:
    return "Unknown";
  }
}

enum class ActionPlace {
  None,
  MZone1,
  MZone2,
  MZone3,
  MZone4,
  MZone5,
  MZone6,
  MZone7,
  SZone1,
  SZone2,
  SZone3,
  SZone4,
  SZone5,
  SZone6,
  SZone7,
  SZone8,
  OpMZone1,
  OpMZone2,
  OpMZone3,
  OpMZone4,
  OpMZone5,
  OpMZone6,
  OpMZone7,
  OpSZone1,
  OpSZone2,
  OpSZone3,
  OpSZone4,
  OpSZone5,
  OpSZone6,
  OpSZone7,
  OpSZone8,
};

enum class SelectionRole : uint8_t {
  None = 0,
  Candidate = 1,
  Target = 2,
  Cost = 3,
  Material = 4,
  Tribute = 5,
};

constexpr int kActionIRFeatures = 24;
constexpr int kHiddenLabelRows = 96;  // label:hidden_ rows (S1 belief targets)
constexpr int kRecipeRows = 80;       // obs:opponent_recipe_ rows: distinct main (at most 60) and extra (15) cards
constexpr int kOwnRecipeWidth = 6;    // obs:own_recipe_ columns (_set_obs_own_recipe)
constexpr int kGlobalFeatures = 25;   // obs:global_ columns: 23 board columns, room format, era
constexpr int kActionSingleRefs = 4;
constexpr int kActionGroupRoles = 5;
constexpr int kMaxActionRoleMembers = 8;
constexpr int kSelectionFeatures = 12;
constexpr int kPublicEventFeatures = 16;
constexpr int kPublicEventRefs = 4;


inline std::vector<ActionPlace> flag_to_usable_places(
  uint32_t flag, bool reverse = false) {
  std::vector<ActionPlace> places;
  for (int j = 0; j < 4; j++) {
    uint32_t value = (flag >> (j * 8)) & 0xff;
    for (int i = 0; i < 8; i++) {
      bool avail = (value & (1 << i)) == 0;
      if (reverse) {
        avail = !avail;
      }
      if (avail) {
        ActionPlace place;
        if (j == 0) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::MZone1));
        } else if (j == 1) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::SZone1));
        } else if (j == 2) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::OpMZone1));
        } else if (j == 3) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::OpSZone1));
        }
        places.push_back(place);
      }
    }
  }
  return places;
}

inline std::string action_place_to_string(ActionPlace place) {
  int i = static_cast<int>(place);
  if (i == 0) {
    return "None";
  }
  else if (i >= static_cast<int>(ActionPlace::MZone1) && i <= static_cast<int>(ActionPlace::MZone7)) {
    return fmt::format("m{}", i - static_cast<int>(ActionPlace::MZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::SZone1) && i <= static_cast<int>(ActionPlace::SZone8)) {
    return fmt::format("s{}", i - static_cast<int>(ActionPlace::SZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::OpMZone1) && i <= static_cast<int>(ActionPlace::OpMZone7)) {
    return fmt::format("om{}", i - static_cast<int>(ActionPlace::OpMZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::OpSZone1) && i <= static_cast<int>(ActionPlace::OpSZone8)) {
    return fmt::format("os{}", i - static_cast<int>(ActionPlace::OpSZone1) + 1);
  }
  else {
    return "Unknown";
  }
}


inline std::pair<uint8_t, uint8_t> float_transform(int x) {
  x = x % 65536;
  return {
      static_cast<uint8_t>(x >> 8),
      static_cast<uint8_t>(x & 0xff),
  };
}

static std::vector<int> find_substrs(const std::string &str,
                                     const std::string &substr) {
  std::vector<int> res;
  int pos = 0;
  while ((pos = str.find(substr, pos)) != std::string::npos) {
    res.push_back(pos);
    pos += substr.length();
  }
  return res;
}

inline std::string time_now() {
  // strftime %Y-%m-%d %H-%M-%S
  time_t now = time(0);
  tm *ltm = localtime(&now);
  char buffer[80];
  strftime(buffer, 80, "%Y-%m-%d %H-%M-%S", ltm);
  return std::string(buffer);
}

// from duel/gframe/replay.h

// replay flag
#define REPLAY_COMPRESSED	0x1
#define REPLAY_TAG			0x2
#define REPLAY_DECODED		0x4
#define REPLAY_SINGLE_MODE	0x8
#define REPLAY_UNIFORM		0x10

// max size
#define MAX_REPLAY_SIZE	0x20000


struct ReplayHeader {
	unsigned int id;
	unsigned int version;
	unsigned int flag;
	unsigned int seed;
	unsigned int datasize;
	unsigned int start_time;
	unsigned char props[8];

	ReplayHeader()
		: id(0), version(0), flag(0), seed(0), datasize(0), start_time(0), props{ 0 } {}
};

// from duel/gframe/replay.h

using PlayerId = uint8_t;
using CardCode = uint32_t;
using CardId = uint16_t;

const int DESCRIPTION_LIMIT = 10000;
const int CARD_EFFECT_OFFSET = 10010;
const uint8_t ACTION_RACE_EFFECT_OFFSET = 128;
const uint8_t ACTION_ANNOUNCE_NUMBER_MAX = 12;

inline uint8_t announce_number_to_id(uint32_t number) {
  return static_cast<uint8_t>(
      std::min<uint32_t>(number, ACTION_ANNOUNCE_NUMBER_MAX));
}

class LegalAction {
public:
  std::string spec_ = "";
  ActionAct act_ = ActionAct::None;
  ActionPhase phase_ = ActionPhase::None;
  bool finish_ = false;
  uint8_t position_ = 0;
  int effect_ = -1;
  uint8_t number_ = 0;
  ActionPlace place_ = ActionPlace::None;
  uint8_t attribute_ = 0;

  int spec_index_ = 0;
  CardId cid_ = 0;
  int msg_ = 0;
  uint32_t response_ = 0;
  // An activate row's activatable option in its prompt message (message order) with the code and description the
  // message gave it, for query_activation_flags (the illegal-activation withdrawal); -1 for other rows.
  int option_ = -1;
  uint32_t option_code_ = 0, option_desc_ = 0;

  static LegalAction from_spec(const std::string &spec) {
    LegalAction la;
    la.spec_ = spec;
    return la;
  }

  static LegalAction act_spec(ActionAct act, const std::string &spec) {
    LegalAction la;
    la.act_ = act;
    la.spec_ = spec;
    return la;
  }

  static LegalAction finish() {
    LegalAction la;
    la.finish_ = true;
    return la;
  }

  static LegalAction cancel() {
    LegalAction la;
    la.act_ = ActionAct::Cancel;
    return la;
  }

  static LegalAction activate_spec(int effect_idx, const std::string &spec) {
    LegalAction la;
    la.act_ = ActionAct::Activate;
    la.effect_ = effect_idx;
    la.spec_ = spec;
    return la;
  }

  static LegalAction phase(ActionPhase phase) {
    LegalAction la;
    la.phase_ = phase;
    return la;
  }

  static LegalAction number(uint8_t number) {
    LegalAction la;
    la.number_ = number;
    return la;
  }

  static LegalAction place(ActionPlace place) {
    LegalAction la;
    la.place_ = place;
    return la;
  }

  static LegalAction attribute(int attribute) {
    LegalAction la;
    la.attribute_ = attribute;
    return la;
  }
};

struct PublicEvent {
  PlayerId actor = 0;
  int msg = 0;
  ActionAct act = ActionAct::None;
  ActionPhase action_phase = ActionPhase::None;
  SelectionRole selection_role = SelectionRole::None;
  bool finish = false;
  bool cancel = false;
  int effect = -1;
  int turn = 0;
  int phase = 0;
  int selection_stage = 0;
  int selected_count = 0;
  uint8_t choice = 0;
  uint8_t payload_size = 0;
  uint8_t payload[5] = {};
  std::string source_spec;
  PlayerId source_spec_player = 0;
  std::string candidate_spec;
  PlayerId candidate_spec_player = 0;
};

class SpecInfo {
public:
  uint16_t index;
  CardId cid;
};

class SearchDuel;

class Card {
  friend class DuelEnvImpl;
  friend class SearchDuel;

protected:
  CardCode code_ = 0;
  uint32_t alias_;
  uint64_t setcode_;
  uint32_t type_;
  uint32_t level_;
  uint32_t lscale_;
  uint32_t rscale_;
  int32_t attack_;
  int32_t defense_;
  uint32_t race_;
  uint32_t attribute_;
  uint32_t link_marker_;
  // uint32_t category_;
  std::string name_;
  std::string desc_;
  std::vector<std::string> strings_;

  uint32_t data_ = 0;

  uint32_t status_ = 0;
  PlayerId controler_ = 0;
  uint32_t location_ = 0;
  uint32_t sequence_ = 0;
  uint32_t position_ = 0;
  uint32_t counter_ = 0;
  uint32_t equip_target_ = 0;  // the queried equip target's info location (0 for none); audits only

public:
  Card() = default;

  Card(CardCode code, uint32_t alias, uint64_t setcode, uint32_t type,
       uint32_t level, uint32_t lscale, uint32_t rscale, int32_t attack,
       int32_t defense, uint32_t race, uint32_t attribute, uint32_t link_marker,
       const std::string &name, const std::string &desc,
       const std::vector<std::string> &strings)
      : code_(code), alias_(alias), setcode_(setcode), type_(type),
        level_(level), lscale_(lscale), rscale_(rscale), attack_(attack),
        defense_(defense), race_(race), attribute_(attribute),
        link_marker_(link_marker), name_(name), desc_(desc), strings_(strings) {
  }

  ~Card() = default;

  void set_location(uint32_t location) {
    controler_ = location & 0xff;
    location_ = (location >> 8) & 0xff;
    sequence_ = (location >> 16) & 0xff;
    position_ = (location >> 24) & 0xff;
  }

  const std::string &name() const { return name_; }
  const std::string &desc() const { return desc_; }
  const uint32_t &type() const { return type_; }
  const uint32_t &level() const { return level_; }
  const std::vector<std::string> &strings() const { return strings_; }

  std::string get_spec(bool opponent) const {
    return ls_to_spec(location_, sequence_, position_, opponent);
  }

  std::string get_spec(PlayerId player) const {
    return get_spec(player != controler_);
  }

  std::string get_position() const { return position_to_string(position_); }

  std::string get_effect_description(CardCode code, int effect_idx) const {
    if (code == 0) {
      return get_system_string(effect_idx);
    }
    if (effect_idx == 0) {
      return "default";
    }
    effect_idx -= CARD_EFFECT_OFFSET;
    if (effect_idx < 0) {
      throw std::runtime_error(
          fmt::format("Invalid effect index: {}", effect_idx));
    }
    auto s = strings_[effect_idx];
    if (s.empty()) {
      return "effect " + std::to_string(effect_idx);
    }
    return s;
  }
};

struct MDuel {
  intptr_t pduel;
  uint64_t seed;
  std::vector<CardCode> main_deck0;
  std::vector<CardCode> extra_deck0;
  std::string deck_name0;
  std::vector<CardCode> main_deck1;
  std::vector<CardCode> extra_deck1;
  std::string deck_name1;
};

inline Card db_query_card(const SQLite::Database &db, CardCode code) {
  SQLite::Statement query1(db, "SELECT * FROM datas WHERE id=?");
  query1.bind(1, code);
  bool found = query1.executeStep();
  if (!found) {
    std::string msg = "[db_query_card] Card not found: " + std::to_string(code);
    throw std::runtime_error(msg);
  }

  uint32_t alias = query1.getColumn("alias");
  uint64_t setcode = query1.getColumn("setcode").getInt64();
  uint32_t type = query1.getColumn("type");
  uint32_t level_ = query1.getColumn("level");
  uint32_t level = level_ & 0xff;
  uint32_t lscale = (level_ >> 24) & 0xff;
  uint32_t rscale = (level_ >> 16) & 0xff;
  int32_t attack = query1.getColumn("atk");
  int32_t defense = query1.getColumn("def");
  uint32_t link_marker = 0;
  if (type & TYPE_LINK) {
    link_marker = defense;
    defense = 0;
  }
  uint32_t race = query1.getColumn("race");
  uint32_t attribute = query1.getColumn("attribute");

  SQLite::Statement query2(db, "SELECT * FROM texts WHERE id=?");
  query2.bind(1, code);
  query2.executeStep();

  std::string name = query2.getColumn(1);
  std::string desc = query2.getColumn(2);
  std::vector<std::string> strings;
  for (int i = 3; i < query2.getColumnCount(); ++i) {
    std::string str = query2.getColumn(i);
    strings.push_back(str);
  }
  return Card(code, alias, setcode, type, level, lscale, rscale, attack,
              defense, race, attribute, link_marker, name, desc, strings);
}

inline card_data db_query_card_data(const SQLite::Database &db, CardCode code) {
  SQLite::Statement query(db, "SELECT * FROM datas WHERE id=?");
  query.bind(1, code);
  query.executeStep();
  card_data card;
  card.code = code;
  card.alias = query.getColumn("alias");
  uint64_t setcode = query.getColumn("setcode").getInt64();
  write_setcode(card.setcode, setcode);  // maintained core: card_data::set_setcode became write_setcode
  card.type = query.getColumn("type");
  uint32_t level_ = query.getColumn("level");
  card.level = level_ & 0xff;
  card.lscale = (level_ >> 24) & 0xff;
  card.rscale = (level_ >> 16) & 0xff;
  card.attack = query.getColumn("atk");
  card.defense = query.getColumn("def");
  if (card.type & TYPE_LINK) {
    card.link_marker = card.defense;
    card.defense = 0;
  } else {
    card.link_marker = 0;
  }
  card.race = query.getColumn("race");
  card.attribute = query.getColumn("attribute");
  return card;
}

struct card_script {
  byte *buf;
  int len;
};

static ankerl::unordered_dense::map<CardCode, Card> cards_;
static ankerl::unordered_dense::map<CardCode, CardId> card_ids_;
static ankerl::unordered_dense::map<CardCode, card_data> cards_data_;
static ankerl::unordered_dense::map<std::string, card_script> cards_script_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    main_decks_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    extra_decks_;
static std::vector<std::string> deck_names_;
static ankerl::unordered_dense::map<std::string, int> deck_names_ids_;
static std::vector<std::vector<std::string>> deck_clusters_;

inline const Card &c_get_card(CardCode code) {
  auto it = cards_.find(code);
  if (it != cards_.end()) {
    return it->second;
  }
  throw std::runtime_error("[c_get_card] Card not found: " + std::to_string(code));
}

inline CardId &c_get_card_id_impl(
    CardCode code, const char *function, int line) {
  auto it = card_ids_.find(code);
  if (it != card_ids_.end()) {
    return it->second;
  }
  throw std::runtime_error(fmt::format(
      "[c_get_card_id] Card not found: {} at {}:{}", code, function, line));
}

#define c_get_card_id(code) c_get_card_id_impl((code), __func__, __LINE__)

inline void sort_extra_deck(std::vector<CardCode> &deck) {
  std::vector<CardCode> c;
  std::vector<std::pair<CardCode, int>> fusion, xyz, synchro, link;

  for (auto code : deck) {
    const Card &cc = c_get_card(code);
    if (cc.type() & TYPE_FUSION) {
      fusion.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_XYZ) {
      xyz.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_SYNCHRO) {
      synchro.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_LINK) {
      link.push_back({code, cc.level()});
    } else {
      throw std::runtime_error("Not extra deck card");
    }
  }

  auto cmp = [](const std::pair<CardCode, int> &a,
                const std::pair<CardCode, int> &b) {
    return a.second < b.second;
  };
  std::sort(fusion.begin(), fusion.end(), cmp);
  std::sort(xyz.begin(), xyz.end(), cmp);
  std::sort(synchro.begin(), synchro.end(), cmp);
  std::sort(link.begin(), link.end(), cmp);

  for (const auto &tc : fusion) {
    c.push_back(tc.first);
  }
  for (const auto &tc : xyz) {
    c.push_back(tc.first);
  }
  for (const auto &tc : synchro) {
    c.push_back(tc.first);
  }
  for (const auto &tc : link) {
    c.push_back(tc.first);
  }

  deck = c;
}

inline void preload_deck(const SQLite::Database &db,
                         const std::vector<CardCode> &deck) {
  for (const auto &code : deck) {
    auto it = cards_.find(code);
    if (it == cards_.end()) {
      cards_[code] = db_query_card(db, code);
      if (card_ids_.find(code) == card_ids_.end()) {
        throw std::runtime_error("Card not found in code list: " +
                                 std::to_string(code));
      }
    }

    auto it2 = cards_data_.find(code);
    if (it2 == cards_data_.end()) {
      cards_data_[code] = db_query_card_data(db, code);
    }
  }
}

inline uint32 card_reader_callback(CardCode code, card_data *card) {
  auto it = cards_data_.find(code);
  if (it == cards_data_.end()) {
    fmt::println("[card_reader_callback] Card not found: " + std::to_string(code));
    throw std::runtime_error("[card_reader_callback] Card not found: " + std::to_string(code));
  }
  *card = it->second;
  return 0;
}

inline byte *read_card_script(const std::string &path, int *lenptr) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *lenptr = 0;
    return nullptr;
  }
  file.seekg(0, std::ios::end);
  int len = file.tellg();
  file.seekg(0, std::ios::beg);
  byte *buf = new byte[len];
  file.read((char *)buf, len);
  *lenptr = len;
  return buf;
}

// The core reports a card-script error (a Lua error, a call of a missing
// function) through the message handler and then continues with the effect
// failed. Each duel runs on one pool thread, so the handler keeps the text
// per thread and the env turns it into a fatal error after OCG_Process.
inline std::vector<std::string> &core_script_errors() {
  thread_local std::vector<std::string> errors;
  return errors;
}

inline uint32_t core_message_callback(intptr_t pduel, uint32_t type) {
  char text[1024] = {0};
  get_log_message(pduel, text);
  core_script_errors().push_back(fmt::format("type {}: {}", type, text));
  return 0;
}

// A generated chunk the search API runs (search_api.h): served under its name on this thread only, while set.
struct ScriptOverride {
  const char *name = nullptr;
  const std::string *source = nullptr;
};
inline ScriptOverride &script_override() {
  static thread_local ScriptOverride current;
  return current;
}

inline byte *script_reader_callback(const char *name, int *lenptr) {
  if (script_override().name != nullptr && std::strcmp(name, script_override().name) == 0) {
    *lenptr = static_cast<int>(script_override().source->size());
    return reinterpret_cast<byte *>(const_cast<char *>(script_override().source->data()));
  }
  std::string path(name);
  auto it = cards_script_.find(path);
  if (it == cards_script_.end()) {
    fmt::println("[script_reader_callback] Script not found: " + path);
    throw std::runtime_error("[script_reader_callback] Script not found: " + path);
  }
  *lenptr = it->second.len;
  return it->second.buf;
}

inline std::string deck_cluster_key(const std::string &name) {
  constexpr std::string_view prefix = "cluster_";
  if (name.rfind(prefix.data(), 0) != 0) {
    return name;
  }
  const auto separator = name.find("__", prefix.size());
  if (separator == std::string::npos || separator == prefix.size()) {
    return name;
  }
  return name.substr(0, separator);
}

static void init_module(const std::string &db_path,
                        const std::string &code_list_file,
                        const std::map<std::string, std::string> &decks) {
  main_decks_.clear();
  extra_decks_.clear();
  deck_names_.clear();
  deck_names_ids_.clear();
  deck_clusters_.clear();

  // parse code from code_list_file
  SQLite::Database db(db_path, SQLite::OPEN_READONLY);

  auto start = std::chrono::steady_clock::now();

  std::ifstream file(code_list_file);
  std::string line;
  int i = 0;
  CardCode code;
  int has_script, script_len;
  while (std::getline(file, line)) {
    i++;
    std::istringstream iss(line);
    if (!(iss >> code >> has_script)) {
        std::cerr << "Failed to parse line in code_list: " << line << std::endl;
        continue;
    }
    card_ids_[code] = i;
    cards_[code] = db_query_card(db, code);
    cards_data_[code] = db_query_card_data(db, code);
    if (has_script) {
      std::string path = "./script/c" + std::to_string(code) + ".lua";
      byte *buf = read_card_script(path, &script_len);
      cards_script_[path] = {buf, script_len};
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  // fmt::println("load {} cards in {}ms", cards_data_.size(), milliseconds);

  std::map<std::string, std::vector<std::string>> clusters;
  for (const auto &[name, deck] : decks) {
    auto [main_deck, extra_deck, side_deck] = read_decks(deck);
    main_decks_[name] = main_deck;
    extra_decks_[name] = extra_deck;
    if (name[0] != '_') {
      deck_names_.push_back(name);
      deck_names_ids_[name] = deck_names_.size() - 1;
      clusters[deck_cluster_key(name)].push_back(name);
    }
  }
  deck_clusters_.reserve(clusters.size());
  for (auto &[cluster, names] : clusters) {
    deck_clusters_.push_back(std::move(names));
  }

  for (auto &[name, deck] : extra_decks_) {
    sort_extra_deck(deck);
  }

  history::ArtworkBases().clear();
  for (const auto &[code, data] : cards_data_)
    if (history::IsArtworkVariant(code, data.alias)) history::ArtworkBases().emplace(code, data.alias);

  card_data card;
  cards_data_[0] = card;

  std::vector<std::string> preload = {
    "./script/constant.lua",
    "./script/utility.lua",
    "./script/procedure.lua",
  };
  for (const auto &path : preload) {
    byte *buf = read_card_script(path, &script_len);
    cards_script_[path] = {buf, script_len};
  }
  cards_script_["./script/c0.lua"] = {nullptr, 0};

  set_card_reader(card_reader_callback);
  set_script_reader(script_reader_callback);
  set_message_handler(core_message_callback);
}

inline std::string getline() {
  char *line = nullptr;
  size_t len = 0;
  ssize_t read;

  read = getline(&line, &len, stdin);

  if (read != -1) {
    // Remove line ending character(s)
    if (line[read - 1] == '\n')
      line[read - 1] = '\0'; // Replace newline character with null terminator
    else if (line[read - 2] == '\r' && line[read - 1] == '\n') {
      line[read - 2] = '\0'; // Replace carriage return and newline characters
                             // with null terminator
      line[read - 1] = '\0';
    }

    std::string input(line);
    free(line);
    return input;
  } else {
    exit(0);
  }

  free(line);
  return "";
}

class Player {
  friend class DuelEnvImpl;

protected:
  const std::string nickname_;
  const int init_lp_;
  const PlayerId duel_player_;
  const bool verbose_;

  bool seen_waiting_ = false;

public:
  Player(const std::string &nickname, int init_lp, PlayerId duel_player,
         bool verbose = false)
      : nickname_(nickname), init_lp_(init_lp), duel_player_(duel_player),
        verbose_(verbose) {}
  virtual ~Player() = default;

  void notify(const std::string &text) {
    if (verbose_) {
      fmt::println("{} {}", duel_player_, text);
    }
  }

  const int &init_lp() const { return init_lp_; }

  virtual int think(const std::vector<LegalAction> &actions) = 0;
};

class GreedyAI : public Player {
protected:
public:
  GreedyAI(const std::string &nickname, int init_lp, PlayerId duel_player,
           bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<LegalAction> &actions) override { return 0; }
};

class RandomAI : public Player {
protected:
  std::mt19937 gen_;
  std::uniform_int_distribution<int> dist_;

public:
  RandomAI(int max_options, int seed, const std::string &nickname, int init_lp,
           PlayerId duel_player, bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose), gen_(seed),
        dist_(0, max_options - 1) {}

  int think(const std::vector<LegalAction> &actions) override {
    return dist_(gen_) % actions.size();
  }
};

class HumanPlayer : public Player {
protected:
public:
  HumanPlayer(const std::string &nickname, int init_lp, PlayerId duel_player,
              bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<LegalAction> &actions) override {
    while (true) {
      std::string input = getline();
      if (input == "quit") {
        exit(0);
      }
      int idx = -1;
      try {
        idx = std::stoi(input) - 1;
      } catch (std::invalid_argument &e) {
        fmt::println("{} Invalid input: {}", duel_player_, input);
        continue;
      }
      if (idx >= 0 && idx < actions.size()) {
        return idx;
      } else {
        fmt::println("{} Choose from {} actions", duel_player_, actions.size());
      }
    }
  }
};

class DuelEnvFns {
public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("deck1"_.Bind(std::string("OldSchool")),
                    "deck2"_.Bind(std::string("OldSchool")), "player"_.Bind(-1),
                    "deck_schedule"_.Bind(std::string("independent")),
                    "anchor_deck"_.Bind(std::string("")),
                    "anchor_probability_percent"_.Bind(50),
        "mirror_probability_percent"_.Bind(0),
        "matched_probability_percent"_.Bind(0),
        "pool_deck_uniform"_.Bind(0),
                    "play_mode"_.Bind(std::string("bot")),
                    "verbose"_.Bind(false), "max_options"_.Bind(64),
                    "max_cards"_.Bind(80), "n_history_actions"_.Bind(16),
                    "record"_.Bind(false), "async_reset"_.Bind(false),
                    "greedy_reward"_.Bind(false), "timeout"_.Bind(600),
                    "oppo_info"_.Bind(false), "max_steps"_.Bind(1000),
                    "belief_labels"_.Bind(false), "public_opponent_recipe"_.Bind(false),
                    "allow_unreviewed_public_effects"_.Bind(false), "export_both_seats"_.Bind(false),
                    "room_format"_.Bind(0), "room_era"_.Bind(0),
                    "history_window"_.Bind(history::kEvents), "history_chunk"_.Bind(history::kChunkRows),
                    "history_chunk_cap"_.Bind(history::kChunkCap), "history_chunk_slots"_.Bind(history::kChunkSlots));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config &conf) {
    int n_action_feats = 12;
    // export_both_seats: the non-acting seat's observation under priv: (critic inputs), shaped as its obs: key; zero
    // rows when the flag is off
    const int both = conf["export_both_seats"_] ? 1 : 0;
    return MakeDict(
        "priv:cards_"_.Bind(Spec<uint8_t>({both * conf["max_cards"_] * 2, 41})),
        "priv:global_"_.Bind(Spec<uint8_t>({both * kGlobalFeatures})),
        "priv:hand_limit_"_.Bind(Spec<uint8_t>({both * 2, 4})),
        "priv:own_recipe_"_.Bind(Spec<uint8_t>({both * kRecipeRows, kOwnRecipeWidth})),
        "priv:opponent_recipe_"_.Bind(Spec<uint8_t>({both * kRecipeRows, 4})),
        "priv:unpositioned_"_.Bind(Spec<uint8_t>({both * history::kUnpositionedRows, history::kUnpositionedWidth})),
        "priv:turn_events_"_.Bind(Spec<uint8_t>({both * conf["history_window"_], history::kEventWidth})),
        "priv:turn_event_refs_"_.Bind(Spec<uint8_t>({both * conf["history_window"_], history::kRefWidth})),
        "priv:chain_"_.Bind(Spec<uint8_t>({both * history::kChainRows, history::kChainWidth})),
        "priv:turn_activations_"_.Bind(Spec<uint8_t>({both * history::kActivationRows, history::kActivationWidth})),
        "priv:turn_ledger_"_.Bind(Spec<uint8_t>({both * 2, history::kLedgerWidth})),
        "priv:player_hints_"_.Bind(Spec<uint8_t>({both * history::kHintRows, history::kHintWidth})),
        "priv:card_turn_"_.Bind(Spec<uint8_t>({both * conf["max_cards"_] * 2, history::kCardTurnWidth})),
        "priv:turn_chunks_"_.Bind(
            Spec<uint8_t>({both * conf["history_chunk_slots"_], conf["history_chunk"_], history::kEventWidth})),
        "priv:turn_chunk_refs_"_.Bind(
            Spec<uint8_t>({both * conf["history_chunk_slots"_], conf["history_chunk"_], history::kRefWidth})),
        "priv:turn_chunk_meta_"_.Bind(Spec<int>({both * conf["history_chunk_slots"_], history::kChunkMetaWidth})),
        "priv:closed_turns_"_.Bind(
            Spec<uint8_t>({both * history::kClosedTurns, conf["history_window"_], history::kEventWidth})),
        "priv:closed_turn_refs_"_.Bind(
            Spec<uint8_t>({both * history::kClosedTurns, conf["history_window"_], history::kRefWidth})),
        "priv:closed_turn_meta_"_.Bind(Spec<int>({both * history::kClosedTurns, history::kClosedMetaWidth})),
        "priv:card_status_"_.Bind(Spec<uint8_t>({both * conf["max_cards"_] * 2, public_effects::kStatusWidth})),
        "priv:public_effects_"_.Bind(
            Spec<uint8_t>({both * public_effects::kEffectRows, public_effects::kEffectWidth})),
        "obs:cards_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 41})),
        "obs:global_"_.Bind(Spec<uint8_t>({kGlobalFeatures})),
        "obs:actions_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], n_action_feats})),
        "obs:h_actions_"_.Bind(
            Spec<uint8_t>({conf["n_history_actions"_], n_action_feats + 2})),
        "obs:action_ir_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], kActionIRFeatures})),
        "obs:action_single_refs_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], kActionSingleRefs})),
        "obs:action_group_refs_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], kActionGroupRoles,
                           kMaxActionRoleMembers})),
        "obs:action_group_mask_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], kActionGroupRoles,
                           kMaxActionRoleMembers})),
        "obs:selection_"_.Bind(Spec<uint8_t>({kSelectionFeatures})),
        "obs:hand_limit_"_.Bind(Spec<uint8_t>({2, 4})),
        "obs:action_discard_"_.Bind(Spec<uint8_t>({conf["max_options"_]})),
        "obs:public_events_"_.Bind(
            Spec<uint8_t>({conf["n_history_actions"_], kPublicEventFeatures})),
        "obs:public_event_refs_"_.Bind(
            Spec<uint8_t>({conf["n_history_actions"_], kPublicEventRefs})),
        "obs:mask_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 14})),
        "obs:turn_events_"_.Bind(Spec<uint8_t>({conf["history_window"_], history::kEventWidth})),
        "obs:turn_event_refs_"_.Bind(Spec<uint8_t>({conf["history_window"_], history::kRefWidth})),
        "obs:chain_"_.Bind(Spec<uint8_t>({history::kChainRows, history::kChainWidth})),
        "obs:turn_activations_"_.Bind(Spec<uint8_t>({history::kActivationRows, history::kActivationWidth})),
        "obs:turn_ledger_"_.Bind(Spec<uint8_t>({2, history::kLedgerWidth})),
        "obs:player_hints_"_.Bind(Spec<uint8_t>({history::kHintRows, history::kHintWidth})),
        "obs:card_turn_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, history::kCardTurnWidth})),
        "obs:turn_chunks_"_.Bind(
            Spec<uint8_t>({conf["history_chunk_slots"_], conf["history_chunk"_], history::kEventWidth})),
        "obs:turn_chunk_refs_"_.Bind(
            Spec<uint8_t>({conf["history_chunk_slots"_], conf["history_chunk"_], history::kRefWidth})),
        "obs:turn_chunk_meta_"_.Bind(Spec<int>({conf["history_chunk_slots"_], history::kChunkMetaWidth})),
        "obs:closed_turns_"_.Bind(Spec<uint8_t>({history::kClosedTurns, conf["history_window"_], history::kEventWidth})),
        "obs:closed_turn_refs_"_.Bind(Spec<uint8_t>({history::kClosedTurns, conf["history_window"_], history::kRefWidth})),
        "obs:closed_turn_meta_"_.Bind(Spec<int>({history::kClosedTurns, history::kClosedMetaWidth})),
        "obs:unpositioned_"_.Bind(Spec<uint8_t>({history::kUnpositionedRows, history::kUnpositionedWidth})),
        "obs:card_status_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, public_effects::kStatusWidth})),
        "obs:public_effects_"_.Bind(Spec<uint8_t>({public_effects::kEffectRows, public_effects::kEffectWidth})),
        "obs:candidates_"_.Bind(Spec<uint8_t>({conf["max_options"_], 3})),
        "label:hidden_"_.Bind(Spec<uint8_t>({kHiddenLabelRows, 4})),
        "obs:opponent_recipe_"_.Bind(Spec<uint8_t>({kRecipeRows, 4})),
        "obs:own_recipe_"_.Bind(Spec<uint8_t>({kRecipeRows, kOwnRecipeWidth})),
        "info:turn_events_dropped"_.Bind(Spec<int>({})),
        "info:turn_chunk_backlog"_.Bind(Spec<int>({})),
        "info:closed_turn_backlog"_.Bind(Spec<int>({})),
        "info:public_effects_dropped"_.Bind(Spec<int>({})),
        "info:turn_rows"_.Bind(Spec<int>({2})),
        "info:closed_turn_rows"_.Bind(Spec<int>({history::kClosedTurns, 2})),
        "info:announce_truncated"_.Bind(Spec<int>({2})),
        "info:announce_empty_union"_.Bind(Spec<int>({}, {0, 1})),
        "info:announce_fixed"_.Bind(Spec<int>({})),
        "info:step_limit"_.Bind(Spec<int>({3})),
        "info:guard"_.Bind(Spec<int>({3})),
        "info:illegal_activation"_.Bind(Spec<int>({6})),
        "info:num_options"_.Bind(Spec<int>({}, {0, conf["max_options"_]})),
        "info:to_play"_.Bind(Spec<int>({}, {0, 1})),
        "info:is_selfplay"_.Bind(Spec<int>({}, {0, 1})),
        "info:win_reason"_.Bind(Spec<int>({}, {-1, 1})),
        "info:step_time"_.Bind(Spec<double>({2})),
        "info:deck"_.Bind(Spec<int>({2})),
        "info:anchor_seat"_.Bind(Spec<int>({}, {0, 2}))
      );
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config &conf) {
    return MakeDict(
        "action"_.Bind(Spec<int>({}, {0, conf["max_options"_] - 1})));
  }
};

using DuelEnvSpec = EnvSpec<DuelEnvFns>;

enum PlayMode { kHuman, kSelfPlay, kRandomBot, kGreedyBot, kCount };

// parse play modes seperated by '+'
inline std::vector<PlayMode> parse_play_modes(const std::string &play_mode) {
  std::vector<PlayMode> modes;
  std::istringstream ss(play_mode);
  std::string token;
  while (std::getline(ss, token, '+')) {
    if (token == "human") {
      modes.push_back(kHuman);
    } else if (token == "self") {
      modes.push_back(kSelfPlay);
    } else if (token == "bot") {
      modes.push_back(kGreedyBot);
    } else if (token == "random") {
      modes.push_back(kRandomBot);
    } else {
      throw std::runtime_error("Unknown play mode: " + token);
    }
  }
  // human mode can't be combined with other modes
  if (std::find(modes.begin(), modes.end(), kHuman) != modes.end() &&
      modes.size() > 1) {
    throw std::runtime_error("Human mode can't be combined with other modes");
  }
  return modes;
}

// rules = 1, Traditional
// rules = 0, Default
// rules = 4, Link
// rules = 5, MR5
constexpr int32_t rules_ = 5;
constexpr int32_t duel_options_ = ((rules_ & 0xFF) << 16) + (0 & 0xFFFF);


// The illegal-activation withdrawal (DuelEnvImpl::step); in the module's guard_laws and so in checkpoint identities.
constexpr const char *kIllegalActivationLaw = "illegal_activation_withdrawal/v1";
// Observation laws that keep the env's arrays equal to what a client can build (client_driver.h): monsters' level,
// ATK, DEF and status as the server last refreshed them, and each player's candidate-selection source from its own
// decisions only. In checkpoint identities.
constexpr const char *kCardViewLaw = "refreshed_view/v1";
constexpr const char *kPendingSourceLaw = "own_decisions/v1";
// Place/disfield source identity comes from the selecting seat's received card
// hint; an entity reference additionally needs an unchanged own choice or one
// uniquely visible card. No private in-progress engine card is consulted.
constexpr const char *kPlacementRefLaw = "received_placement_card_ref/v1";
// obs:unpositioned_'s deck and face-down extra deck rules (unpositioned.h OpponentZones): a shown card that stays in
// its location re-enters the known multiset when the reveal ends (Area Zero's excavated cards shuffled back).
constexpr const char *kUnpositionedLaw = "reveal_returns/v1";
// obs:card_status_, obs:public_effects_ and the full reason in history rows (public_status.h; the table's SHA-256 is
// public_effects::kTableSha256).
constexpr const char *kPublicEffectsLaw = public_effects::kLaw;

// public_effects/v1 is reviewed card by card: a pool duel or a client whose cards include one outside the table is
// refused, unless its config allows unreviewed cards (allow_unreviewed_public_effects: tests and audits of other
// pools, whose cards' lingering effects are then not shown).
template <class Codes>
void RequireReviewed(const Codes &codes, bool allow_unreviewed, const std::string &where) {
  if (allow_unreviewed) return;
  std::set<uint32_t> outside;
  for (const auto code : codes)
    if (code != 0 && !public_effects::Reviewed(history::ArtworkBase(static_cast<uint32_t>(code))))
      outside.insert(static_cast<uint32_t>(code));
  if (!outside.empty())
    throw std::runtime_error(fmt::format("{}: card(s) {} are outside the public effect table ({}, table {}); review "
                                         "them, or allow unreviewed cards (allow_unreviewed_public_effects) for a "
                                         "test or audit", where, fmt::join(outside, ", "), kPublicEffectsLaw,
                                         public_effects::kTableSha256));
}

class DuelEnvImpl {
  friend class SearchDuel;

protected:
  const EnvSpec<DuelEnvFns> spec_;

  constexpr static int init_lp_ = 8000;
  constexpr static int startcount_ = 5;
  constexpr static int drawcount_ = 1;

  const std::string deck1_;
  const std::string deck2_;
  const std::string deck_schedule_;
  const std::string anchor_deck_;
  const int anchor_probability_percent_;
  int mirror_probability_percent_ = 0;
  int matched_probability_percent_ = 0;
  int pool_deck_uniform_ = 0;
  uint8_t room_format_ = 0, room_era_ = 0;  // obs:global_ 23 and 24 (config room_format, room_era)

  std::vector<uint32> main_deck0_;
  std::vector<uint32> main_deck1_;
  std::vector<uint32> extra_deck0_;
  std::vector<uint32> extra_deck1_;

  // The players' recipes as obs:own_recipe_ rows without the remaining counts (fixed for a game; set with the decks).
  struct RecipeRow {
    CardCode code = 0;
    CardId id = 0;
    uint8_t location = 0, count = 0, share = 0;
  };
  std::array<std::vector<RecipeRow>, 2> own_recipe_rows_;
  // Whether a card may sit hidden in the control of a player that does not own it, per possible owner: set (for the
  // rest of the game) by a move between controllers or a swap (note_control_changes); only then can an owner's deck
  // and extra deck stop being what its received stream determines (unseen_own_cards).
  std::array<bool, 2> control_changed_{};
  // Search worlds: whether a player's hidden identities were replaced (search_api.h ReplaceHidden). Its own view is
  // then refused until restore: its private history still holds the truth's identities.
  std::array<bool, 2> replaced_{};

  std::string deck_name_[2] = {"", ""};
  std::string nickname_[2] = {"Alice", "Bob"};

  const std::vector<PlayMode> play_modes_;

  // if play_mode_ == 'bot' or 'human', player_ is the order of the ai player
  // -1 means random, 0 and 1 means the first and second player respectively
  const int player_;

  PlayMode play_mode_;
  bool verbose_ = false;

  PlayerId ai_player_;

  intptr_t pduel_ = 0;
  std::unique_ptr<Player> players_[2]; //  abstract class must be pointer

  std::uniform_int_distribution<uint64_t> dist_int_;
  bool done_{true};
  long step_count_{0};
  bool duel_started_{false};
  bool keep_core_after_end_{false};  // search_api.h: the core duel outlives the game's end
  uint32_t eng_flag_{0};
  uint32_t disabled_field_{0};

  PlayerId winner_;
  uint8_t win_reason_;
  const bool greedy_reward_;

  int lp_[2];

  // turn player
  PlayerId tp_;
  int current_phase_;
  int turn_count_;

  int msg_;
  std::vector<LegalAction> legal_actions_;
  PlayerId to_play_;
  std::function<void(int)> callback_;

  byte data_[4096];
  int dp_ = 0;
  int dl_ = 0;

  byte query_buf_[4096];
  int qdp_ = 0;

  byte resp_buf_[128];

  using IdleCardSpec = std::tuple<CardCode, std::string, uint32_t>;

  // chain
  PlayerId chaining_player_;

  const int n_history_actions_;

  // circular buffer for history actions
  TArray<uint8_t> history_actions_1_;
  TArray<uint8_t> history_actions_2_;
  int ha_p_1_ = 0;
  int ha_p_2_ = 0;

  // Cards a confirm message showed, by (controller, location, sequence), with the players who saw them (bit p for
  // player p), as the host routes the message (netduel/host_view.py): deck-top and extra-top excavations and
  // confirms of cards outside the deck reach both players, a confirm of deck cards only the confirming player.
  // Forgotten when a chain link resolves, or when a move or shuffle may have displaced the card (end_reveals). The
  // code is the card the message showed there (the audit checks the place still holds it).
  // obs:candidates_ per player: the public candidate law's list for the seen sets of these sizes (they only grow
  // within a duel, so the sizes name the contents; restored with the env state by search_api.h).
  struct CandidateCache {
    size_t own = SIZE_MAX, opponent = SIZE_MAX;
    std::vector<std::pair<CardCode, int>> rows;  // code, tier
  };
  std::array<CandidateCache, 2> candidate_cache_;
  struct Reveal {
    uint8_t viewers = 0;
    CardCode code = 0;
  };
  std::map<std::tuple<uint8_t, uint8_t, uint8_t>, Reveal> revealed_;
  std::vector<PublicEvent> public_events_;

  // Public history (history_obs.h): every core buffer and multi-option decision, and the card rows of the last
  // written observation as the public tracker sees them.
  history::History history_;
  mfenv::TrackerView obs_view_;

  // Scripted duels (scripted_driver.h): the deal's duel in place of a sampled one, and every response the core is
  // given, as the bytes a network client would send.
  std::function<MDuel()> scripted_duel_;
  std::vector<std::vector<uint8_t>> *response_log_ = nullptr;
  // Every response and menu index of the current game, for the fatal repro
  // record; the message offsets of the current core buffer.
  std::vector<std::vector<uint8_t>> repro_responses_;
  std::vector<int> repro_actions_;
  std::vector<int> message_starts_;
  // Exact resume: both generators as they were when the current game started, and a running FNV-1a digest of
  // every core buffer of the game (reset by reset()).
  std::string start_gen_, start_duel_gen_;
  uint64_t stream_hash_ = 1469598103934665603ULL;
  uint32_t duel_seed_ = 0;
  // Set by a summon, special summon or set command; read and cleared by the
  // next message (handle_message).
  bool procedure_choice_ = false;

  // The last announce prompt's law result counters (announce_law.h): belief and empty-union truncation, the branch.
  std::array<int, 2> announce_truncated_{};
  bool announce_empty_union_ = false;
  int announce_fixed_ = 0;  // candidates in tiers 1-4 (never truncated; must fit the cap)

  // step_limit_timeout_loss/v1: each player's multi-option decisions in the turn of the latest one (every
  // sub-selection round counts, forced one-option prompts do not), and whether the limit ended the game.
  int decisions_turn_ = -1;
  std::array<int, 2> turn_decisions_{};
  bool step_limited_ = false;

  // No-progress guards (guards.h): their state, the rows of the full menu the player was shown at the current prompt
  // (step indexes them), the guarded prompt to record the choice at, and this prompt's counts for info:guard.
  guards::Guards guards_;
  std::vector<int> visible_;
  std::optional<guards::Prompt> guard_prompt_;
  std::array<int, 3> guard_info_{};

  // illegal_activation_withdrawal/v1 (at step): the frames of the activation in progress, the rows of the full menu
  // withheld at the current prompt, the next observation's info:illegal_activation (decisions withdrawn of player 0
  // and 1, the card, prompts withdrawn past, shortfalls outside a frame and the last one's card), every withdrawal of
  // the game for the repro record (menu indices so far, the row withheld, the card, prompts withdrawn past), and the
  // core's shortfall count at the last check; whether the current response is inside a frame and its activation was
  // chained.
  struct Frame;
  std::vector<std::shared_ptr<const Frame>> activation_frames_;

  // Client mode (client_driver.h ClientDuel): no engine; the messages come from one seat's received stream and every
  // card query is answered from the client's card view, which the caller sets per zone (set_client_cards).
 public:
  struct ClientCard {
    uint32_t code = 0;
    uint8_t controller = 0, location = 0, sequence = 0, position = 0;
    uint32_t level = 0, rank = 0;
    int32_t attack = 0, defense = 0;
    uint32_t equip = 0;  // the equip target's info location, 0 for none
    std::vector<uint32_t> overlay;
    std::vector<std::pair<uint32_t, uint32_t>> counters;  // (type, count)
    uint32_t owner = 0, status = 0, lscale = 0, rscale = 0, link = 0, link_marker = 0;
    // whether a server refresh filled the stats in; a card the stream only named (a deck card, one followed into a
    // zone the server does not refresh) shows the card database's stats, as the engine reports them off the field
    bool stats_known = false;
  };
 protected:
  bool client_mode_ = false;
  std::map<std::pair<int, int>, std::vector<std::optional<ClientCard>>> client_cards_;

  // card_view_law refreshed_view/v1 (env mode): a monster's level, ATK, DEF and status in the observation are what
  // the duel server last sent the clients for its slot (MSG_UPDATE_DATA / MSG_UPDATE_CARD: the refresh schedule of
  // netduel/host_view.py, queried at the same point of the stream), not a fresh engine query; a client holds no
  // more. A card that moves within the monster zone keeps its values until the next refresh.
  struct RefreshedStats {
    bool valid = false;
    uint32_t level = 0, status = 0;
    int32_t attack = 0, defense = 0;
  };
  std::array<std::array<RefreshedStats, 7>, 2> refreshed_mzone_{};
  std::set<int> illegal_held_;
  std::array<int, 6> illegal_info_{};
  std::vector<std::array<int64_t, 4>> illegal_log_;
  int32_t shortfall_seen_ = 0;
  bool shortfall_armed_ = false;
  bool chained_seen_ = false;

  // Each player's last own source action (an activation, summon or set, by its place, card and effect), the source
  // of its later candidate selections (pending_source_law: own decisions only, so a client, which never sees the
  // opponent's decisions, keeps the same); a phase command clears it.
  std::array<std::string, 2> pending_source_spec_;
  std::array<CardId, 2> pending_source_cid_{};
  std::array<int, 2> pending_source_effect_{-1, -1};
  struct PlacementAnchor {
    std::string spec;
    CardId cid = 0;
  };
  std::array<PlacementAnchor, 2> placement_source_;
  std::array<CardId, 2> placement_hint_{};
  CardId placement_prompt_cid_ = 0;

  // multi select
  int ms_idx_ = -1;
  int ms_mode_ = 0;
  int ms_min_ = 0;
  int ms_max_ = 0;
  int ms_must_ = 0;
  std::vector<std::string> ms_specs_;
  std::vector<std::vector<int>> ms_combs_;
  ankerl::unordered_dense::map<std::string, int> ms_spec2idx_;
  std::vector<int> ms_r_idxs_;

  // discard hand cards
  bool discard_hand_ = false;

  // replay
  bool record_ = false;
  FILE* fp_ = nullptr;
  bool is_recording = false;

  // MSG_SELECT_COUNTER
  int n_counters_ = 0;

  std::mt19937 gen_;

  std::mt19937 duel_gen_;


public:
  // step return
  float ret_reward_ = 0;
  int ret_win_reason_ = 0;

  DuelEnvImpl();

  DuelEnvImpl(const EnvSpec<DuelEnvFns> &spec, uint64_t env_seed)
      : spec_(spec), dist_int_(0, 0xffffffff),
        deck1_(spec.config["deck1"_]), deck2_(spec.config["deck2"_]),
        deck_schedule_(spec.config["deck_schedule"_]),
        anchor_deck_(spec.config["anchor_deck"_]),
        anchor_probability_percent_(
            spec.config["anchor_probability_percent"_]),
        player_(spec.config["player"_]), players_{nullptr, nullptr},
        play_modes_(parse_play_modes(spec.config["play_mode"_])),
        verbose_(spec.config["verbose"_]), record_(spec.config["record"_]),
        n_history_actions_(spec.config["n_history_actions"_]),
        greedy_reward_(spec.config["greedy_reward"_]) {
    mirror_probability_percent_ =
        spec.config["mirror_probability_percent"_];
    matched_probability_percent_ =
        spec.config["matched_probability_percent"_];
    pool_deck_uniform_ = spec.config["pool_deck_uniform"_];
    for (const int value : {static_cast<int>(spec.config["room_format"_]), static_cast<int>(spec.config["room_era"_])})
      if (value < 0 || value > 255) throw std::runtime_error(fmt::format("room format/era {} outside 0..255", value));
    room_format_ = static_cast<uint8_t>(static_cast<int>(spec.config["room_format"_]));
    history_.SetLaw({spec.config["history_window"_], spec.config["history_chunk"_], spec.config["history_chunk_cap"_],
                     spec.config["history_chunk_slots"_]});
    history_.SetPendingCap(spec.config["max_steps"_]);
    room_era_ = static_cast<uint8_t>(static_cast<int>(spec.config["room_era"_]));
    if (anchor_probability_percent_ < 0 ||
        anchor_probability_percent_ > 100) {
      throw std::runtime_error(
          "anchor_probability_percent must be between 0 and 100");
    }
    if (record_) {
      if (!verbose_) {
        throw std::runtime_error("record mode must be used with verbose mode and num_envs=1");
      }
    }
    // fmt::println("env_id: {}, seed: {}, x: {}", env_id_, seed_, dist_int_(gen_));

    gen_ = std::mt19937(env_seed);
    duel_gen_ = std::mt19937(dist_int_(gen_));

    int max_options = spec.config["max_options"_];
    int n_action_feats = spec.state_spec["obs:actions_"_].shape[1];
    history_actions_1_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
    history_actions_2_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
  }

  int max_options() const { return spec_.config["max_options"_]; }

  int max_cards() const { return spec_.config["max_cards"_]; }

  bool done() const { return done_; }

  bool random_mode() const { return play_modes_.size() > 1; }

  bool self_play() const {
    return std::find(play_modes_.begin(), play_modes_.end(), kSelfPlay) !=
           play_modes_.end();
  }

  void update_time_stat(const clock_t& start, uint64_t time_count, double& time_stat) {
    double seconds = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    time_stat = time_stat * (static_cast<double>(time_count) /
      (time_count + 1)) + seconds / (time_count + 1);
  }

  // void update_time_stat(const std::string& deck, double seconds) {
  //   uint64_t& time_count = deck_time_count_[deck];
  //   double& time_stat = deck_time_[deck];
  //   time_stat = time_stat * (static_cast<double>(time_count) /
  //     (time_count + 1)) + seconds / (time_count + 1);
  //   time_count++;
  // }

  std::string sample_deck_name(
      const std::string &configured_name, std::mt19937 &gen) const {
    if (configured_name != "random") {
      return configured_name;
    }
    if (deck_names_.empty()) {
      throw std::runtime_error("No decks are available for random sampling");
    }
    std::uniform_int_distribution<size_t> distribution(
        0, deck_names_.size() - 1);
    return deck_names_[distribution(gen)];
  }

  std::string sample_cluster_deck_name(
      const std::string &configured_name, std::mt19937 &gen) const {
    if (configured_name != "random") {
      return configured_name;
    }
    if (deck_clusters_.empty()) {
      throw std::runtime_error(
          "No deck clusters are available for random sampling");
    }
    std::uniform_int_distribution<size_t> cluster_distribution(
        0, deck_clusters_.size() - 1);
    const auto &cluster = deck_clusters_[cluster_distribution(gen)];
    std::uniform_int_distribution<size_t> variant_distribution(
        0, cluster.size() - 1);
    return cluster[variant_distribution(gen)];
  }

  std::string sample_pool_deck_name(std::mt19937 &gen) const {
    // Opponent-side draw for the expert / mix schedules.  The default
    // cluster_uniform gives every cluster the same mass, which leaves the
    // large sky-family cluster at 1/(number of clusters); pool_deck_uniform
    // draws a deck uniformly instead, i.e. weights each cluster by its deck
    // count, so a 64-deck cluster gets 64/273 of the draws.
    if (pool_deck_uniform_ != 0) {
      if (deck_names_.empty()) {
        throw std::runtime_error("No decks are available for random sampling");
      }
      std::uniform_int_distribution<size_t> deck_distribution(
          0, deck_names_.size() - 1);
      return deck_names_[deck_distribution(gen)];
    }
    return sample_cluster_deck_name("random", gen);
  }

  std::pair<std::string, std::string> sample_deck_pair(
      std::mt19937 &gen) const {
    if (deck_schedule_ == "independent") {
      return {
          sample_deck_name(deck1_, gen),
          sample_deck_name(deck2_, gen),
      };
    }
    if (deck_schedule_ == "cluster_uniform") {
      return {
          sample_cluster_deck_name(deck1_, gen),
          sample_cluster_deck_name(deck2_, gen),
      };
    }
    if (deck_schedule_ == "mix") {
      if (mirror_probability_percent_ < 0 || matched_probability_percent_ < 0 ||
          mirror_probability_percent_ + matched_probability_percent_ > 100) {
        throw std::runtime_error("mix requires 0 <= mirror+matched <= 100");
      }
      if (deck_clusters_.empty()) {
        throw std::runtime_error("mix requires deck clusters");
      }
      std::uniform_int_distribution<int> mix_percent(0, 99);
      std::uniform_int_distribution<size_t> cluster_pick(0, deck_clusters_.size() - 1);
      const int roll = mix_percent(gen);
      if (roll < mirror_probability_percent_) {
        const auto &cluster = deck_clusters_[cluster_pick(gen)];
        std::uniform_int_distribution<size_t> variant_pick(0, cluster.size() - 1);
        const std::string &deck = cluster[variant_pick(gen)];
        return {deck, deck};
      }
      if (roll < mirror_probability_percent_ + matched_probability_percent_) {
        const auto &cluster = deck_clusters_[cluster_pick(gen)];
        std::uniform_int_distribution<size_t> variant_pick(0, cluster.size() - 1);
        std::string first = cluster[variant_pick(gen)];
        std::string second = first;
        if (cluster.size() > 1) {
          while (second == first) {
            second = cluster[variant_pick(gen)];
          }
        }
        std::uniform_int_distribution<int> side(0, 1);
        if (side(gen) != 0) {
          std::swap(first, second);
        }
        return {first, second};
      }
      return {
          sample_cluster_deck_name(deck1_, gen),
          sample_cluster_deck_name(deck2_, gen),
      };
    }
    if (deck_schedule_ == "expert") {
      // Expert schedule: the anchor deck (the deck we are specialising on)
      // takes the field far more often than cluster_uniform would give it.
      //   roll <  mirror                      -> anchor vs anchor (true mirror)
      //   roll <  mirror + matched            -> anchor vs random pool deck
      //   otherwise                           -> random pool pair
      if (mirror_probability_percent_ < 0 || matched_probability_percent_ < 0 ||
          mirror_probability_percent_ + matched_probability_percent_ > 100) {
        throw std::runtime_error("expert requires 0 <= mirror+matched <= 100");
      }
      if (deck_clusters_.empty()) {
        throw std::runtime_error("expert requires deck clusters");
      }
      if (anchor_deck_.empty() ||
          main_decks_.find(anchor_deck_) == main_decks_.end()) {
        throw std::runtime_error("expert requires a valid anchor_deck");
      }
      std::uniform_int_distribution<int> expert_percent(0, 99);
      const int expert_roll = expert_percent(gen);
      if (expert_roll < mirror_probability_percent_) {
        return {anchor_deck_, anchor_deck_};
      }
      if (expert_roll <
          mirror_probability_percent_ + matched_probability_percent_) {
        std::pair<std::string, std::string> pair = {
            anchor_deck_, sample_pool_deck_name(gen)};
        std::uniform_int_distribution<int> expert_seat(0, 1);
        if (expert_seat(gen) != 0) {
          std::swap(pair.first, pair.second);
        }
        return pair;
      }
      return {
          sample_pool_deck_name(gen),
          sample_pool_deck_name(gen),
      };
    }
    if (deck_schedule_ != "anchor_half") {
      throw std::runtime_error(
          "Unknown deck schedule: " + deck_schedule_);
    }
    if (anchor_deck_.empty() ||
        main_decks_.find(anchor_deck_) == main_decks_.end()) {
      throw std::runtime_error(
          "anchor_half requires a valid anchor_deck");
    }

    std::vector<std::string> alternatives;
    alternatives.reserve(deck_names_.size());
    for (const auto &name : deck_names_) {
      if (name != anchor_deck_) {
        alternatives.push_back(name);
      }
    }
    if (alternatives.empty()) {
      throw std::runtime_error(
          "anchor_half requires at least one non-anchor deck");
    }

    std::uniform_int_distribution<int> percent(0, 99);
    std::uniform_int_distribution<size_t> pick(0, alternatives.size() - 1);
    if (percent(gen) < anchor_probability_percent_) {
      std::pair<std::string, std::string> pair = {
          anchor_deck_, alternatives[pick(gen)]};
      std::uniform_int_distribution<int> seat(0, 1);
      if (seat(gen) != 0) {
        std::swap(pair.first, pair.second);
      }
      return pair;
    }

    const auto first_index = pick(gen);
    auto second_index = first_index;
    if (alternatives.size() > 1) {
      while (second_index == first_index) {
        second_index = pick(gen);
      }
    }
    return {alternatives[first_index], alternatives[second_index]};
  }

  MDuel new_duel(uint32_t seed) {
    auto pduel = OCG_CreateDuel(seed);
    dormant::CreateIn(pduel);  // the registered table's dormant identities, before the decks (dormant_law.h)
    MDuel mduel{pduel, seed};
    const auto [deck_name0, deck_name1] = sample_deck_pair(duel_gen_);

    for (PlayerId i = 0; i < 2; i++) {
      OCG_SetPlayerInfo(pduel, i, init_lp_, startcount_, drawcount_);
      const auto &deck_name = i == 0 ? deck_name0 : deck_name1;
      auto [main_deck, extra_deck, loaded_deck_name] =
          load_deck(pduel, i, deck_name, duel_gen_);
      if (i == 0) {
        mduel.main_deck0 = main_deck;
        mduel.extra_deck0 = extra_deck;
        mduel.deck_name0 = loaded_deck_name;
      } else {
        mduel.main_deck1 = main_deck;
        mduel.extra_deck1 = extra_deck;
        mduel.deck_name1 = loaded_deck_name;
      }
    }
    OCG_StartDuel(pduel, duel_options_);
    return mduel;
  }

  void reset() {
    // clock_t start = clock();
    start_gen_ = generator_text(gen_);
    start_duel_gen_ = generator_text(duel_gen_);
    stream_hash_ = 1469598103934665603ULL;

    if (random_mode()) {
      play_mode_ = play_modes_[dist_int_(gen_) % play_modes_.size()];
    } else {
      play_mode_ = play_modes_[0];
    }

    if (play_mode_ != kSelfPlay) {
      if (player_ == -1) {
        ai_player_ = dist_int_(gen_) % 2;
      } else {
        ai_player_ = player_;
      }
    }

    turn_count_ = 0;
    disabled_field_ = 0;
    ms_idx_ = -1;

    history_actions_1_.Zero();
    history_actions_2_.Zero();
    ha_p_1_ = 0;
    ha_p_2_ = 0;
    public_events_.clear();
    history_.Reset(history::CardRows(card_ids_));
    candidate_cache_ = {};
    decisions_turn_ = -1;
    turn_decisions_ = {0, 0};
    step_limited_ = false;
    guards_ = guards::Guards{};
    visible_.clear();
    guard_prompt_.reset();
    guard_info_ = {0, 0, 0};
    refreshed_mzone_ = {};
    activation_frames_.clear();
    illegal_held_.clear();
    illegal_info_ = {0, 0, 0, 0, 0, 0};
    illegal_log_.clear();
    shortfall_seen_ = 0;
    shortfall_armed_ = false;
    chained_seen_ = false;
    pending_source_spec_ = {};
    pending_source_cid_ = {0, 0};
    pending_source_effect_ = {-1, -1};
    placement_source_ = {};
    placement_hint_ = {};
    placement_prompt_cid_ = 0;

    // clock_t _start = clock();

    intptr_t old_duel = pduel_;
    if (duel_started_) {
      OCG_EndDuel(pduel_);
    }
    MDuel mduel;
    mduel = scripted_duel_ ? scripted_duel_() : new_duel(dist_int_(gen_));

    auto duel_seed = mduel.seed;
    duel_seed_ = duel_seed;
    repro_responses_.clear();
    repro_actions_.clear();
    pduel_ = mduel.pduel;

    deck_name_[0] = mduel.deck_name0;
    deck_name_[1] = mduel.deck_name1;
    main_deck0_ = mduel.main_deck0;
    extra_deck0_ = mduel.extra_deck0;
    main_deck1_ = mduel.main_deck1;
    extra_deck1_ = mduel.extra_deck1;
    own_recipe_rows_ = {recipe_rows(main_deck0_, extra_deck0_), recipe_rows(main_deck1_, extra_deck1_)};
    if (!scripted_duel_)  // a pool duel (training, evaluation); a client checks its cards as they come
      for (const auto *deck : {&main_deck0_, &extra_deck0_, &main_deck1_, &extra_deck1_})
        RequireReviewed(*deck, spec_.config["allow_unreviewed_public_effects"_],
                        fmt::format("pool duel {} vs {}", deck_name_[0], deck_name_[1]));
    control_changed_ = {false, false};
    replaced_ = {false, false};

    for (PlayerId i = 0; i < 2; i++) {
      std::string nickname = i == 0 ? "Alice" : "Bob";
      if (i == ai_player_) {
        nickname = "Agent";
      }
      nickname_[i] = nickname;
      if ((play_mode_ == kHuman) && (i != ai_player_)) {
        players_[i] = std::make_unique<HumanPlayer>(nickname, init_lp_, i, verbose_);
      } else if (play_mode_ == kRandomBot) {
        players_[i] = std::make_unique<RandomAI>(max_options(), dist_int_(gen_), nickname, init_lp_, i, verbose_);
      } else {
        players_[i] = std::make_unique<GreedyAI>(nickname, init_lp_, i, verbose_);
      }
      lp_[i] = players_[i]->init_lp_;
    }

    if (record_) {
      if (is_recording && fp_ != nullptr) {
        fclose(fp_);
      }
      auto time_str = time_now();
      // Use last 4 digits of seed as unique id
      auto seed_ = duel_seed % 10000;
      std::string fname;
      while (true) {
        fname = fmt::format("./replay/a{} {:04d}.yrp", time_str, seed_);
        // check existence
        if (std::filesystem::exists(fname)) {
          seed_ = (seed_ + 1) % 10000;
        } else {
          break;
        } 
      }
      fp_ = fopen(fname.c_str(), "wb");
      if (!fp_) {
        throw std::runtime_error("Failed to open file for replay: " + fname);
      }

      is_recording = true;

      ReplayHeader rh;
      rh.id = 0x31707279;
      rh.version = 0x00001360;
      rh.flag = REPLAY_UNIFORM;
      rh.seed = duel_seed;
      rh.start_time = (unsigned int)time(nullptr);
      fwrite(&rh, sizeof(rh), 1, fp_);

      for (PlayerId i = 0; i < 2; i++) {
        uint16_t name[20];
        memset(name, 0, 40);
        std::string name_str = fmt::format("{} {}", nickname_[i], deck_name_[i]);
        if (name_str.size() > 20) {
          // truncate
          name_str = name_str.substr(0, 20);
        }
        fmt::println("name: {}", name_str);
        str_to_uint16(name_str.c_str(), name);
        fwrite(name, 40, 1, fp_);
      }

      ReplayWriteInt32(init_lp_);
      ReplayWriteInt32(startcount_);
      ReplayWriteInt32(drawcount_);
      ReplayWriteInt32(duel_options_);

      for (PlayerId i = 0; i < 2; i++) {
        auto &main_deck = i == 0 ? main_deck0_ : main_deck1_;
        auto &extra_deck = i == 0 ? extra_deck0_ : extra_deck1_;
        ReplayWriteInt32(main_deck.size());
        for (auto code : main_deck) {
          ReplayWriteInt32(code);
        }
        ReplayWriteInt32(extra_deck.size());
        for (int j = int(extra_deck.size()) - 1; j >= 0; --j) {
          ReplayWriteInt32(extra_deck[j]);
        }
      }

    }

    duel_started_ = true;
    eng_flag_ = 0;
    winner_ = 255;
    win_reason_ = 255;
    discard_hand_ = false;

    done_ = false;
    step_count_ = 0;

    // update_time_stat(_start, reset_time_count_, reset_time_2_);
    // _start = clock();

    play_gen_ = generator_text(gen_);
    if (client_mode_) return;  // the stream arrives through ClientDuel::Feed
    next();

    ret_reward_ = 0;
    ret_win_reason_ = 0;
  }

  // The env generator as play begins (after reset's own draws): steps draw from it (the end phase's random discard),
  // so a clone of the game (rollout_pool.h) must start play from it.
  std::string play_gen_;

  void init_multi_select(
    int min, int max, int must, const std::vector<std::string> &specs,
    int mode = 0, const std::vector<std::vector<int>> &combs = {}) {
    ms_idx_ = 0;
    ms_mode_ = mode;
    ms_min_ = min;
    ms_max_ = max;
    ms_must_ = must;
    ms_specs_ = specs;
    ms_r_idxs_.clear();
    ms_spec2idx_.clear();

    for (int j = 0; j < ms_specs_.size(); ++j) {
      const auto &spec = ms_specs_[j];
      ms_spec2idx_[spec] = j;
    }

    if (ms_mode_ == 0) {
      for (int j = 0; j < ms_specs_.size(); ++j) {
        const auto &spec = ms_specs_[j];
        legal_actions_.push_back(LegalAction::from_spec(spec));
      }
    } else {
      ms_combs_ = combs;
      _callback_multi_select_2_prepare();
    }
  }

  void handle_multi_select() {
    legal_actions_.clear();
    if (ms_mode_ == 0) {
      for (int j = 0; j < ms_specs_.size(); ++j) {
        if (ms_spec2idx_.find(ms_specs_[j]) != ms_spec2idx_.end()) {
          legal_actions_.push_back(
            LegalAction::from_spec(ms_specs_[j]));
        }
      }
      if (ms_idx_ == ms_max_ - 1) {
        if (ms_idx_ >= ms_min_) {
          legal_actions_.push_back(LegalAction::finish());
        }
        callback_ = [this](int idx) {
          _callback_multi_select(idx, true);
        };
      } else if (ms_idx_ >= ms_min_) {
        legal_actions_.push_back(LegalAction::finish());
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };
      } else {
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };    
      }
    } else {
      _callback_multi_select_2_prepare();
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };
    }
  }

  int get_ms_spec_idx(const std::string &spec) const {
    auto it = ms_spec2idx_.find(spec);
    if (it != ms_spec2idx_.end()) {
      return it->second;
    }
    // TODO(2): find the root cause
    // print ms_spec2idx
    show_deck(0);
    show_deck(1);
    show_buffer();
    show_turn();
    fmt::println("MS: idx: {}, mode: {}, min: {}, max: {}, must: {}, specs: {}, combs: {}, r_idx: {}", ms_idx_, ms_mode_, ms_min_, ms_max_, ms_must_, ms_specs_, ms_combs_, ms_r_idxs_);
    fmt::print("ms_spec2idx: ");
    for (const auto &[k, v] : ms_spec2idx_) {
      fmt::print("({}, {}), ", k, v);
    }
    fmt::print("\n");
    return -1;
    // throw std::runtime_error("Spec not found: " + spec);
  }

  void _callback_multi_select_2(int idx) {
    const auto &action = legal_actions_[idx];
    if (action.finish_) {
      _callback_multi_select_2_finish();
      return;
    }

    const int selected_idx = get_ms_spec_idx(action.spec_);
    if (selected_idx == -1) {
      // TODO(2): find the root cause
      std::vector<std::string> specs;
      for (const auto &la : legal_actions_) {
        specs.push_back(la.spec_);
      }
      fmt::println(
          "specs: {}, idx: {}, spec: {}", specs, selected_idx, action.spec_);
      throw std::runtime_error("Spec not found");
    }

    bool can_finish = false;
    std::vector<std::vector<int>> remaining_combinations;
    for (const auto &combination : ms_combs_) {
      if (combination.empty() || combination.front() != selected_idx) {
        continue;
      }
      if (combination.size() == 1) {
        can_finish = true;
      } else {
        remaining_combinations.emplace_back(
            combination.begin() + 1, combination.end());
      }
    }

    if (can_finish) {
      remaining_combinations.push_back({});
    }
    if (remaining_combinations.empty()) {
      throw std::runtime_error("No valid continuation for multi select");
    }

    ms_r_idxs_.push_back(selected_idx);
    ms_idx_++;
    const bool has_continuation = std::any_of(
        remaining_combinations.begin(), remaining_combinations.end(),
        [](const std::vector<int> &combination) {
          return !combination.empty();
        });
    ms_combs_ = std::move(remaining_combinations);
    if (can_finish && !has_continuation) {
      _callback_multi_select_2_finish();
    }
  }

  void _callback_multi_select_2_prepare() {
    bool can_finish = false;
    std::set<int> candidates;
    for (const auto &combination : ms_combs_) {
      if (combination.empty()) {
        can_finish = true;
      } else {
        candidates.insert(combination.front());
      }
    }
    for (const auto index : candidates) {
      const auto &spec = ms_specs_[index];
      legal_actions_.push_back(LegalAction::from_spec(spec));
    }
    if (can_finish) {
      legal_actions_.push_back(LegalAction::finish());
    }
  }

  void _callback_multi_select_2_finish() {
    ms_idx_ = -1;
    resp_buf_[0] = ms_r_idxs_.size() + ms_must_;
    for (int i = 0; i < ms_must_; ++i) {
      resp_buf_[i + 1] = 0;
    }
    for (int i = 0; i < ms_r_idxs_.size(); ++i) {
      resp_buf_[i + ms_must_ + 1] = ms_r_idxs_[i];
    }
    OCG_SetResponseb(pduel_, resp_buf_);
  }

  void _callback_multi_select(int idx, bool finish) {
    const auto &action = legal_actions_[idx];
    // fmt::println("Select card: {}, finish: {}", option, finish);
    if (action.finish_) {
      finish = true;
    } else {
      idx = get_ms_spec_idx(action.spec_);
      if (idx != -1) {
        ms_r_idxs_.push_back(idx);
      } else {
        // TODO(2): find the root cause
        std::vector<std::string> specs;
        for (const auto &la : legal_actions_) {
          specs.push_back(la.spec_);
        }
        fmt::println("specs: {}, idx: {}, spec: {}", specs, idx, action.spec_);
        ms_idx_ = -1;
        resp_buf_[0] = ms_min_;
        for (int i = 0; i < ms_min_; ++i) {
          resp_buf_[i + 1] = i;
        }
        OCG_SetResponseb(pduel_, resp_buf_);
        return;
      }
    }
    if (finish) {
      ms_idx_ = -1;
      resp_buf_[0] = ms_r_idxs_.size();
      for (int i = 0; i < ms_r_idxs_.size(); ++i) {
        resp_buf_[i + 1] = ms_r_idxs_[i];
      }
      OCG_SetResponseb(pduel_, resp_buf_);
    } else {
      ms_idx_++;
      ms_spec2idx_.erase(action.spec_);
    }
  }

  SelectionRole selection_role_for_msg(int msg) const {
    if (msg == MSG_SELECT_TRIBUTE) {
      return SelectionRole::Tribute;
    }
    if (msg == MSG_SELECT_SUM) {
      return SelectionRole::Material;
    }
    if (msg == MSG_SELECT_CARD || msg == MSG_SELECT_UNSELECT_CARD) {
      return SelectionRole::Candidate;
    }
    return SelectionRole::None;
  }

  bool is_candidate_selection_msg(int msg) const {
    return msg == MSG_SELECT_CARD || msg == MSG_SELECT_TRIBUTE ||
           msg == MSG_SELECT_SUM || msg == MSG_SELECT_UNSELECT_CARD;
  }

  bool is_source_action_msg(int msg) const {
    return msg == MSG_SELECT_IDLECMD || msg == MSG_SELECT_BATTLECMD ||
           msg == MSG_SELECT_CHAIN || msg == MSG_SELECT_EFFECTYN;
  }

  std::string spec_for_observer(
      const std::string &spec, PlayerId source_player,
      PlayerId observer) const {
    if (spec.empty()) {
      return "";
    }
    const bool points_to_opponent = spec[0] == 'o';
    const auto core = points_to_opponent ? spec.substr(1) : spec;
    const PlayerId controller =
        points_to_opponent ? 1 - source_player : source_player;
    return controller == observer ? core : "o" + core;
  }

  void push_public_event(const PublicEvent &event) {
    public_events_.insert(public_events_.begin(), event);
    if (public_events_.size() > static_cast<size_t>(n_history_actions_)) {
      public_events_.pop_back();
    }
  }

  void record_protocol_event(
      PlayerId player, int msg, const std::vector<uint8_t> &payload) {
    PublicEvent event;
    event.actor = player;
    event.msg = msg;
    event.turn = turn_count_;
    event.phase = current_phase_;
    event.payload_size = static_cast<uint8_t>(
        std::min<size_t>(payload.size(), 5));
    for (int i = 0; i < event.payload_size; ++i) {
      event.payload[i] = payload[i];
    }
    push_public_event(event);
  }

  void record_public_event(PlayerId player, const LegalAction &action) {
    record_placement_choice(player, action);
    if (action.act_ == ActionAct::Cancel) {
      return;
    }

    PublicEvent event;
    event.actor = player;
    event.msg = action.msg_;
    event.act = action.act_;
    event.action_phase = action.phase_;
    event.selection_role = selection_role_for_msg(action.msg_);
    event.finish = action.finish_;
    event.cancel = action.act_ == ActionAct::Cancel;
    event.effect = action.effect_;
    event.turn = turn_count_;
    event.phase = current_phase_;
    event.selection_stage = std::max(0, ms_idx_);
    event.selected_count = static_cast<int>(ms_r_idxs_.size());
    event.choice = action.number_;
    if (event.choice == 0 && action.attribute_ != 0) {
      event.choice = attribute_to_id(action.attribute_);
    }

    if (is_candidate_selection_msg(action.msg_)) {
      event.candidate_spec = action.spec_;
      event.candidate_spec_player = player;
      event.source_spec = pending_source_spec_.at(player);
      event.source_spec_player = player;
    } else if (!action.spec_.empty()) {
      event.source_spec = action.spec_;
      event.source_spec_player = player;
    }

    push_public_event(event);

    if (is_source_action_msg(action.msg_) && !action.spec_.empty() &&
        action.act_ != ActionAct::Cancel) {
      pending_source_spec_.at(player) = action.spec_;
      pending_source_cid_.at(player) = action.cid_;
      pending_source_effect_.at(player) = action.effect_;
    } else if (action.phase_ != ActionPhase::None) {
      pending_source_spec_.at(player).clear();
      pending_source_cid_.at(player) = 0;
      pending_source_effect_.at(player) = -1;
    }
  }

  void record_placement_choice(PlayerId player, const LegalAction &action) {
    if (is_source_action_msg(action.msg_)) {
      placement_source_[player] = {};
      const bool procedure = action.act_ == ActionAct::Summon || action.act_ == ActionAct::SpSummon ||
                             action.act_ == ActionAct::MSet || action.act_ == ActionAct::Set;
      // An activated hand card is placed; an on-field effect's handler need
      // not be the card its effect will place, even if the names are equal.
      const auto handler = cards_.find(action.option_code_);
      const bool hand_activation = action.act_ == ActionAct::Activate && action.msg_ != MSG_SELECT_EFFECTYN &&
          handler != cards_.end() && (handler->second.type() & (TYPE_SPELL | TYPE_TRAP)) && !action.spec_.empty() &&
          std::get<0>(spec_to_ls(player, action.spec_)) == player &&
          std::get<1>(spec_to_ls(player, action.spec_)) == LOCATION_HAND;
      if ((procedure || hand_activation) && !action.spec_.empty() && action.cid_)
        placement_source_[player] = {action.spec_, action.cid_};
    }
    // A generic SELECT_CARD can be a cost/target, not the card later placed.
    // Never infer an instance from that choice merely because its name agrees.
    if (action.phase_ != ActionPhase::None || action.msg_ == MSG_SELECT_PLACE || action.msg_ == MSG_SELECT_DISFIELD) {
      placement_source_[player] = {};
    }
  }

  void update_history_actions(PlayerId player, const LegalAction& action) {
    if (action.act_ == ActionAct::Cancel) {
      return;
    }
    auto& ha_p = player == 0 ? ha_p_1_ : ha_p_2_;
    auto& history_actions = player == 0 ? history_actions_1_ : history_actions_2_;
    ha_p--;
    if (ha_p < 0) {
      ha_p = n_history_actions_ - 1;
    }
    history_actions[ha_p].Zero();
    _set_obs_action(history_actions, ha_p, action);
    // Spec index not available in history actions
    history_actions[ha_p](0) = 0;
    // history_actions[ha_p](12) = static_cast<uint8_t>(player);
    history_actions[ha_p](12) = static_cast<uint8_t>(turn_count_);
    history_actions[ha_p](13) = static_cast<uint8_t>(phase_to_id(current_phase_));
  }

  void show_deck(const std::vector<CardCode> &deck, const std::string &prefix) const {
    fmt::print("{} deck: [", prefix);
    for (int i = 0; i < deck.size(); i++) {
      fmt::print(" '{}'", c_get_card(deck[i]).name());
    }
    fmt::print(" ]\n");
  }

  void show_turn() const {
    fmt::println("turn: {}, phase: {}, tplayer: {}", turn_count_, phase_to_string(current_phase_), tp_);
  }

  void show_buffer() const {
    fmt::println("msg: {}, dp: {}, dl: {}", msg_to_string(msg_), dp_, dl_);
    for (int i = 0; i < dl_; ++i) {
      fmt::print("{:02x} ", data_[i]);
    }
    fmt::print("\n");
  }

  void show_deck(PlayerId player) const {
    fmt::print("Player {}'s deck: {}\n", player, deck_name_[player]);
    // show_deck(player == 0 ? main_deck0_ : main_deck1_, "Main");
    // show_deck(player == 0 ? extra_deck0_ : extra_deck1_, "Extra");
  }

  void show_history_actions(PlayerId player) const {
    const auto &ha = player == 0 ? history_actions_1_ : history_actions_2_;
    // print card ids of history actions
    for (int i = 0; i < n_history_actions_; ++i) {
      fmt::print("history {}\n", i);
      uint8_t msg_id = uint8_t(ha(i, 3));
      int msg = _msgs[msg_id - 1];
      fmt::print("msg: {},", msg_to_string(msg));
      uint8_t v1 = ha(i, 1);
      uint8_t v2 = ha(i, 2);
      CardId card_id = (static_cast<CardId>(v1) << 8) + static_cast<CardId>(v2);
      fmt::print(" {};", card_id);
      for (int j = 4; j < ha.Shape()[1]; j++) {
        fmt::print(" {}", uint8_t(ha(i, j)));
      }
      fmt::print("\n");
    }
  }

  // The observer's own multi-option decision as a history row (history_obs.h).
  void record_own_choice(const LegalAction &action) {
    history::OwnChoice choice;
    choice.msg = msg_;
    choice.act = static_cast<int>(action.act_);
    choice.effect = action.effect_;
    choice.card_row = action.cid_;
    if (!action.spec_.empty()) {
      const auto [controller, location, sequence, position] = spec_to_ls(to_play_, action.spec_);
      choice.has_place = true;
      choice.side = controller == to_play_ ? 0 : 1;
      // a material's spec ends with its index letter ("m3a"); spec_to_ls returns its host's location
      choice.overlay = std::isalpha(static_cast<unsigned char>(action.spec_.back())) != 0;
      choice.location = location & 0x7f;
      // a deck's order is hidden from both players (design item 5a), and the opponent's extra deck positions are
      // not public: their rows carry no sequence
      const bool hidden_order = choice.location == LOCATION_DECK || (choice.side == 1 && choice.location == LOCATION_EXTRA);
      choice.sequence = hidden_order ? 0 : sequence;
      choice.overlay_index = position;
    }
    history_.ConsumeOwnChoice(to_play_, choice);
  }

  // illegal_activation_withdrawal/v1. OCG ruling: an activation's targets are chosen after its cost is paid, so an
  // activation whose cost leaves no legal target cannot be activated. (The reference server instead lets it proceed:
  // Duel.SelectTarget returns no card and the script reads a nil target -- a script error, the link resolves
  // nothing.) The env follows the ruling. Before a response that activates an effect with a card target and a cost
  // (a cost function or a cost paid in its target function; query_activation_flags), and before every response after it until that activation is chained (MSG_CHAINED, or a
  // command or chain prompt), the env takes a frame: the duel's core arena and env state. When the core then reports
  // a target selection with fewer candidates than its minimum (query_target_shortfall), the env returns to the last
  // frame and withholds that response at its prompt; a prompt with every option withheld withdraws to the frame
  // before it (the cost choice, then the activation itself), and a prompt left with one option is answered inside
  // the env like any one-option prompt. Withdrawn decisions are not transitions: info:illegal_activation counts them
  // per player (the trainer drops that player's last decisions), with the activation's card, the prompts withdrawn
  // past, and shortfalls outside a frame (nothing to return to: counted, the game goes on). The repro record keeps
  // every menu index, a withdrawn one included, so a replay withdraws the same way.
  void step(int idx) {
    illegal_info_ = {0, 0, 0, 0, 0, 0};
    const int shown = idx;
    // ``idx`` indexes the menu the player was shown; the guards may have withheld rows of the full menu (guards.h)
    if (!visible_.empty()) idx = visible_.at(idx);
    const PlayerId player = to_play_;
    int row = idx;
    bool decision = true, advanced = false;
    while (true) {
      std::shared_ptr<const Frame> frame = illegal_guarded(row) ? take_frame(row, decision) : nullptr;
      shortfall_armed_ = frame != nullptr;
      chained_seen_ = false;
      try {
        if (decision) {
          repro_actions_.push_back(shown);
          decide(row);
        } else {
          answer_sole(row);
        }
      } catch (const IllegalActivation &shortfall) {
        row = withdraw(frame, shortfall.code);
        if (row < 0) break;  // the restored prompt is shown again
        decision = false;
        continue;
      }
      settle_frames(frame);
      advanced = decision;
      break;
    }
    if (advanced) step_count_++;
    finish_step(player);
  }

  // Raw response bytes for the current prompt (a recorded game's player, as its client sent them), bypassing the
  // menu: the core takes them as the server would; that player's own-choice bookkeeping (guards, its private history
  // row, history actions) is skipped, so only the other player's observations stay exact. The repro record's menu
  // indices cannot replay such a game.
  void respond_raw(const std::vector<uint8_t> &bytes) {
    if (done_ || legal_actions_.empty()) throw std::runtime_error("respond: no prompt is waiting");
    if (bytes.empty() || bytes.size() > 64) throw std::runtime_error("respond: a response is 1 to 64 bytes");
    if (client_mode_) throw std::runtime_error("respond: a client sends its responses, it does not take them");
    illegal_info_ = {0, 0, 0, 0, 0, 0};
    const PlayerId player = to_play_;
    byte buf[64] = {0};
    std::memcpy(buf, bytes.data(), bytes.size());
    visible_.clear();
    guard_prompt_.reset();
    illegal_held_.clear();
    activation_frames_.clear();
    ms_idx_ = -1;
    if (record_) {
      ReplayWriteInt8(static_cast<int8_t>(bytes.size()));
      fwrite(buf, bytes.size(), 1, fp_);
    }
    if (response_log_) response_log_->push_back(bytes);
    repro_responses_.push_back(bytes);
    set_responseb(pduel_, buf);
    legal_actions_.clear();
    next();
    step_count_++;
    finish_step(player);
  }

  // The player's choice of ``idx`` (a row of the full menu) at the current prompt, and the duel up to its next prompt.
  void decide(int idx) {
    if (guard_prompt_) guards_.Record(*guard_prompt_, idx);
    visible_.clear();
    guard_prompt_.reset();
    const auto selected_action = legal_actions_.at(idx);
    if (legal_actions_.size() > 1) {
      record_own_choice(selected_action);
      if (turn_count_ != decisions_turn_) {
        decisions_turn_ = turn_count_;
        turn_decisions_ = {0, 0};
      }
      ++turn_decisions_.at(to_play_);
    }
    callback_(idx);
    update_history_actions(to_play_, selected_action);
    record_public_event(to_play_, selected_action);

    if (verbose_) {
      show_decision(idx);
    }

    if (ms_idx_ != -1) {
      handle_multi_select();
    } else {
      next();
    }
  }

  // The one option a withdrawal left at the current prompt, answered inside the env as next() answers a one-option
  // prompt, and the duel up to its next prompt.
  void answer_sole(int row) {
    visible_.clear();
    guard_prompt_.reset();
    callback_(row);
    auto la = legal_actions_.at(row);
    la.msg_ = msg_;
    if (la.cid_ == 0 && !la.spec_.empty()) {
      la.cid_ = spec_to_card_id(la.spec_, to_play_);
    }
    update_history_actions(to_play_, la);
    record_public_event(to_play_, la);
    if (ms_idx_ != -1) {
      handle_multi_select();
    } else {
      next();
    }
  }

  struct IllegalActivation {
    uint32_t code;  // the handler of the chain link whose target selection fell short
  };

  // Whether a response at ``row`` takes a frame: inside an activation's frames, or a row activating an effect with a
  // card target and a cost.
  bool illegal_guarded(int row) {
    if (client_mode_) return false;  // the server resolves what it is sent: nothing to withdraw to
    if (!activation_frames_.empty()) return true;
    const LegalAction &action = legal_actions_.at(row);
    if (action.option_ < 0 || ms_idx_ != -1) return false;
    if (msg_ != MSG_SELECT_IDLECMD && msg_ != MSG_SELECT_BATTLECMD && msg_ != MSG_SELECT_CHAIN &&
        msg_ != MSG_SELECT_EFFECTYN)
      return false;
    uint32_t effect[2] = {0, 0};
    const int32_t flags = query_activation_flags(pduel_, action.option_, effect);
    if (flags < 0) throw std::runtime_error("query_activation_flags: invalid duel");
    if (flags == 0) {
      if (msg_ == MSG_SELECT_EFFECTYN) return false;  // a script's yes/no: no activation
      throw std::runtime_error(fmt::format("illegal-activation guard: {} option {} activates nothing in the core",
                                           msg_to_string(msg_), action.option_));
    }
    if (effect[0] != action.option_code_ || (msg_ != MSG_SELECT_EFFECTYN && effect[1] != action.option_desc_))
      throw std::runtime_error(fmt::format("illegal-activation guard: {} option {} is {} ({}) in the core, {} ({}) in "
                                           "the message", msg_to_string(msg_), action.option_, effect[0], effect[1],
                                           action.option_code_, action.option_desc_));
    // a card target and a cost (a cost function, or a cost paid in the target function, which reads
    // Effect.IsCostChecked): paying it may change the field between the activation's check and its target selection
    return (flags & 2) && (flags & (4 | 16));
  }

  // After each core buffer: a target selection that fell short since the last check withdraws inside a frame, and
  // is counted outside one.
  void check_target_shortfall() {
    if (client_mode_) return;
    uint32_t code = 0;
    const int32_t count = query_target_shortfall(pduel_, &code);
    if (count < 0) throw std::runtime_error("query_target_shortfall: invalid duel");
    if (count == shortfall_seen_) return;
    shortfall_seen_ = count;
    if (shortfall_armed_) throw IllegalActivation{code};
    ++illegal_info_[4];
    illegal_info_[5] = static_cast<int>(code);
  }

  // A response went through: its frame is kept while the activation is not yet chained.
  void settle_frames(const std::shared_ptr<const Frame> &frame) {
    shortfall_armed_ = false;
    illegal_held_.clear();
    const bool outside = done_ || chained_seen_ || msg_ == MSG_SELECT_IDLECMD || msg_ == MSG_SELECT_BATTLECMD ||
                         msg_ == MSG_SELECT_CHAIN;
    if (frame && !outside) {
      activation_frames_.push_back(frame);
    } else {
      activation_frames_.clear();
    }
  }

  // Returns to ``frame`` (and further back while a prompt has nothing left) and withholds the response taken there.
  // Returns the one option left at the restored prompt (answered inside the env), or -1 to show it again.
  int withdraw(std::shared_ptr<const Frame> frame, uint32_t code) {
    if (!frame) throw std::runtime_error("illegal-activation guard: a shortfall without a frame");
    const std::vector<int> actions = repro_actions_;
    std::vector<std::array<int64_t, 4>> log = illegal_log_;
    std::array<int, 6> info = illegal_info_;
    int popped = 0;
    while (true) {
      restore_frame(*frame);
      if (frame->decision) ++info.at(frame->player);
      std::set<int> held = illegal_held_;  // the rows already withheld at that prompt when the frame was taken
      held.insert(frame->row);
      std::vector<int> left;
      for (int row = 0; row < static_cast<int>(legal_actions_.size()); ++row)
        if (!held.count(row)) left.push_back(row);
      if (!left.empty()) {
        illegal_held_ = std::move(held);
        log.push_back({static_cast<int64_t>(actions.size()), frame->row, static_cast<int64_t>(code), popped});
        repro_actions_ = actions;
        illegal_log_ = std::move(log);
        info[2] = static_cast<int>(code);
        info[3] += popped;
        illegal_info_ = info;
        core_script_errors().clear();
        return left.size() == 1 ? left[0] : -1;
      }
      if (activation_frames_.empty())
        throw std::runtime_error(fmt::format("illegal-activation guard: card {} -- every option of the activation's "
                                             "first prompt was withdrawn", code));
      frame = activation_frames_.back();  // its restore sets the frames below it
      ++popped;
    }
  }

  // After a step: the step limit, and the reward when the game is over (``player`` moved).
  void finish_step(PlayerId player) {
    if (!done_ && (step_count_ >= spec_.config["max_steps"_])) {
      // step_limit_timeout_loss/v1 (a client timeout): in the turn current at the limit, the player who made more
      // multi-option decisions loses; equal counts are a draw
      const auto counts = step_limit_counts();
      _duel_end(counts[0] == counts[1] ? 2 : (counts[0] > counts[1] ? 1 : 0), 0);
      step_limited_ = true;
      done_ = true;
      legal_actions_.clear();
    }

    float reward = 0;
    int reason = 0;
    if (done_) {
      if (winner_ > 1) {
        if (record_ && is_recording && fp_ != nullptr) {
          fclose(fp_);
          is_recording = false;
        }
        ret_reward_ = 0;
        ret_win_reason_ = 0;
        return;
      }
      float base_reward;
      if (greedy_reward_) {
        if (winner_ == 0) {
          if (turn_count_ <= 1) {
            // FTK
            base_reward = 16.0;
          } else if (turn_count_ <= 3) {
            base_reward = 8.0;
          } else if (turn_count_ <= 5) {
            base_reward = 4.0;
          } else if (turn_count_ <= 7) {
            base_reward = 2.0;
          } else {
            base_reward = 0.5 + 1.0 / (turn_count_ - 7);
          }
        } else {
          if (turn_count_ <= 1) {
            base_reward = 8.0;
          } else if (turn_count_ <= 3) {
            base_reward = 4.0;
          } else if (turn_count_ <= 5) {
            base_reward = 2.0;
          } else {
            base_reward = 0.5 + 1.0 / (turn_count_ - 5);
          }
        }
      } else {
        base_reward = 1.0;
      }

      if (play_mode_ == kSelfPlay) {
        // if (spec_.config["oppo_info"_]) {
        if (false) {
          reward = winner_ == 0 ? base_reward : -base_reward;
        } else {
          // to_play_ is the previous player
          reward = winner_ == player ? base_reward : -base_reward;
        }
      } else {
        reward = winner_ == ai_player_ ? base_reward : -base_reward;
      }

      if (win_reason_ == 0x01) {
        reason = 1;
      } else if (win_reason_ == 0x02) {
        reason = -1;
      }

      if (record_) {
        if (!is_recording || fp_ == nullptr) {
          throw std::runtime_error("Recording is not started");
        }
        fclose(fp_);
        is_recording = false;
      }
    }


    // update_time_stat(start, step_time_count_, step_time_);
    // step_time_count_++;

    // double step_time = 0;
    // if (done_) {
    //   step_time = step_time_;
    //   step_time_ = 0;
    //   step_time_count_ = 0;
    // }

    // if (done_) {
    //   update_time_stat(deck_name_[0], step_time_);
    //   update_time_stat(deck_name_[1], step_time_);
    //   step_time_ = 0;
    //   step_time_count_ = 0;
    // }
    // if (step_time_count_ % 3000 == 0) {
    //   fmt::println("Step time: {:.3f}", step_time_ * 1000);
    // }
    ret_reward_ = reward;
    ret_win_reason_ = reason;
  }

  using DuelEnvSpec = EnvSpec<DuelEnvFns>;
  using State =
      Dict<typename DuelEnvSpec::StateKeys,
           typename SpecToTArray<typename DuelEnvSpec::StateSpec::Values>::Type>;

  // The registered announce law (announce_law.h) for ``player``'s prompt: the filter program, the player's recipe, the
  // cards its own facts have shown (tokens are no recipe cards and are left out), the registered tables and cap.
  // obs:candidates_ (S1): the public candidate law's list for the player to move with no filter -- the cards the
  // belief head scores -- as rows (card id high byte, low byte, tier), in the law's order.
  void _set_obs_candidates(TArray<uint8_t> out) {
    out.Zero();
    const auto &law = announce::Registered();
    const auto &seen = history_.Seen(to_play_);
    CandidateCache &cache = candidate_cache_[to_play_];
    if (cache.own != seen[0].size() || cache.opponent != seen[1].size()) {
      std::vector<uint32_t> card_table;
      card_table.reserve(card_ids_.size());
      for (const auto &[code, id] : card_ids_) card_table.push_back(code);
      auto card_id = [](uint32_t code) -> int64_t {
        const auto it = card_ids_.find(code);
        return it == card_ids_.end() ? 0 : it->second;
      };
      cache.rows.clear();
      for (const auto &[code, tier] : announce::BeliefCandidates(
               recipe_cards(seen[1]), recipe_cards(seen[0]), to_play_ == 0 ? main_deck0_ : main_deck1_,
               to_play_ == 0 ? extra_deck0_ : extra_deck1_, law.tables, law.room_format, law.cap, card_table, card_id))
        cache.rows.emplace_back(code, tier);
      cache.own = seen[0].size();
      cache.opponent = seen[1].size();
    }
    if (static_cast<int>(cache.rows.size()) > out.Shape()[0])
      throw std::runtime_error(fmt::format("{} belief candidates, {} rows", cache.rows.size(), out.Shape()[0]));
    for (size_t i = 0; i < cache.rows.size(); ++i) {
      const CardId id = c_get_card_id(cache.rows[i].first);
      out(i, 0) = static_cast<uint8_t>(id >> 8);
      out(i, 1) = static_cast<uint8_t>(id & 0xff);
      out(i, 2) = static_cast<uint8_t>(cache.rows[i].second);
    }
  }

  // label:hidden_ (privileged, S1 belief targets): the true identities of ``viewer``'s opponent's cards whose rows
  // show no code -- hand, deck, face-down field, face-down banished and face-down extra deck -- as rows (card id high
  // byte, low byte,
  // location id as in cards_, count) in (location, card id) order. More rows than the table holds throws.
  void _set_label_hidden(TArray<uint8_t> &out, PlayerId viewer) {
    std::map<std::pair<uint8_t, CardId>, int> counts;
    for (uint8_t location : {LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_REMOVED,
                             LOCATION_EXTRA})
      for (const Card &c : get_cards_in_location(1 - viewer, location))
        if (!(c.location_ & LOCATION_OVERLAY) && visible_code(viewer, c) == 0)
          ++counts[{location_to_id(location), c_get_card_id(c.code_)}];
    if (static_cast<int>(counts.size()) > out.Shape()[0])
      throw std::runtime_error(fmt::format("{} hidden label rows, {} in the table", counts.size(), out.Shape()[0]));
    int row = 0;
    for (const auto &[key, count] : counts) {
      out(row, 0) = static_cast<uint8_t>(key.second >> 8);
      out(row, 1) = static_cast<uint8_t>(key.second & 0xff);
      out(row, 2) = key.first;
      out(row, 3) = static_cast<uint8_t>(std::min(count, 255));
      ++row;
    }
  }

  // obs:opponent_recipe_ (with the public_opponent_recipe config: a mirror or an open-decklist mode declares the
  // opponent's decklist public, and search may then place particles drawn from it -- search plan S1/S2): the
  // opponent's deck list as rows (card id high byte, low byte, location id 1 main / 7 extra, count) in (location,
  // card id) order.
  void _set_obs_opponent_recipe(TArray<uint8_t> &out, PlayerId viewer) {
    std::map<std::pair<uint8_t, CardId>, int> counts;
    for (CardCode code : viewer == 0 ? main_deck1_ : main_deck0_) ++counts[{1, c_get_card_id(code)}];
    for (CardCode code : viewer == 0 ? extra_deck1_ : extra_deck0_) ++counts[{7, c_get_card_id(code)}];
    if (static_cast<int>(counts.size()) > out.Shape()[0])
      throw std::runtime_error(fmt::format("{} opponent recipe rows, {} in the table", counts.size(), out.Shape()[0]));
    int row = 0;
    for (const auto &[key, count] : counts) {
      out(row, 0) = static_cast<uint8_t>(key.second >> 8);
      out(row, 1) = static_cast<uint8_t>(key.second & 0xff);
      out(row, 2) = key.first;
      out(row, 3) = static_cast<uint8_t>(std::min(count, 255));
      ++row;
    }
  }

  // The base archetypes of a card as IsSetCard reads them: the low 12 bits of each 16-bit setcode with nonzero low
  // bits; an artwork variant reads its base card's (deck features' series law, common/card_exact.py's artwork rule).
  static std::vector<uint16_t> archetype_bases(CardCode code) {
    const Card *card = &c_get_card(code);
    if (history::IsArtworkVariant(code, card->alias_)) card = &c_get_card(card->alias_);
    std::vector<uint16_t> out;
    for (int slot = 0; slot < 4; ++slot) {
      const uint16_t part = static_cast<uint16_t>((card->setcode_ >> (16 * slot)) & 0xFFFF);
      if (part & 0xFFF) out.push_back(part & 0xFFF);
    }
    return out;
  }

  // A recipe as obs:own_recipe_ rows in (location, card id) order: count, and the archetype share -- the share of the
  // recipe's main-deck copies sharing a base archetype with the card, as round(255 * share) (runtime_model.py
  // deck_features: same_series_share).
  static std::vector<RecipeRow> recipe_rows(const std::vector<uint32> &main, const std::vector<uint32> &extra) {
    std::map<std::pair<uint8_t, CardId>, RecipeRow> rows;
    for (const auto &[location, cards] : {std::pair<uint8_t, const std::vector<uint32> *>{1, &main}, {7, &extra}})
      for (CardCode code : *cards) {
        RecipeRow &row = rows[{location, c_get_card_id(code)}];
        row.code = code;
        row.id = c_get_card_id(code);
        row.location = location;
        row.count = static_cast<uint8_t>(std::min(row.count + 1, 255));
      }
    if (static_cast<int>(rows.size()) > kRecipeRows)
      throw std::runtime_error(fmt::format("{} own recipe rows, {} in the table", rows.size(), kRecipeRows));
    std::vector<std::pair<std::vector<uint16_t>, int>> main_bases;  // per distinct main card: its bases, its copies
    int main_copies = 0;
    for (const auto &[key, row] : rows)
      if (row.location == 1) {
        main_bases.emplace_back(archetype_bases(row.code), row.count);
        main_copies += row.count;
      }
    std::vector<RecipeRow> out;
    for (auto &[key, row] : rows) {
      const auto bases = archetype_bases(row.code);
      int shared = 0;
      for (const auto &[other, copies] : main_bases) {
        bool same = false;
        for (uint16_t a : bases)
          for (uint16_t b : other) same = same || a == b;
        if (same) shared += copies;
      }
      row.share = static_cast<uint8_t>(main_copies ? std::lround(255.0 * shared / main_copies) : 0);
      out.push_back(row);
    }
    return out;
  }

  // The codes of ``player``'s cards in one location as the engine holds them (face-down ones only when asked).
  std::map<CardCode, int> code_counts(PlayerId player, uint8_t location, bool facedown_only) {
    const int32_t length = OCG_QueryFieldCard(pduel_, player, location, QUERY_CODE | QUERY_POSITION, query_buf_);
    std::map<CardCode, int> out;
    int32_t at = 0;
    while (at < length) {
      uint32_t size = 0, code = 0;
      std::memcpy(&size, query_buf_ + at, 4);
      if (size == LEN_EMPTY) {
        at += 4;
        continue;
      }
      if (size < 16 || at + static_cast<int32_t>(size) > length)
        throw std::runtime_error(fmt::format("malformed query record of {} bytes at {} of {}", size, at, length));
      std::memcpy(&code, query_buf_ + at + 8, 4);
      if (!facedown_only || (query_buf_[at + 15] & POS_FACEDOWN)) ++out[code];
      at += static_cast<int32_t>(size);
    }
    return out;
  }

  // What an owner receives of its own cards (the duel server, gframe/single_duel.cpp MSG_MOVE: the full message,
  // code included, goes to the card's controller after the move; the other player gets code 0 for a deck, hand or
  // face-down destination; graveyard and banishment always put a card in its owner's zones, ocgcore operations.cpp
  // send_to). So every own card outside the own deck is seen by its owner unless it sits hidden in the opponent's
  // control, which only a move between controllers or a swap can bring about.
  void note_control_changes(int msg, const uint8_t *body, size_t size) {
    if (msg == MSG_MOVE && size >= 12) {
      const uint8_t from_controller = body[4], from_location = body[5], to_controller = body[8];
      if (from_location != 0 && from_controller != to_controller && to_controller < 2)
        control_changed_[1 - to_controller] = true;  // the possible owner that is not the new controller
    } else if (msg == MSG_SWAP) {
      control_changed_ = {true, true};
    }
  }

  // The viewer's own cards it cannot see (by code, main-deck and extra-deck kinds apart): those in a hand, deck, extra
  // deck or face-down field place of the opponent whose identity its stream does not show (visible_code). Empty unless
  // a card changed controller this game.
  std::array<std::map<CardCode, int>, 2> unseen_own_cards(PlayerId viewer) {
    std::array<std::map<CardCode, int>, 2> out;
    if (!control_changed_[viewer]) return out;
    const PlayerId opponent = 1 - viewer;
    for (uint8_t location : {LOCATION_HAND, LOCATION_DECK, LOCATION_EXTRA, LOCATION_MZONE, LOCATION_SZONE}) {
      const int32_t length =
          OCG_QueryFieldCard(pduel_, opponent, location, QUERY_CODE | QUERY_POSITION | QUERY_OWNER, query_buf_);
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
          throw std::runtime_error(fmt::format("malformed owner query record of {} bytes at {} of {}", size, at, length));
        std::memcpy(&code, query_buf_ + at + 8, 4);
        std::memcpy(&owner, query_buf_ + at + 16, 4);
        Card c = (client_mode_ && code == 0) ? Card() : c_get_card(code);
        c.controler_ = query_buf_[at + 12];
        c.location_ = query_buf_[at + 13];
        c.sequence_ = query_buf_[at + 14];
        c.position_ = query_buf_[at + 15];
        at += 20;
        if (owner != viewer) continue;
        const bool hidden_place = c.location_ & (LOCATION_HAND | LOCATION_DECK) || (c.position_ & POS_FACEDOWN);
        if (!hidden_place || visible_code(viewer, c) != 0) continue;
        ++out[(c.type_ & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK)) ? 1 : 0][code];
      }
    }
    return out;
  }

  // How many copies of each code the viewer can be sure its own deck (kind 0) or face-down extra deck (kind 1) holds:
  // the engine's count when it sees all its cards; else what its stream determines -- it holds the multiset of
  // held + unseen and knows how many are unseen, so a code's copies are at least that count less the unseen ones.
  std::array<std::map<CardCode, int>, 2> own_known_counts(PlayerId viewer) {
    std::array<std::map<CardCode, int>, 2> held = {code_counts(viewer, LOCATION_DECK, false),
                                                  code_counts(viewer, LOCATION_EXTRA, true)};
    const auto unseen = unseen_own_cards(viewer);
    for (int kind = 0; kind < 2; ++kind) {
      int count = 0;
      for (const auto &[code, n] : unseen[kind]) count += n;
      if (count == 0) continue;
      std::map<CardCode, int> bound;
      for (const auto &[code, n] : held[kind]) {
        const auto u = unseen[kind].find(code);
        const int known = n + (u == unseen[kind].end() ? 0 : u->second) - count;
        if (known > 0) bound[code] = known;
      }
      held[kind] = std::move(bound);
    }
    return held;
  }

  // obs:own_recipe_: the viewer's own recipe (client-known), one row per distinct (location, card id): card id high
  // byte, low byte, location id 1 main / 7 extra, count, remaining, archetype share. Remaining: a main card's copies in
  // the viewer's deck now, an extra card's copies face-down in its extra deck now, as its stream determines them
  // (own_known_counts: the engine's counts unless an own card sits unseen in the opponent's control; face-up extra
  // deck cards are public and not remaining; cards_ lists the same own deck by identity).
  void _set_obs_own_recipe(TArray<uint8_t> &out, PlayerId viewer) {
    const auto known = own_known_counts(viewer);
    const auto &deck = known[0];
    const auto &extra = known[1];
    int row = 0;
    for (const RecipeRow &r : own_recipe_rows_.at(viewer)) {
      const auto &held = r.location == 1 ? deck : extra;
      const auto it = held.find(r.code);
      out(row, 0) = static_cast<uint8_t>(r.id >> 8);
      out(row, 1) = static_cast<uint8_t>(r.id & 0xff);
      out(row, 2) = r.location;
      out(row, 3) = r.count;
      out(row, 4) = static_cast<uint8_t>(it == held.end() ? 0 : std::min(it->second, 255));
      out(row, 5) = r.share;
      ++row;
    }
  }

  // The cards of a seen set that can be in a recipe (tokens are not), each checked against the card table.
  static std::vector<uint32_t> recipe_cards(const std::set<uint32_t> &codes) {
    std::vector<uint32_t> out;
    for (uint32_t code : codes) {
      const auto it = cards_data_.find(code);
      if (it == cards_data_.end()) throw std::runtime_error(fmt::format("seen card {} is not in the card table", code));
      if (!(it->second.type & TYPE_TOKEN)) out.push_back(code);
    }
    return out;
  }

  std::vector<CardCode> announce_candidates(PlayerId player, const std::vector<uint32_t> &opcodes) {
    const auto &law = announce::Registered();
    if (law.cap > max_options())
      throw std::runtime_error(fmt::format("the announce cap {} exceeds max_options {}", law.cap, max_options()));
    const auto &seen = history_.Seen(player);
    auto cards = [](const std::set<uint32_t> &codes) {
      std::vector<uint32_t> out;
      for (uint32_t code : codes) {
        const auto it = cards_data_.find(code);
        if (it == cards_data_.end()) throw std::runtime_error(fmt::format("seen card {} is not in the card table", code));
        if (!(it->second.type & TYPE_TOKEN)) out.push_back(code);
      }
      return out;
    };
    std::vector<uint32_t> card_table;
    card_table.reserve(card_ids_.size());
    for (const auto &[code, id] : card_ids_) card_table.push_back(code);
    auto card_id = [](uint32_t code) -> int64_t {
      const auto it = card_ids_.find(code);
      return it == card_ids_.end() ? 0 : it->second;
    };
    auto declarable = [&opcodes](uint32_t code) {
      const auto it = cards_data_.find(code);
      return it != cards_data_.end() && evaluate_announce_card_filter(it->second, opcodes);
    };
    const auto result = announce::Candidates(
        opcodes, cards(seen[1]), cards(seen[0]), player == 0 ? main_deck0_ : main_deck1_,
        player == 0 ? extra_deck0_ : extra_deck1_, law.tables, law.room_format, law.cap, card_table, card_id,
        declarable);
    if (result.candidates.empty()) throw std::runtime_error("No declarable cards match announce-card filter");
    announce_truncated_ = {result.truncated_belief, result.truncated_empty_union};
    announce_empty_union_ = result.empty_union;
    announce_fixed_ = result.tiers[0] + result.tiers[1] + result.tiers[2] + result.tiers[3];
    return result.candidates;
  }

  // The acting player's own-view key (guards.h): the prompt message and this observation with its full menu, without
  // the history arrays (turn_events_, turn_event_refs_, public_events_, public_event_refs_, h_actions_) and with the
  // activation tallies reduced to their support: the activation table to its sorted (card, description) keys, the
  // turn ledger's per-turn counts and the card rows' activation and attack counts to whether they are nonzero.
  std::string own_view_key(State &state) {
    guards::KeyBuilder key(msg_);
    auto raw = [&](const char *name, const Array &array) {
      key.Add(name, static_cast<const uint8_t *>(array.Data()), array.size * array.element_size);
    };
    raw("cards", state["obs:cards_"_]);
    raw("global", state["obs:global_"_]);
    raw("selection", state["obs:selection_"_]);
    raw("chain", state["obs:chain_"_]);
    raw("player_hints", state["obs:player_hints_"_]);
    raw("unpositioned", state["obs:unpositioned_"_]);
    raw("actions", state["obs:actions_"_]);
    raw("action_ir", state["obs:action_ir_"_]);
    raw("action_single_refs", state["obs:action_single_refs_"_]);
    raw("action_group_refs", state["obs:action_group_refs_"_]);
    raw("action_group_mask", state["obs:action_group_mask_"_]);
    {
      const TArray<uint8_t> ledger = state["obs:turn_ledger_"_];
      const auto *data = static_cast<const uint8_t *>(ledger.Data());
      std::vector<uint8_t> bytes(data, data + ledger.size);
      for (int side = 0; side < 2; ++side)
        for (int column = 0; column < 14; ++column) {
          uint8_t &value = bytes[side * history::kLedgerWidth + column];
          value = value ? 1 : 0;
        }
      key.Add("turn_ledger", bytes);
    }
    {
      const TArray<uint8_t> card_turn = state["obs:card_turn_"_];
      const auto *data = static_cast<const uint8_t *>(card_turn.Data());
      std::vector<uint8_t> bytes(data, data + card_turn.size);
      for (size_t row = 0; row * history::kCardTurnWidth < bytes.size(); ++row)
        for (int column = 0; column < 2; ++column) {
          uint8_t &value = bytes[row * history::kCardTurnWidth + column];
          value = value ? 1 : 0;
        }
      key.Add("card_turn", bytes);
    }
    {
      const TArray<uint8_t> activations = state["obs:turn_activations_"_];
      const auto *data = static_cast<const uint8_t *>(activations.Data());
      std::set<std::array<uint8_t, 3>> keys;
      for (int row = 0; row < history::kActivationRows; ++row) {
        const uint8_t *r = data + row * history::kActivationWidth;
        if (r[0]) keys.insert({r[1], r[2], r[3]});
      }
      std::vector<uint8_t> bytes;
      for (const auto &k : keys) bytes.insert(bytes.end(), k.begin(), k.end());
      key.Add("activation_keys", bytes);
    }
    return key.Finish();
  }

  // Applies the guards (guards.h) to the current prompt: rows they withhold leave the menu the player is shown (the
  // menu arrays and info:num_options are rewritten); step maps the shown row back to the full menu.
  void apply_guards(State &state, const ankerl::unordered_dense::map<std::string, SpecInfo> &spec_infos) {
    visible_.clear();
    guard_prompt_.reset();
    guard_info_ = {0, 0, 0};
    if (legal_actions_.size() < 2) return;
    // rows an illegal-activation withdrawal withheld at this prompt (step), then the no-progress guards' rows,
    // which never take the last row
    std::set<int> withheld;
    for (int row : illegal_held_)
      if (row >= 0 && static_cast<size_t>(row) < legal_actions_.size()) withheld.insert(row);
    if (withheld.size() >= legal_actions_.size())
      throw std::runtime_error("illegal-activation guard: every row of the prompt is withheld");
    guards::Prompt prompt;
    prompt.player = to_play_;
    prompt.msg = msg_;
    prompt.turn = turn_count_;
    prompt.phase = current_phase_;
    prompt.chain = history_.Chain();
    prompt.facts = history_.FactCount(to_play_);
    if (guards::Guarded(prompt)) {
      prompt.key = own_view_key(state);
      for (const auto &action : legal_actions_) prompt.activate.push_back(action.act_ == ActionAct::Activate);
      const guards::Exclusion exclusion = guards_.Exclusions(prompt);
      std::set<int> both = withheld;
      for (int row : exclusion.rows()) both.insert(row);
      const bool kept = exclusion.kept_menu || (both.size() >= legal_actions_.size() && !exclusion.rows().empty());
      guard_info_ = {kept ? 0 : static_cast<int>(exclusion.no_progress.size()),
                     kept ? 0 : static_cast<int>(exclusion.cycle.size()), kept ? 1 : 0};
      if (!kept) withheld = std::move(both);
      guard_prompt_ = std::move(prompt);
      for (int i = 0; i < 3; ++i) state["info:guard"_][i] = guard_info_[i];
    }
    if (withheld.empty()) return;
    for (int row = 0; row < static_cast<int>(legal_actions_.size()); ++row)
      if (!withheld.count(row)) visible_.push_back(row);
    std::vector<LegalAction> shown;
    for (int row : visible_) shown.push_back(legal_actions_[row]);
    std::swap(legal_actions_, shown);
    state["obs:actions_"_].Zero();
    state["obs:action_ir_"_].Zero();
    state["obs:action_single_refs_"_].Zero();
    state["obs:action_group_refs_"_].Zero();
    state["obs:action_group_mask_"_].Zero();
    state["obs:selection_"_].Zero();
    _set_obs_actions(state["obs:actions_"_], legal_actions_);
    _set_obs_action_discard(state["obs:action_discard_"_], legal_actions_);
    _set_obs_action_ir(state["obs:action_ir_"_], state["obs:action_single_refs_"_], state["obs:action_group_refs_"_],
                       state["obs:action_group_mask_"_], spec_infos);
    _set_obs_selection(state["obs:selection_"_]);
    std::swap(legal_actions_, shown);
    state["info:num_options"_] = static_cast<int>(visible_.size());
  }

  // The End Phase discards the turn player's hand down to its hand limit (core query_hand_limit: the last
  // EFFECT_HAND_LIMIT affecting the player, else 6). obs:hand_limit_ rows (own, opponent): valid, the limit, the hand
  // size and the excess max(0, hand - limit), clipped at 255; all public (hand sizes are public, so is a card that
  // changes the limit).
  void _set_obs_hand_limit(TArray<uint8_t> out) {
    out.Zero();
    for (int i = 0; i < 2; ++i) {
      const PlayerId player = (to_play_ + i) % 2;
      const auto [limit, hand] = hand_limit(player);
      out(i, 0) = 1;
      out(i, 1) = static_cast<uint8_t>(std::min(limit, 255));
      out(i, 2) = static_cast<uint8_t>(std::min(hand, 255));
      out(i, 3) = static_cast<uint8_t>(std::min(std::max(hand - limit, 0), 255));
    }
  }

  // (limit, hand size) of a player.
  std::pair<int, int> hand_limit(PlayerId player) {
    // client mode: the default limit (a client builder for a pool with a limit-changing card must derive it from
    // public messages first; no card of A0's pool changes it)
    const int32_t limit = client_mode_ ? 6 : query_hand_limit(pduel_, player);
    if (limit < 0) throw std::runtime_error("query_hand_limit returned " + std::to_string(limit));
    return {limit, static_cast<int>(OCG_QueryFieldCount(pduel_, player, LOCATION_HAND))};
  }

  // obs:action_discard_ per shown menu row: the cards the End Phase would discard from the player's hand if this row
  // ends the turn now (the command menus' move to the End Phase), clipped at 15; 0 on every other row.
  void _set_obs_action_discard(TArray<uint8_t> out, const std::vector<LegalAction> &actions) {
    out.Zero();
    if (msg_ != MSG_SELECT_IDLECMD && msg_ != MSG_SELECT_BATTLECMD) return;
    int excess = -1;
    for (size_t i = 0; i < actions.size(); ++i) {
      if (actions[i].phase_ != ActionPhase::End) continue;
      if (excess < 0) {
        const auto [limit, hand] = hand_limit(to_play_);
        excess = std::max(hand - limit, 0);
      }
      out(i) = static_cast<uint8_t>(std::min(excess, 15));
    }
  }

  void _set_obs_history(State &state) {
    TArray<uint8_t> events = state["obs:turn_events_"_], refs = state["obs:turn_event_refs_"_],
                    chain = state["obs:chain_"_], activations = state["obs:turn_activations_"_],
                    ledger = state["obs:turn_ledger_"_], hints = state["obs:player_hints_"_],
                    card_turn = state["obs:card_turn_"_], chunks = state["obs:turn_chunks_"_],
                    chunk_refs = state["obs:turn_chunk_refs_"_], closed = state["obs:closed_turns_"_],
                    closed_refs = state["obs:closed_turn_refs_"_];
    TArray<int> chunk_meta = state["obs:turn_chunk_meta_"_], closed_meta = state["obs:closed_turn_meta_"_];
    for (TArray<uint8_t> *array : {&events, &refs, &chain, &activations, &ledger, &hints, &card_turn, &chunks,
                                   &chunk_refs, &closed, &closed_refs})
      array->Zero();
    chunk_meta.Zero();
    closed_meta.Zero();
    // the decision id: an observation written twice at one decision delivers the same chunks and closed windows
    const auto written = history_.Write(
        to_play_, step_count_, obs_view_, static_cast<uint8_t *>(events.Data()), static_cast<uint8_t *>(refs.Data()),
        static_cast<uint8_t *>(chain.Data()), static_cast<uint8_t *>(activations.Data()),
        static_cast<uint8_t *>(ledger.Data()), static_cast<uint8_t *>(hints.Data()),
        static_cast<uint8_t *>(card_turn.Data()), card_turn.Shape()[0], static_cast<uint8_t *>(chunks.Data()),
        static_cast<uint8_t *>(chunk_refs.Data()), static_cast<int32_t *>(chunk_meta.Data()),
        static_cast<uint8_t *>(closed.Data()), static_cast<uint8_t *>(closed_refs.Data()),
        static_cast<int32_t *>(closed_meta.Data()));
    state["info:turn_events_dropped"_] = static_cast<int>(written[0]);
    state["info:turn_chunk_backlog"_] = static_cast<int>(written[1]);
    // completed turns still queued for this observer after this decision's deliveries (two closed windows per
    // decision): a decision over a long run of the observer's forced-only turns sees an older memory until it drains
    state["info:closed_turn_backlog"_] = static_cast<int>(written[2]);
    {
      // measurement: the uncapped condensed rows of the current turn and of the closed windows (turn, rows)
      const auto current = history_.CurrentTurnRows(to_play_);
      state["info:turn_rows"_][0] = static_cast<int>(current[0]);
      state["info:turn_rows"_][1] = static_cast<int>(current[1]);
      const auto closed_rows = history_.ClosedTurnRows(to_play_);
      for (int k = 0; k < history::kClosedTurns; ++k)
        for (int j = 0; j < 2; ++j) state["info:closed_turn_rows"_](k, j) = static_cast<int>(closed_rows[k][j]);
    }
    TArray<uint8_t> unpositioned = state["obs:unpositioned_"_];
    unpositioned.Zero();
    history_.WriteUnpositioned(to_play_, obs_view_, static_cast<uint8_t *>(unpositioned.Data()));
    {
      // public_effects/v1 (public_status.h): after Write, whose tracker export may resync a graveyard
      TArray<uint8_t> status = state["obs:card_status_"_], effects = state["obs:public_effects_"_];
      status.Zero();
      effects.Zero();
      state["info:public_effects_dropped"_] = static_cast<int>(
          history_.WriteStatus(to_play_, obs_view_, static_cast<uint8_t *>(status.Data()), status.Shape()[0],
                               static_cast<uint8_t *>(effects.Data())));
    }
    const bool announce = msg_ == MSG_ANNOUNCE_CARD;
    state["info:announce_truncated"_][0] = announce ? announce_truncated_[0] : 0;
    state["info:announce_truncated"_][1] = announce ? announce_truncated_[1] : 0;
    state["info:announce_empty_union"_] = announce && announce_empty_union_ ? 1 : 0;
    state["info:announce_fixed"_] = announce ? announce_fixed_ : 0;
  }

  // export_both_seats (privileged: critic inputs): the non-acting seat's observation as if it decided now -- its card
  // rows, globals, hand limit, recipes, identities without positions, history window, chunks, closed turns and tables,
  // public statuses -- under priv:. The same writers run for that seat on a saved and restored env, so its history
  // delivery bookkeeping is untouched (its chunks and closed windows are the ones its next decision receives); the
  // obs: arrays they used are zeroed for the acting seat's own write.
  void write_other_seat(State &state) {
    if (client_mode_) throw std::runtime_error("export_both_seats: a client has no other seat's view");
    const EnvState saved = save_env();
    to_play_ = 1 - to_play_;
    {
      auto [spec_infos, loc_n_cards] = _set_obs_cards(state["obs:cards_"_], to_play_);
      _set_obs_global(state["obs:global_"_], to_play_, loc_n_cards);
    }
    _set_obs_hand_limit(state["obs:hand_limit_"_]);
    _set_obs_history(state);
    {
      TArray<uint8_t> recipe = state["obs:opponent_recipe_"_];
      recipe.Zero();
      if (spec_.config["public_opponent_recipe"_]) _set_obs_opponent_recipe(recipe, to_play_);
      TArray<uint8_t> own = state["obs:own_recipe_"_];
      own.Zero();
      _set_obs_own_recipe(own, to_play_);
    }
    auto move = [](const Array &from, const Array &to) {
      if (from.size != to.size || from.element_size != to.element_size)
        throw std::runtime_error("export_both_seats: a priv: key differs from its obs: key");
      std::memcpy(to.Data(), from.Data(), from.size * from.element_size);
      from.Zero();
    };
    move(state["obs:cards_"_], state["priv:cards_"_]);
    move(state["obs:global_"_], state["priv:global_"_]);
    move(state["obs:hand_limit_"_], state["priv:hand_limit_"_]);
    move(state["obs:own_recipe_"_], state["priv:own_recipe_"_]);
    move(state["obs:opponent_recipe_"_], state["priv:opponent_recipe_"_]);
    move(state["obs:unpositioned_"_], state["priv:unpositioned_"_]);
    move(state["obs:turn_events_"_], state["priv:turn_events_"_]);
    move(state["obs:turn_event_refs_"_], state["priv:turn_event_refs_"_]);
    move(state["obs:chain_"_], state["priv:chain_"_]);
    move(state["obs:turn_activations_"_], state["priv:turn_activations_"_]);
    move(state["obs:turn_ledger_"_], state["priv:turn_ledger_"_]);
    move(state["obs:player_hints_"_], state["priv:player_hints_"_]);
    move(state["obs:card_turn_"_], state["priv:card_turn_"_]);
    move(state["obs:turn_chunks_"_], state["priv:turn_chunks_"_]);
    move(state["obs:turn_chunk_refs_"_], state["priv:turn_chunk_refs_"_]);
    move(state["obs:turn_chunk_meta_"_], state["priv:turn_chunk_meta_"_]);
    move(state["obs:closed_turns_"_], state["priv:closed_turns_"_]);
    move(state["obs:closed_turn_refs_"_], state["priv:closed_turn_refs_"_]);
    move(state["obs:closed_turn_meta_"_], state["priv:closed_turn_meta_"_]);
    move(state["obs:card_status_"_], state["priv:card_status_"_]);
    move(state["obs:public_effects_"_], state["priv:public_effects_"_]);
    load_env(saved);
  }

  // The code ``viewer`` may see for card ``c``, 0 when hidden (design item 10): its own cards; xyz materials; the
  // opponent's face-up cards outside the deck (public hand cards and face-up extra deck cards included); cards a
  // confirm showed this viewer; and hand cards whose identity this viewer's public tracker still places (history_obs.h),
  // as the tracker knows them. Never the engine's identity of a card the viewer has no public evidence for.
  CardCode visible_code(PlayerId viewer, const Card &c) const {
    if (c.controler_ == viewer || (c.location_ & LOCATION_OVERLAY)) return c.code_;
    // a shown place shows the card the message showed there (the engine's card in env mode, which the audit holds
    // equal; a client's card view does not hold it)
    if (revealed_to(viewer, c.controler_, c.location_, c.sequence_))
      return revealed_.at({c.controler_, c.location_, c.sequence_}).code;
    if (c.location_ != LOCATION_DECK && !(c.position_ & POS_FACEDOWN)) return c.code_;
    // a hand card the tracker places, and a face-down field or banished card whose identity was public before
    // (turned face-down, banished face-down from the graveyard): the tracker follows it
    if (c.location_ == LOCATION_HAND || c.location_ == LOCATION_MZONE || c.location_ == LOCATION_SZONE ||
        c.location_ == LOCATION_REMOVED)
      return static_cast<CardCode>(history_.KnownCode(viewer, 1, c.location_, c.sequence_));
    return 0;
  }

  // Audit only (privileged: reads the engine's cards; never part of an observation, used by scripted_driver.h tests):
  // every opponent card ``viewer`` is shown, checked against the engine's identity, by why it is shown.
  struct VisibleAudit {
    int64_t shown = 0, hand_public = 0, hand_known = 0, extra_faceup = 0, revealed = 0, unpositioned = 0;
    std::vector<std::string> mismatches;
  };
  VisibleAudit audit_visible(PlayerId viewer) {
    VisibleAudit audit;
    const PlayerId opponent = 1 - viewer;
    for (uint8_t location : {LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE,
                             LOCATION_REMOVED, LOCATION_EXTRA}) {
      for (const Card &c : get_cards_in_location(opponent, location)) {
        const CardCode code = visible_code(viewer, c);
        if (code == 0) continue;
        ++audit.shown;
        if (code != c.code_)
          audit.mismatches.push_back(fmt::format("{} at {}:{}:{} shown as {}", c.code_, opponent, c.location_,
                                                 c.sequence_, code));
        if (revealed_to(viewer, c.controler_, c.location_, c.sequence_)) {
          ++audit.revealed;
          const CardCode confirmed = revealed_.at({c.controler_, c.location_, c.sequence_}).code;
          if (confirmed != c.code_)
            audit.mismatches.push_back(fmt::format("{}:{}:{} shows {}, revealed there as {} (a stale reveal)", opponent,
                                                   c.location_, c.sequence_, c.code_, confirmed));
        } else if (location == LOCATION_HAND) ++((c.position_ & POS_FACEDOWN) ? audit.hand_known : audit.hand_public);
        else if (location == LOCATION_EXTRA) ++audit.extra_faceup;
      }
    }
    // the identities known without positions are among the opponent's unshown cards of their location
    std::map<std::pair<int, int64_t>, int64_t> unshown;
    for (uint8_t location : {LOCATION_HAND, LOCATION_DECK, LOCATION_EXTRA})
      for (const Card &c : get_cards_in_location(opponent, location))
        if (visible_code(viewer, c) == 0) ++unshown[{location, c.code_}];
    for (const auto &[key, copies] : history_.Unpositioned(viewer)) {
      audit.unpositioned += copies;
      if (unshown[key] < copies)
        audit.mismatches.push_back(fmt::format("{} copies of {} known at location {} of the opponent, {} there", copies,
                                               key.second, key.first, unshown[key]));
    }
    return audit;
  }

  // Audit only (privileged: the engine's equips): (equip card controller, location, sequence, target info location)
  // for every equipped card on the field.
  std::vector<std::array<uint32_t, 4>> engine_equips() {
    std::vector<std::array<uint32_t, 4>> out;
    for (PlayerId player = 0; player < 2; ++player)
      for (uint8_t location : {LOCATION_MZONE, LOCATION_SZONE})
        for (const Card &c : get_cards_in_location(player, location))
          if (!(c.location_ & LOCATION_OVERLAY) && c.equip_target_)
            out.push_back({c.controler_, c.location_, c.sequence_, c.equip_target_});
    return out;
  }

  bool revealed_to(PlayerId viewer, uint8_t controller, uint8_t location, uint8_t sequence) const {
    const auto it = revealed_.find({controller, location, sequence});
    return it != revealed_.end() && ((it->second.viewers >> viewer) & 1);
  }

  // A reveal (revealed_, keyed by place) holds while its card stays where it was shown: a message that moves, draws,
  // swaps or shuffles cards ends the reveals it may have displaced -- a list zone's (its sequences shift, a shuffle
  // reorders it) wholly, a field slot by itself. Without this a shown place would show whatever card took it (a hand
  // confirmed, then shuffled). What a client keeps across those moves is its public tracker's (hand identities follow
  // their entity; a shuffle moves them to obs:unpositioned_).
  void end_reveals(int msg, const uint8_t *body, size_t size) {
    if (revealed_.empty()) return;
    auto need = [&](size_t n) {
      if (size < n) throw std::runtime_error(fmt::format("truncated message {} ({} bytes) ending reveals", msg, size));
    };
    auto zone = [&](int controller, int location) {
      for (auto it = revealed_.begin(); it != revealed_.end();)
        it = (std::get<0>(it->first) == controller && std::get<1>(it->first) == location) ? revealed_.erase(it)
                                                                                           : std::next(it);
    };
    auto location_everywhere = [&](int location) {
      zone(0, location);
      zone(1, location);
    };
    auto place = [&](const uint8_t *info, bool arriving) {  // controller, location, sequence, position
      const int controller = info[0], location = info[1], sequence = info[2];
      if (location == 0 || (location & LOCATION_OVERLAY) || controller > 1) return;
      if (location == LOCATION_MZONE || location == LOCATION_SZONE) {
        revealed_.erase(
            {static_cast<uint8_t>(controller), static_cast<uint8_t>(location), static_cast<uint8_t>(sequence)});
      } else if (!arriving || location == LOCATION_DECK || location == LOCATION_EXTRA) {
        zone(controller, location);  // hand, graveyard and banished cards arrive at the end: nothing shifts
      }
    };
    switch (msg) {
      case MSG_MOVE:
        need(16);
        place(body + 4, false);
        place(body + 8, true);
        break;
      case MSG_SWAP:
        need(16);
        place(body + 4, false);
        place(body + 12, false);
        break;
      case MSG_DRAW:
        need(1);
        zone(body[0], LOCATION_DECK);
        break;
      case MSG_SHUFFLE_HAND:
        need(1);
        zone(body[0], LOCATION_HAND);
        break;
      case MSG_SHUFFLE_DECK:
        need(1);
        zone(body[0], LOCATION_DECK);
        break;
      case MSG_SHUFFLE_EXTRA:
        need(1);
        zone(body[0], LOCATION_EXTRA);
        break;
      case MSG_SWAP_GRAVE_DECK:
        need(1);
        zone(body[0], LOCATION_DECK);
        zone(body[0], LOCATION_GRAVE);
        break;
      case MSG_SHUFFLE_SET_CARD:
        need(1);
        location_everywhere(body[0]);
        break;
      case MSG_REVERSE_DECK:
        location_everywhere(LOCATION_DECK);
        break;
      default:
        break;
    }
  }

  bool zone_revealed_to(PlayerId viewer, uint8_t controller, uint8_t location) const {
    for (const auto &[place, reveal] : revealed_)
      if (std::get<0>(place) == controller && std::get<1>(place) == location && ((reveal.viewers >> viewer) & 1))
        return true;
    return false;
  }

  // Each player's multi-option decisions in the current turn (step_limit_timeout_loss/v1).
  std::array<int, 2> step_limit_counts() const {
    return decisions_turn_ == turn_count_ ? turn_decisions_ : std::array<int, 2>{0, 0};
  }

  // A registered deck's id; a scripted duel's decks are not registered (-1). Any other name is an error.
  int deck_id(const std::string &name) const {
    const auto it = deck_names_ids_.find(name);
    if (it != deck_names_ids_.end()) return it->second;
    if (scripted_duel_) return -1;
    throw std::runtime_error("deck " + name + " is not registered");
  }

  void WriteState(State &state) {
    float reward = ret_reward_;
    int win_reason = ret_win_reason_;
    int n_options = legal_actions_.size();
    state["reward"_] = reward;
    state["info:to_play"_] = int(to_play_);
    state["info:is_selfplay"_] = int(play_mode_ == kSelfPlay);
    state["info:win_reason"_] = win_reason;
    int anchor_seat = 0;
    if (!anchor_deck_.empty()) {
      if (deck_name_[0] == anchor_deck_) {
        anchor_seat = 1;
      } else if (deck_name_[1] == anchor_deck_) {
        anchor_seat = 2;
      }
    }
    state["info:anchor_seat"_] = anchor_seat;
    for (int i = 0; i < 3; ++i) state["info:guard"_][i] = 0;
    for (int i = 0; i < 6; ++i) state["info:illegal_activation"_][i] = illegal_info_[i];
    {
      const auto counts = step_limit_counts();
      state["info:step_limit"_][0] = step_limited_ ? 1 : 0;
      state["info:step_limit"_][1] = counts[0];
      state["info:step_limit"_][2] = counts[1];
    }
    if (done_) {
      state["info:step_time"_][0] = 0;
      state["info:step_time"_][1] = 0;
      state["info:deck"_][0] = deck_id(deck_name_[0]);
      state["info:deck"_][1] = deck_id(deck_name_[1]);
    }

    if (n_options == 0) {
      state["info:num_options"_] = 1;
      state["obs:global_"_][22] = uint8_t(1);
      state["obs:global_"_][23] = room_format_;
      state["obs:global_"_][24] = room_era_;
      // if (step_count_ >= spec_.config["max_steps"_]) {
      //   fmt::println("Max steps reached return");
      // }
      return;
    }

    if (spec_.config["export_both_seats"_]) write_other_seat(state);

    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;

    if (spec_.config["oppo_info"_]) {
      _set_obs_g_cards(state["obs:cards_"_], to_play_);
      auto [spec_infos_, loc_n_cards_] = _set_obs_mask(state["obs:mask_"_], to_play_);
      spec_infos = spec_infos_;
      loc_n_cards = loc_n_cards_;
    } else {
      auto [spec_infos_, loc_n_cards_] = _set_obs_cards(state["obs:cards_"_], to_play_);
      spec_infos = spec_infos_;
      loc_n_cards = loc_n_cards_;
    }

    _set_obs_global(state["obs:global_"_], to_play_, loc_n_cards);
    _set_obs_hand_limit(state["obs:hand_limit_"_]);

    if (n_options > max_options()) {
      throw std::runtime_error(fmt::format(
          "Legal action overflow: {} > max_options={} (msg={}, deck0={}, "
          "deck1={}, turn={}, phase={})",
          n_options, max_options(), msg_to_string(msg_), deck_name_[0],
          deck_name_[1], turn_count_, phase_to_string(current_phase_)));
    }

    n_options = legal_actions_.size();
    state["info:num_options"_] = n_options;
    bind_placement_actions(spec_infos);

    for (int i = 0; i < n_options; ++i) {
      auto &action = legal_actions_[i];
      action.msg_ = msg_;
      const auto &spec = action.spec_;
      if (!spec.empty()) {
        const auto& spec_info = find_spec_info(spec_infos, spec);
        action.spec_index_ = spec_info.index;
        if (action.cid_ == 0) {
          action.cid_ = spec_info.cid;
        }
      }
    }

    _set_obs_actions(state["obs:actions_"_], legal_actions_);
    _set_obs_action_discard(state["obs:action_discard_"_], legal_actions_);
    _set_obs_action_ir(
        state["obs:action_ir_"_], state["obs:action_single_refs_"_],
        state["obs:action_group_refs_"_],
        state["obs:action_group_mask_"_], spec_infos);
    _set_obs_selection(state["obs:selection_"_]);
    // The 32 decision events carried opponent decisions and stale slot references; they are no longer written
    // (zero until the model drops the keys). Public history is the history_obs.h arrays below.
    state["obs:public_events_"_].Zero();
    state["obs:public_event_refs_"_].Zero();
    _set_obs_history(state);
    _set_obs_candidates(state["obs:candidates_"_]);
    {
      TArray<uint8_t> hidden = state["label:hidden_"_];
      hidden.Zero();
      if (spec_.config["belief_labels"_]) _set_label_hidden(hidden, to_play_);
      TArray<uint8_t> recipe = state["obs:opponent_recipe_"_];
      recipe.Zero();
      if (spec_.config["public_opponent_recipe"_]) _set_obs_opponent_recipe(recipe, to_play_);
      TArray<uint8_t> own = state["obs:own_recipe_"_];
      own.Zero();
      _set_obs_own_recipe(own, to_play_);
    }
    apply_guards(state, spec_infos);

    // write history actions

    auto ha_p = to_play_ == 0 ? ha_p_1_ : ha_p_2_;
    auto &history_actions = to_play_ == 0 ? history_actions_1_ : history_actions_2_;

    int offset = n_history_actions_ - ha_p;
    int n_h_action_feats = history_actions.Shape()[1];

    state["obs:h_actions_"_].Assign(
      (uint8_t *)history_actions[ha_p].Data(), n_h_action_feats * offset);
    state["obs:h_actions_"_][offset].Assign(
      (uint8_t *)history_actions.Data(), n_h_action_feats * ha_p);
    
    for (int i = 0; i < n_history_actions_; ++i) {
      if (uint8_t(state["obs:h_actions_"_](i, 3)) == 0) {
        break;
      }
      // state["obs:h_actions_"_](i, 12) = static_cast<uint8_t>(uint8_t(state["obs:h_actions_"_](i, 12)) == to_play_);
      int turn_diff = std::min(16, turn_count_ - uint8_t(state["obs:h_actions_"_](i, 12)));
      state["obs:h_actions_"_](i, 12) = static_cast<uint8_t>(turn_diff);
    }
  }

private:
  using SpecInfos = ankerl::unordered_dense::map<std::string, SpecInfo>;

  void ensure_card_slot(int offset) const {
    const int capacity = max_cards() * 2;
    if (offset >= capacity) {
      throw std::runtime_error(fmt::format(
          "Card observation overflow: offset={} capacity={} "
          "(deck0={}, deck1={}, turn={}, phase={})",
          offset, capacity, deck_name_[0], deck_name_[1], turn_count_,
          phase_to_string(current_phase_)));
    }
  }

  std::tuple<SpecInfos, std::vector<int>> _set_obs_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;
    int offset = 0;
    obs_view_ = mfenv::TrackerView{};
    // the viewer's own deck and face-down extra deck by identity as far as its stream determines them: all of it
    // unless an own card sits unseen in the opponent's control (own_known_counts); copies beyond that show no identity
    std::optional<std::array<std::map<CardCode, int>, 2>> own_bound;
    {
      const auto unseen = unseen_own_cards(to_play);
      if (!unseen[0].empty() || !unseen[1].empty()) own_bound = own_known_counts(to_play);
    }
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        // the opponent's deck is anonymous unless a confirm showed this player a card of it; its hand and extra deck
        // are always read card by card (public hand cards, face-up extra deck cards and known identities show)
        if (opponent && location == LOCATION_DECK && !zone_revealed_to(to_play, player, location)) {
          auto n_cards = OCG_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          obs_view_.zone_counts[{pi, location}] = n_cards;
          for (auto i = 0; i < n_cards; i++) {
            ensure_card_slot(offset);
            f_cards(offset, 2) = location_to_id(location);
            f_cards(offset, 4) = 1;
            // the row stands for the card at sequence i (a selection may name it); its identity stays hidden
            spec_infos[ls_to_spec(location, i, 0, true)] = {static_cast<uint16_t>(offset + 1), 0};
            mfenv::TrackerToken token;
            token.kind = mfenv::TrackerToken::OTHER;
            token.side = pi;
            token.location = location;
            obs_view_.tokens.push_back(token);
            offset++;
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          if (!opponent && location == LOCATION_DECK) {
            // The deck order is hidden even from its owner: own deck rows in card id order (stable, so copies of
            // one card keep identical rows), never in the engine's order (design item 5a).
            std::stable_sort(cards.begin(), cards.end(), [](const Card &a, const Card &b) {
              return c_get_card_id(a.code_) < c_get_card_id(b.code_);
            });
          }
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          int zone_cards = 0;
          for (const auto &c : cards) zone_cards += (c.location_ & LOCATION_OVERLAY) ? 0 : 1;
          obs_view_.zone_counts[{pi, location}] = zone_cards;
          for (int i = 0; i < n_cards; ++i) {
            ensure_card_slot(offset);
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            CardCode code = visible_code(to_play, c);
            if (own_bound && !opponent &&
                (location == LOCATION_DECK || (location == LOCATION_EXTRA && (c.position_ & POS_FACEDOWN)))) {
              int &left = (*own_bound)[location == LOCATION_DECK ? 0 : 1][c.code_];
              if (left > 0) --left;
              else code = 0;
            }
            const bool hide = code == 0;
            const CardId card_id = hide ? 0 : c_get_card_id(code);
            // a known identity is shown as the tracker knows it (its card data), never from the engine's card
            Card shown = c;
            if (!hide && code != c.code_) {
              shown = c_get_card(code);
              shown.controler_ = c.controler_;
              shown.location_ = c.location_;
              shown.sequence_ = c.sequence_;
              shown.position_ = c.position_;
            }
            {
              // the row as the public tracker sees it: place, and the code only when this row shows it
              mfenv::TrackerToken token;
              token.side = pi;
              token.sequence = c.sequence_;
              token.code = code;
              if (c.location_ & LOCATION_OVERLAY) {
                token.kind = mfenv::TrackerToken::OVERLAY_MATERIAL;
                token.location = c.location_ & 0x7f;
                token.overlay_index = c.position_;
              } else {
                token.kind = mfenv::TrackerToken::CARD;
                token.location = c.location_;
              }
              obs_view_.tokens.push_back(token);
            }
            if (!client_mode_ && location == LOCATION_MZONE && !(c.location_ & LOCATION_OVERLAY)) {
              // refreshed_view/v1: what the server last sent for this slot
              const RefreshedStats &r = refreshed_mzone_.at(player).at(c.sequence_);
              if (!r.valid) throw std::runtime_error("refreshed view: a monster no refresh has reached");
              shown.level_ = r.level;
              shown.attack_ = r.attack;
              shown.defense_ = r.defense;
              shown.status_ = r.status;
            }
            _set_obs_card_(f_cards, offset, shown, hide, card_id);
            offset++;

            spec_infos[spec] = {static_cast<uint16_t>(offset), card_id};
          }
        }
      }
    }
    return {spec_infos, loc_n_cards};
  }

  void _set_obs_g_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      std::vector<uint8_t> configs = {
          LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE,
          LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED,
          LOCATION_EXTRA,
      };
      for (auto location : configs) {
        std::vector<Card> cards = get_cards_in_location(player, location);
        int n_cards = cards.size();
        for (int i = 0; i < n_cards; ++i) {
          ensure_card_slot(offset);
          const auto &c = cards[i];
          CardId card_id = c_get_card_id(c.code_);
          _set_obs_card_(f_cards, offset, c, false, card_id, false);
          offset++;
        }
      }
    }
  }

  std::tuple<SpecInfos, std::vector<int>> _set_obs_mask(TArray<uint8_t> &mask, PlayerId to_play) {
    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        if (opponent && hidden_for_opponent && !zone_revealed_to(to_play, player, location)) {
          auto n_cards = OCG_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          for (auto i = 0; i < n_cards; i++) {
            ensure_card_slot(offset);
            mask(offset, 1) = 1;
            mask(offset, 3) = 1;
            spec_infos[ls_to_spec(location, i, 0, true)] = {static_cast<uint16_t>(offset + 1), 0};
            offset++;
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          for (int i = 0; i < n_cards; ++i) {
            ensure_card_slot(offset);
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            bool hide = false;
            // xyz materials are public (the host sends their codes to both players); a material's position_ holds
            // its index under the xyz monster, not a position
            if (opponent && !(c.location_ & LOCATION_OVERLAY)) {
              hide = hidden_for_opponent || (c.position_ & POS_FACEDOWN);
              if (revealed_to(to_play, c.controler_, c.location_, c.sequence_)) {
                hide = false;
              }
            }
            CardId card_id = 0;
            if (!hide) {
              card_id = c_get_card_id(c.code_);
            }
            _set_obs_mask_(mask, offset, c, hide, card_id);
            offset++;

            spec_infos[spec] = {static_cast<uint16_t>(offset), card_id};
          }
        }
      }
    }
    return {spec_infos, loc_n_cards};
  }

  void _set_obs_card_(TArray<uint8_t> &f_cards, int offset, const Card &c,
                      bool hide, CardId card_id = 0, bool global = false) {
    // check offset exceeds max_cards
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      f_cards(offset, 0) = static_cast<uint8_t>(card_id >> 8);
      f_cards(offset, 1) = static_cast<uint8_t>(card_id & 0xff);
    }
    f_cards(offset, 2) = location_to_id(location);

    uint8_t seq = 0;
    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      seq = c.sequence_ + 1;
    }
    f_cards(offset, 3) = seq;
    f_cards(offset, 4) = global ? c.controler_ : ((c.controler_ != to_play_) ? 1 : 0);
    if (overlay) {
      f_cards(offset, 5) = position_to_id(POS_FACEUP);
      f_cards(offset, 6) = 1;
    } else {
      if (location == LOCATION_DECK || location == LOCATION_HAND || location == LOCATION_EXTRA) {
        if (hide || (c.position_ & POS_FACEDOWN)) {
          f_cards(offset, 5) = position_to_id(POS_FACEDOWN);
        }
        // else {
        //   fmt::println("location: {}, position: {}", location2str.at(location), position_to_string(c.position_));
        // }
      } else {
        f_cards(offset, 5) = position_to_id(c.position_);
      }
    }
    if (!hide) {
      f_cards(offset, 7) = attribute_to_id(c.attribute_);
      f_cards(offset, 8) = race_to_id(c.race_);
      f_cards(offset, 9) = c.level_;
      f_cards(offset, 10) = std::min(c.counter_, static_cast<uint32_t>(15));
      f_cards(offset, 11) = static_cast<uint8_t>((c.status_ & (STATUS_DISABLED | STATUS_FORBIDDEN)) != 0);
      auto [atk1, atk2] = float_transform(c.attack_);
      f_cards(offset, 12) = atk1;
      f_cards(offset, 13) = atk2;

      auto [def1, def2] = float_transform(c.defense_);
      f_cards(offset, 14) = def1;
      f_cards(offset, 15) = def2;

      auto type_ids = type_to_ids(c.type_);
      for (int j = 0; j < type_ids.size(); ++j) {
        f_cards(offset, 16 + j) = type_ids[j];
      }
    }
  }

  void _set_obs_mask_(TArray<uint8_t> &mask, int offset, const Card &c,
                      bool hide, CardId card_id = 0, bool global = false) {
    // check offset exceeds max_cards
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      if (card_id != 0) {
        mask(offset, 0) = 1;
      }
    }
    mask(offset, 1) = 1;

    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      mask(offset, 2) = 1;
    }
    mask(offset, 3) = 1;
    if (overlay) {
      mask(offset, 4) = 1;
      mask(offset, 5) = 1;
    } else {
      if (location == LOCATION_DECK || location == LOCATION_HAND || location == LOCATION_EXTRA) {
        if (hide || (c.position_ & POS_FACEDOWN)) {
          mask(offset, 4) = 1;
        }
      } else {
        mask(offset, 4) = 1;
      }
    }
    if (!hide) {
      mask(offset, 6) = 1;
      mask(offset, 7) = 1;
      mask(offset, 8) = 1;
      mask(offset, 9) = 1;
      mask(offset, 10) = 1;
      mask(offset, 11) = 1;
      mask(offset, 12) = 1;
      mask(offset, 13) = 1;
    }
  }

  void _set_obs_global(TArray<uint8_t> &feat, PlayerId player, const std::vector<int> &loc_n_cards) {
    uint8_t me = player;
    uint8_t op = 1 - player;

    auto [me_lp_1, me_lp_2] = float_transform(lp_[me]);
    feat(0) = me_lp_1;
    feat(1) = me_lp_2;

    auto [op_lp_1, op_lp_2] = float_transform(lp_[op]);
    feat(2) = op_lp_1;
    feat(3) = op_lp_2;

    feat(4) = std::min(turn_count_, 16);
    feat(5) = phase_to_id(current_phase_);
    feat(6) = (me == 0) ? 1 : 0;
    feat(7) = (me == tp_) ? 1 : 0;

    for (int i = 0; i < loc_n_cards.size(); i++) {
      feat(8 + i) = static_cast<uint8_t>(loc_n_cards[i]);
    }
    // the room's public format (registered table index) and its era, fixed for the run (era_format.py; the
    // index and era law are resolved from the registered table by mirrorforce/agent/env/room_format.py)
    feat(23) = room_format_;
    feat(24) = room_era_;
  }

  // Every card of the board has a row and a spec (hidden opponent rows included), so a menu spec without one is an
  // error, never a silent row 0.
  const SpecInfo& find_spec_info(SpecInfos &spec_infos, const std::string &spec) {
    auto it = spec_infos.find(spec);
    if (it == spec_infos.end()) {
      throw std::runtime_error(fmt::format("menu spec {} names no card row (msg {}, turn {})", spec,
                                           msg_to_string(msg_), turn_count_));
    }
    return it->second;
  }

  void _set_obs_action_spec(
    TArray<uint8_t> &feat, int i, int idx) {
    feat(i, 0) = static_cast<uint8_t>(idx);
  }

  void _set_obs_action_card_id(
    TArray<uint8_t> &feat, int i, CardId cid) {
    feat(i, 1) = static_cast<uint8_t>(cid >> 8);
    feat(i, 2) = static_cast<uint8_t>(cid & 0xff);
  }

  void _set_obs_action_msg(TArray<uint8_t> &feat, int i, int msg) {
    feat(i, 3) = msg_to_id(msg);
  }

  void _set_obs_action_act(TArray<uint8_t> &feat, int i, ActionAct act) {
    feat(i, 4) = static_cast<uint8_t>(act);
  }

  void _set_obs_action_finish(TArray<uint8_t> &feat, int i) {
    feat(i, 5) = 1;
  }

  uint8_t action_effect_id(int effect) const {
    if (effect == -1) {
      return 0;
    }
    if (effect == 0) {
      return 1;
    }
    if (effect >= CARD_EFFECT_OFFSET) {
      return static_cast<uint8_t>(
          std::min(255, effect - CARD_EFFECT_OFFSET + 2));
    }
    return system_string_to_id(effect);
  }

  void _set_obs_action_effect(TArray<uint8_t> &feat, int i, int effect) {
    // 0: None
    // 1: default
    // 2-15: card effect
    // 16+: system
    feat(i, 6) = action_effect_id(effect);
  }

  void _set_obs_action_phase(TArray<uint8_t> &feat, int i, ActionPhase phase){
    feat(i, 7) = static_cast<uint8_t>(phase);
  }

  void _set_obs_action_position(TArray<uint8_t> &feat, int i, uint8_t position) {
    feat(i, 8) = position_to_id(position);
  }

  void _set_obs_action_number(TArray<uint8_t> &feat, int i, uint8_t number) {
    feat(i, 9) = number;
  }

  void _set_obs_action_place(TArray<uint8_t> &feat, int i, ActionPlace place) {
    feat(i, 10) = static_cast<uint8_t>(place);
  }

  void _set_obs_action_attrib(TArray<uint8_t> &feat, int i, uint8_t attrib) {
    feat(i, 11) = attribute_to_id(attrib);
  }

  void _set_obs_action(TArray<uint8_t> &feat, int i, const LegalAction &action) {
    auto msg = action.msg_;
    _set_obs_action_msg(feat, i, msg);
    _set_obs_action_card_id(feat, i, action.cid_);
    if (msg == MSG_SELECT_CARD || msg == MSG_SELECT_TRIBUTE ||
        msg == MSG_SELECT_SUM || msg == MSG_SELECT_UNSELECT_CARD) {
      if (action.finish_) {
        _set_obs_action_finish(feat, i);
      } else {
        _set_obs_action_spec(feat, i, action.spec_index_);
      }
    } else if (msg == MSG_SELECT_POSITION) {
      _set_obs_action_position(feat, i, action.position_);
    } else if (msg == MSG_SELECT_EFFECTYN) {
      _set_obs_action_spec(feat, i, action.spec_index_);
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (msg == MSG_SELECT_YESNO || msg == MSG_SELECT_OPTION) {
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (
      msg == MSG_SELECT_BATTLECMD ||
      msg == MSG_SELECT_IDLECMD ||
      msg == MSG_SELECT_CHAIN) {
      _set_obs_action_phase(feat, i, action.phase_);
      _set_obs_action_spec(feat, i, action.spec_index_);
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (msg == MSG_SELECT_PLACE || msg == MSG_SELECT_DISFIELD) {
      _set_obs_action_spec(feat, i, action.spec_index_);
      _set_obs_action_place(feat, i, action.place_);
    } else if (msg == MSG_ANNOUNCE_CARD) {
      // card id, already set
    } else if (msg == MSG_ANNOUNCE_ATTRIB) {
      _set_obs_action_attrib(feat, i, action.attribute_);
    } else if (msg == MSG_ANNOUNCE_NUMBER) {
      _set_obs_action_number(feat, i, action.number_);
    } else if (msg == MSG_ANNOUNCE_RACE) {
      feat(i, 6) = static_cast<uint8_t>(
          ACTION_RACE_EFFECT_OFFSET + action.number_);
    } else {
      throw std::runtime_error("Unsupported message " + msg_to_string(msg));
    }
  }

  CardId spec_to_card_id(const std::string &spec, PlayerId player) {
    const PlayerId viewer = player;
    int offset = 0;
    bool opponent = false;
    if (spec[0] == 'o') {
      player = 1 - player;
      opponent = true;
      offset++;
    }
    auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
    if (std::isalpha(static_cast<unsigned char>(spec.back()))) {
      // an xyz material ("m3a"): public to both players; find it under its monster
      for (const Card &c : get_cards_in_location(player, LOCATION_MZONE))
        if ((c.location_ & LOCATION_OVERLAY) && c.sequence_ == seq && c.position_ == pos)
          return c_get_card_id(c.code_);
      throw std::runtime_error("no xyz material at " + spec);
    }
    if (opponent) {
      const CardCode code = visible_code(viewer, get_card(player, loc, seq));
      return code == 0 ? 0 : c_get_card_id(code);
    }
    return c_get_card_id(get_card_code(player, loc, seq));
  }

  void _set_obs_actions(TArray<uint8_t> &feat, const std::vector<LegalAction> &actions) {
    for (int i = 0; i < actions.size(); ++i) {
      _set_obs_action(feat, i, actions[i]);
    }
  }

  uint8_t lookup_spec_ref(
      const SpecInfos &spec_infos, const std::string &spec) const {
    if (spec.empty()) {
      return 0;
    }
    const auto it = spec_infos.find(spec);
    if (it == spec_infos.end()) {
      return 0;
    }
    if (it->second.index > 255) {
      throw std::runtime_error("Card reference exceeds uint8 range");
    }
    return static_cast<uint8_t>(it->second.index);
  }

  CardId lookup_spec_card_id(
      const SpecInfos &spec_infos, const std::string &spec) const {
    if (spec.empty()) {
      return 0;
    }
    const auto it = spec_infos.find(spec);
    return it == spec_infos.end() ? 0 : it->second.cid;
  }

  void bind_placement_actions(const SpecInfos &spec_infos) {
    if (msg_ != MSG_SELECT_PLACE && msg_ != MSG_SELECT_DISFIELD) return;
    std::string bound;
    if (placement_prompt_cid_) {
      const auto &anchor = placement_source_[to_play_];
      if (anchor.cid == placement_prompt_cid_ && !anchor.spec.empty() &&
          lookup_spec_card_id(spec_infos, anchor.spec) == anchor.cid)
        bound = anchor.spec;
      if (bound.empty()) {
        size_t matches = 0;
        for (const auto &[spec, info] : spec_infos) {
          if (info.cid == placement_prompt_cid_ &&
              !std::isalpha(static_cast<unsigned char>(spec.back()))) {
            bound = spec;
            ++matches;
          }
        }
        if (matches != 1) bound.clear();  // never pick a same-name instance arbitrarily
      }
    }
    for (auto &action : legal_actions_) {
      action.cid_ = placement_prompt_cid_;
      action.spec_ = bound;
      action.spec_index_ = bound.empty() ? 0 : lookup_spec_ref(spec_infos, bound);
    }
  }

  void _set_obs_action_ir(
      TArray<uint8_t> &ir, TArray<uint8_t> &single_refs,
      TArray<uint8_t> &group_refs, TArray<uint8_t> &group_mask,
      const SpecInfos &spec_infos) {
    const bool finishable = std::any_of(
        legal_actions_.begin(), legal_actions_.end(),
        [](const LegalAction &action) { return action.finish_; });
    const bool cancelable = std::any_of(
        legal_actions_.begin(), legal_actions_.end(),
        [](const LegalAction &action) {
          return action.act_ == ActionAct::Cancel;
        });
    const auto role = selection_role_for_msg(msg_);
    const bool active_selection = ms_idx_ >= 0;
    const int stage = active_selection ? ms_idx_ : 0;
    const int selected_count = active_selection
        ? static_cast<int>(ms_r_idxs_.size())
        : 0;
    const int minimum = active_selection ? ms_min_ : 0;
    const int maximum = active_selection ? ms_max_ : 0;
    const int must = active_selection ? ms_must_ : 0;
    const bool mandatory =
        active_selection && selected_count < std::max(minimum, must);

    std::vector<uint8_t> selected_refs;
    selected_refs.reserve(kMaxActionRoleMembers);
    if (active_selection) {
      for (const auto selected_index : ms_r_idxs_) {
        if (selected_index < 0 ||
            selected_index >= static_cast<int>(ms_specs_.size())) {
          continue;
        }
        const auto ref = lookup_spec_ref(
            spec_infos, ms_specs_[selected_index]);
        if (ref != 0) {
          selected_refs.push_back(ref);
        }
        if (selected_refs.size() == kMaxActionRoleMembers) {
          break;
        }
      }
    }

    for (int i = 0; i < static_cast<int>(legal_actions_.size()); ++i) {
      const auto &action = legal_actions_[i];
      const bool candidate_action =
          is_candidate_selection_msg(action.msg_) &&
          !action.finish_ && action.act_ != ActionAct::Cancel;
      const bool placement_action = action.msg_ == MSG_SELECT_PLACE || action.msg_ == MSG_SELECT_DISFIELD;

      uint8_t source_ref = 0;
      uint8_t candidate_ref = 0;
      CardId source_cid = 0;
      int source_effect = action.effect_;
      uint8_t binding_confidence = 0;

      if (placement_action) {
        source_ref = static_cast<uint8_t>(std::min(action.spec_index_, 255));
        source_cid = action.cid_;  // a received identity may be known without a unique instance
        binding_confidence = source_ref == 0 ? 0 : 2;
      } else if (candidate_action) {
        candidate_ref = static_cast<uint8_t>(
            std::min(action.spec_index_, 255));
      } else if (!action.spec_.empty()) {
        source_ref = static_cast<uint8_t>(
            std::min(action.spec_index_, 255));
        source_cid = action.cid_;
        binding_confidence = source_ref == 0 ? 0 : 2;
      }

      if (source_ref == 0 && !pending_source_spec_.at(to_play_).empty() &&
          !is_source_action_msg(action.msg_) && !placement_action) {
        const auto &observer_spec = pending_source_spec_.at(to_play_);
        source_ref = lookup_spec_ref(spec_infos, observer_spec);
        source_cid = lookup_spec_card_id(spec_infos, observer_spec);
        if (source_cid == 0) {
          source_cid = pending_source_cid_.at(to_play_);
        }
        source_effect = pending_source_effect_.at(to_play_);
        binding_confidence = source_ref == 0 ? 0 : 1;
      }

      single_refs(i, 0) = source_ref;
      single_refs(i, 1) = candidate_ref;
      if (action.act_ == ActionAct::Attack ||
          action.act_ == ActionAct::DirectAttack) {
        single_refs(i, 2) = source_ref;
      }

      for (int member = 0;
           member < static_cast<int>(selected_refs.size()); ++member) {
        group_refs(i, 4, member) = selected_refs[member];
        group_mask(i, 4, member) = 1;
        if (role == SelectionRole::Material) {
          group_refs(i, 2, member) = selected_refs[member];
          group_mask(i, 2, member) = 1;
        } else if (role == SelectionRole::Tribute) {
          group_refs(i, 3, member) = selected_refs[member];
          group_mask(i, 3, member) = 1;
        } else if (role == SelectionRole::Target) {
          group_refs(i, 0, member) = selected_refs[member];
          group_mask(i, 0, member) = 1;
        } else if (role == SelectionRole::Cost) {
          group_refs(i, 1, member) = selected_refs[member];
          group_mask(i, 1, member) = 1;
        }
      }

      ir(i, 0) = msg_to_id(action.msg_);
      ir(i, 1) = static_cast<uint8_t>(action.act_);
      ir(i, 2) = static_cast<uint8_t>(action.phase_);
      ir(i, 3) = static_cast<uint8_t>(role);
      ir(i, 4) = static_cast<uint8_t>(std::min(stage, 15));
      ir(i, 5) = static_cast<uint8_t>(action.finish_);
      ir(i, 6) = static_cast<uint8_t>(
          action.act_ == ActionAct::Cancel);
      ir(i, 7) = static_cast<uint8_t>(mandatory);
      ir(i, 8) = static_cast<uint8_t>(finishable);
      ir(i, 9) = static_cast<uint8_t>(cancelable);
      ir(i, 10) = static_cast<uint8_t>(std::min(minimum, 255));
      ir(i, 11) = static_cast<uint8_t>(std::min(maximum, 255));
      ir(i, 12) = static_cast<uint8_t>(
          std::min(selected_count, 255));
      ir(i, 13) = static_cast<uint8_t>(std::min(must, 255));
      ir(i, 14) = action_effect_id(source_effect);
      ir(i, 15) = static_cast<uint8_t>(source_cid >> 8);
      ir(i, 16) = static_cast<uint8_t>(source_cid & 0xff);
      ir(i, 17) = binding_confidence;
      ir(i, 18) = position_to_id(action.position_);
      ir(i, 19) = static_cast<uint8_t>(action.place_);
      ir(i, 20) = action.number_;
      ir(i, 21) = attribute_to_id(action.attribute_);
      ir(i, 22) = static_cast<uint8_t>(std::min(i + 1, 255));
      ir(i, 23) = static_cast<uint8_t>(
          active_selection ? ms_mode_ : 0);
    }
  }

  void _set_obs_selection(TArray<uint8_t> &selection) {
    const bool finishable = std::any_of(
        legal_actions_.begin(), legal_actions_.end(),
        [](const LegalAction &action) { return action.finish_; });
    const bool cancelable = std::any_of(
        legal_actions_.begin(), legal_actions_.end(),
        [](const LegalAction &action) {
          return action.act_ == ActionAct::Cancel;
        });
    const bool active_selection = ms_idx_ >= 0;
    const int selected_count = active_selection
        ? static_cast<int>(ms_r_idxs_.size())
        : 0;
    const int minimum = active_selection ? ms_min_ : 0;
    const int maximum = active_selection ? ms_max_ : 0;
    const int must = active_selection ? ms_must_ : 0;
    selection(0) = msg_to_id(msg_);
    selection(1) = static_cast<uint8_t>(selection_role_for_msg(msg_));
    selection(2) = static_cast<uint8_t>(
        active_selection ? ms_idx_ : 0);
    selection(3) = static_cast<uint8_t>(std::min(minimum, 255));
    selection(4) = static_cast<uint8_t>(std::min(maximum, 255));
    selection(5) = static_cast<uint8_t>(std::min(must, 255));
    selection(6) = static_cast<uint8_t>(
        std::min(selected_count, 255));
    selection(7) = static_cast<uint8_t>(finishable);
    selection(8) = static_cast<uint8_t>(cancelable);
    selection(9) = static_cast<uint8_t>(
        active_selection && selected_count < std::max(minimum, must));
    selection(10) = static_cast<uint8_t>(
        active_selection ? ms_mode_ : 0);
    selection(11) = static_cast<uint8_t>(
        std::min(static_cast<int>(legal_actions_.size()), 255));
  }

  void _set_obs_public_events(
      TArray<uint8_t> &events, TArray<uint8_t> &event_refs,
      const SpecInfos &spec_infos) {
    const int count = std::min(
        static_cast<int>(public_events_.size()), n_history_actions_);
    for (int i = 0; i < count; ++i) {
      const auto &event = public_events_[i];
      const auto source_spec = spec_for_observer(
          event.source_spec, event.source_spec_player, to_play_);
      const auto candidate_spec = spec_for_observer(
          event.candidate_spec, event.candidate_spec_player, to_play_);
      const auto source_ref = lookup_spec_ref(spec_infos, source_spec);
      const auto candidate_ref = lookup_spec_ref(
          spec_infos, candidate_spec);

      events(i, 0) = 1;
      events(i, 1) = event.actor == to_play_ ? 1 : 2;
      events(i, 2) = msg_to_id(event.msg);
      events(i, 3) = static_cast<uint8_t>(event.act);
      events(i, 4) = static_cast<uint8_t>(event.action_phase);
      events(i, 5) = static_cast<uint8_t>(event.selection_role);
      events(i, 6) = static_cast<uint8_t>(
          std::min(16, std::max(0, turn_count_ - event.turn)));
      events(i, 7) = phase_to_id(event.phase);
      events(i, 8) = static_cast<uint8_t>(event.finish);
      events(i, 9) = static_cast<uint8_t>(event.cancel);
      if (event.payload_size != 0) {
        events(i, 10) = event.payload_size;
        for (int payload_index = 0;
             payload_index < event.payload_size; ++payload_index) {
          events(i, 11 + payload_index) =
              event.payload[payload_index];
        }
      } else {
        events(i, 10) = action_effect_id(event.effect);
        events(i, 11) = static_cast<uint8_t>(
            std::min(event.selection_stage, 15));
        events(i, 12) = static_cast<uint8_t>(
            std::min(event.selected_count, 255));
        events(i, 13) = static_cast<uint8_t>(source_ref != 0);
        events(i, 14) = static_cast<uint8_t>(candidate_ref != 0);
        events(i, 15) = event.choice;
      }
      event_refs(i, 0) = source_ref;
      event_refs(i, 1) = candidate_ref;
      if (event.act == ActionAct::Attack ||
          event.act == ActionAct::DirectAttack) {
        event_refs(i, 2) = source_ref;
      }
    }
  }


  void str_to_uint16(const char* src, uint16_t* dest) {
      for (int i = 0; i < strlen(src); i += 1) {
        dest[i] = src[i];
      }

      // Add null terminator
      dest[strlen(src) + 1] = '\0';
  }

  void ReplayWriteInt8(int8_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  void ReplayWriteInt32(int32_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  // duel-core API
public:
  // The current game as a rollout root (S2, rollout_pool.h): the outer seed given to OCG_CreateDuel, both decks
  // exactly as loaded, the menu indices so far, the core stream digest (the clone must reach it), the player to move
  // and the env generator as play began (play_gen_). Only at a decision point of a game in progress.
  std::string root_record() const {
    if (!duel_started_ || done_) throw std::runtime_error("root_record: no game in progress");
    return fmt::format("{{\"seed\":{},\"main_decks\":[[{}],[{}]],\"extra_decks\":[[{}],[{}]],\"actions\":[{}],"
                       "\"stream_hash\":{},\"to_play\":{},\"play_gen\":\"{}\"}}",
                       duel_seed_, fmt::join(main_deck0_, ","), fmt::join(main_deck1_, ","),
                       fmt::join(extra_deck0_, ","), fmt::join(extra_deck1_, ","), fmt::join(repro_actions_, ","),
                       stream_hash_, static_cast<int>(to_play_), play_gen_);
  }

  uint64_t stream_hash() const { return stream_hash_; }

  // One-line JSON that replays the failing game: the outer seed given to
  // OCG_CreateDuel, both decks exactly as loaded, every response so far (hex)
  // and the prompt being handled.
  std::string repro_json(const char *error) const {
    std::string responses;
    for (size_t i = 0; i < repro_responses_.size(); i++) {
      std::string hex;
      for (auto byte_value : repro_responses_[i]) {
        hex += fmt::format("{:02x}", byte_value);
      }
      responses += fmt::format("{}\"{}\"", i ? "," : "", hex);
    }
    std::string buffer;
    for (int i = 0; i < dl_; i++) {
      buffer += fmt::format("{:02x}", data_[i]);
    }
    // illegal-activation withdrawals: [menu indices so far, row withheld, card, prompts withdrawn past]
    std::string withdrawn;
    for (size_t i = 0; i < illegal_log_.size(); ++i)
      withdrawn += fmt::format("{}[{}]", i ? "," : "", fmt::join(illegal_log_[i], ","));
    std::string escaped;
    for (const char *c = error; *c; ++c) {
      if (*c == '"' || *c == '\\') escaped += '\\';
      escaped += (*c == '\n' ? ' ' : *c);
    }
    return fmt::format(
        "{{\"error\":\"{}\",\"seed\":{},\"deck_names\":[\"{}\",\"{}\"],"
        "\"main_decks\":[[{}],[{}]],\"extra_decks\":[[{}],[{}]],\"msg\":{},\"turn\":{},"
        "\"actions\":[{}],\"responses\":[{}],\"buffer\":\"{}\",\"dp\":{},\"withdrawn\":[{}]}}",
        escaped, duel_seed_, deck_name_[0], deck_name_[1], fmt::join(main_deck0_, ","),
        fmt::join(main_deck1_, ","), fmt::join(extra_deck0_, ","), fmt::join(extra_deck1_, ","),
        msg_, turn_count_, fmt::join(repro_actions_, ","), responses, buffer, dp_, withdrawn);
  }

  uint32_t repro_seed() const { return duel_seed_; }

  static std::string generator_text(const std::mt19937 &generator) {
    std::ostringstream out;
    out << generator;
    return out.str();
  }

  // ---- exact resume (design section 11, before MT) ----
  // The export holds the generators at the start of the current game and now, whether a game is in progress, its
  // core stream digest and the menu indices it was stepped with. Import restores the start generators, replays the
  // game -- reset(), then every index through step(), writing the observation after each as the env does, so the
  // guards' shown menus replay too -- and refuses unless the replayed indices, stream digest and both generators
  // equal the export. A finished game (or none) restores only the generators.
  std::string export_state() const {
    std::ostringstream out;
    out << "mirrorforce_env_state/v1\n" << start_gen_ << "\n" << start_duel_gen_ << "\n" << generator_text(gen_)
        << "\n" << generator_text(duel_gen_) << "\n" << (duel_started_ && !done_ ? 1 : 0) << " " << stream_hash_
        << " " << repro_actions_.size();
    for (int action : repro_actions_) out << " " << action;
    out << "\n";
    return out.str();
  }

  void import_state(const std::string &text) {
    std::istringstream in(text);
    std::string schema, start_gen, start_duel_gen, gen, duel_gen, tail;
    if (!std::getline(in, schema) || schema != "mirrorforce_env_state/v1" || !std::getline(in, start_gen) ||
        !std::getline(in, start_duel_gen) || !std::getline(in, gen) || !std::getline(in, duel_gen) ||
        !std::getline(in, tail)) {
      throw std::runtime_error("malformed env state");
    }
    std::istringstream fields(tail);
    int active = 0;
    uint64_t hash = 0;
    size_t count = 0;
    fields >> active >> hash >> count;
    std::vector<int> actions(count);
    for (auto &action : actions) fields >> action;
    if (!fields) throw std::runtime_error("malformed env state actions");
    if (active) {
      std::istringstream(start_gen) >> gen_;
      std::istringstream(start_duel_gen) >> duel_gen_;
      reset();
      std::vector<Array> arrays;
      for (ShapeSpec shape : spec_.state_spec.template AllValues<ShapeSpec>()) {
        for (int &dim : shape.shape)
          if (dim == -1) dim = 1;
        arrays.emplace_back(shape);
      }
      State scratch(arrays);
      WriteState(scratch);
      for (int action : actions) {
        step(action);
        WriteState(scratch);
      }
      if (repro_actions_ != actions || stream_hash_ != hash || generator_text(gen_) != gen ||
          generator_text(duel_gen_) != duel_gen) {
        throw std::runtime_error(fmt::format("exact resume: the replayed game differs from the export (actions {}, "
                                             "stream {}, generators {})", repro_actions_ == actions,
                                             stream_hash_ == hash, generator_text(gen_) == gen));
      }
    } else {
      std::istringstream(gen) >> gen_;
      std::istringstream(duel_gen) >> duel_gen_;
    }
  }

 protected:
  // The env's duel creation from an outer seed, for scripted replays of repro records.
  intptr_t create_duel_from_outer_seed(uint32_t seed) { return OCG_CreateDuel(seed); }

 private:
  intptr_t OCG_CreateDuel(uint32_t seed) {
    // Replay headers store the outer seed; standard Duel derives the core
    // seed with one MT draw before create_duel.
    std::mt19937 rnd(seed);
    return checked_duel(create_duel(rnd()));
  }

 protected:
  // The core refuses a duel beyond its live-duel capacity (or when its arena cannot be mapped): a fatal error here,
  // never a duel outside an arena.
  static intptr_t checked_duel(intptr_t pduel) {
    if (pduel == 0)
      throw std::runtime_error(fmt::format("duel creation refused: at most {} duels live in one process (the core's "
                                           "arena registry), or the arena could not be mapped",
                                           duel_live_capacity()));
    return pduel;
  }

 private:

  void OCG_SetPlayerInfo(intptr_t pduel, int32 playerid, int32 lp, int32 startcount, int32 drawcount) const {
    set_player_info(pduel, playerid, lp, startcount, drawcount);
  }

  void OCG_NewCard(intptr_t pduel, uint32 code, uint8 owner, uint8 playerid, uint8 location, uint8 sequence, uint8 position) const {
    new_card(pduel, code, owner, playerid, location, sequence, position);
  }

  void OCG_StartDuel(intptr_t pduel, int32 options) const {
    start_duel(pduel, options);
  }

  void OCG_EndDuel(intptr_t pduel) const {
    // The maintained core's end_duel removes the duel from the core's
    // lock-protected live-duel set before freeing it; deleting the object
    // directly would leave a dangling entry in that set.
    end_duel(pduel);
  }

  // Client mode: a card's query segment (card::get_infos's layout) for the fields the env asks for.
  static void client_segment(const ClientCard &c, uint32_t flags, std::vector<uint8_t> &out) {
    constexpr uint32_t supported = QUERY_CODE | QUERY_POSITION | QUERY_LEVEL | QUERY_RANK | QUERY_ATTACK |
                                   QUERY_DEFENSE | QUERY_EQUIP_CARD | QUERY_OVERLAY_CARD | QUERY_COUNTERS |
                                   QUERY_OWNER | QUERY_STATUS | QUERY_LSCALE | QUERY_RSCALE | QUERY_LINK;
    if (flags & ~supported) throw std::runtime_error(fmt::format("client mode: query flags {:#x} not supported", flags));
    std::vector<uint8_t> body;
    auto put = [&](uint32_t v) {
      for (int i = 0; i < 4; ++i) body.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
    };
    put(flags);
    const bool from_db = !c.stats_known && c.code != 0;
    const Card db = from_db ? c_get_card(c.code) : Card();
    if (flags & QUERY_CODE) put(c.code);
    if (flags & QUERY_POSITION)
      put(static_cast<uint32_t>(c.controller) | (static_cast<uint32_t>(c.location) << 8) |
          (static_cast<uint32_t>(c.sequence) << 16) | (static_cast<uint32_t>(c.position) << 24));
    if (flags & QUERY_LEVEL) put(from_db ? db.level_ : c.level);
    if (flags & QUERY_RANK) put(from_db ? 0 : c.rank);
    if (flags & QUERY_ATTACK) put(static_cast<uint32_t>(from_db ? db.attack_ : c.attack));
    if (flags & QUERY_DEFENSE) put(static_cast<uint32_t>(from_db ? db.defense_ : c.defense));
    if (flags & QUERY_EQUIP_CARD) put(c.equip);
    if (flags & QUERY_OVERLAY_CARD) {
      put(static_cast<uint32_t>(c.overlay.size()));
      for (uint32_t code : c.overlay) put(code);
    }
    if (flags & QUERY_COUNTERS) {
      put(static_cast<uint32_t>(c.counters.size()));
      for (const auto &[type, count] : c.counters) put((type & 0xFFFF) | (count << 16));
    }
    if (flags & QUERY_OWNER) put(c.owner);
    if (flags & QUERY_STATUS) put(c.status);
    if (flags & QUERY_LSCALE) put(from_db ? db.lscale_ : c.lscale);
    if (flags & QUERY_RSCALE) put(from_db ? db.rscale_ : c.rscale);
    if (flags & QUERY_LINK) {
      // off the field the engine reports a Link Monster's rating and markers here (the database keeps the rating
      // in its level and the markers apart)
      const bool link = from_db && (db.type_ & TYPE_LINK);
      put(from_db ? (link ? db.level_ : 0) : c.link);
      put(from_db ? (link ? db.link_marker_ : 0) : c.link_marker);
    }
    const uint32_t length = static_cast<uint32_t>(body.size() + 4);
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((length >> (8 * i)) & 0xFF));
    out.insert(out.end(), body.begin(), body.end());
  }

  const std::vector<std::optional<ClientCard>> &client_zone(uint8_t player, uint8_t location) const {
    static const std::vector<std::optional<ClientCard>> empty;
    const auto it = client_cards_.find({player, location});
    return it == client_cards_.end() ? empty : it->second;
  }

  static int32_t client_copy(const std::vector<uint8_t> &bytes, byte *buf) {
    if (bytes.size() > 0x40000) throw std::runtime_error("client mode: a query buffer over 256 KB");
    std::memcpy(buf, bytes.data(), bytes.size());
    return static_cast<int32_t>(bytes.size());
  }

  int32 OCG_GetMessage(intptr_t pduel, byte* buf) {
    return get_message(pduel, buf);
  }

  uint32 OCG_Process(intptr_t pduel) {
    return process(pduel);
  }

  int32 OCG_QueryCard(intptr_t pduel, uint8 playerid, uint8 location, uint8 sequence, int32 query_flag, byte* buf) {
    if (client_mode_) {
      std::vector<uint8_t> out;
      const auto &zone = client_zone(playerid, location);
      if (sequence < zone.size() && zone[sequence]) {
        client_segment(*zone[sequence], static_cast<uint32_t>(query_flag), out);
      } else {
        out = {LEN_EMPTY & 0xFF, 0, 0, 0};
      }
      return client_copy(out, buf);
    }
    return query_card(pduel, playerid, location, sequence, query_flag, buf, 0);
  }

  int32 OCG_QueryFieldCount(intptr_t pduel, uint8 playerid, uint8 location) {
    if (client_mode_) {
      int32 n = 0;
      for (const auto &c : client_zone(playerid, location)) n += c.has_value() ? 1 : 0;
      return n;
    }
    return query_field_count(pduel, playerid, location);
  }

  int32 OCG_QueryFieldCard(intptr_t pduel, uint8 playerid, uint8 location, uint32 query_flag, byte* buf) {
    if (client_mode_) {
      std::vector<uint8_t> out;
      for (const auto &c : client_zone(playerid, location)) {
        if (c) {
          client_segment(*c, query_flag, out);
        } else {
          for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((LEN_EMPTY >> (8 * i)) & 0xFF));
        }
      }
      return client_copy(out, buf);
    }
    return query_field_card(pduel, playerid, location, query_flag, buf, 0);
  }

  void OCG_SetResponsei(intptr_t pduel, int32 value) {
    if (record_) {
      ReplayWriteInt8(4);
      ReplayWriteInt32(value);
    }
    std::vector<uint8_t> bytes(sizeof(value));
    std::memcpy(bytes.data(), &value, sizeof(value));
    if (response_log_) {
      response_log_->push_back(bytes);
    }
    repro_responses_.push_back(std::move(bytes));
    if (!client_mode_) set_responsei(pduel, value);
  }

  // The meaningful length of a byte response to the current prompt (what a replay or a client carries).
  size_t response_size(const byte *buf) const {
    switch (msg_) {
      case MSG_SORT_CARD: return 1;
      case MSG_SELECT_COUNTER: return 2 * n_counters_;
      case MSG_SELECT_PLACE:
      case MSG_SELECT_DISFIELD: return 3;
      default: return static_cast<size_t>(buf[0]) + 1;
    }
  }

  void OCG_SetResponseb(intptr_t pduel, byte* buf) {
    const size_t size = response_size(buf);
    if (record_) {
      ReplayWriteInt8(static_cast<int8_t>(size));
      fwrite(buf, size, 1, fp_);
    }
    if (response_log_) response_log_->emplace_back(buf, buf + size);
    repro_responses_.emplace_back(buf, buf + size);
    if (!client_mode_) set_responseb(pduel, buf);
  }

  // duel-core API

  void show_decision(int idx) {
    std::string s;
    const auto& a = legal_actions_[idx];
    if (!a.spec_.empty()) {
      s = a.spec_;
    } else if (a.place_ != ActionPlace::None) {
      s = action_place_to_string(a.place_);
    } else if (a.position_ != 0) {
      s = position_to_string(a.position_);
    } else {
      s = fmt::format("{}", a);
    }
    fmt::print("Player {} chose \"{}\" in {}\n", to_play_, s, legal_actions_);
  }

  std::tuple<std::vector<CardCode>, std::vector<CardCode>, std::string>
  load_deck(
    intptr_t pduel, PlayerId player, const std::string &deck_name,
    std::mt19937& gen, bool shuffle = true) const {
    std::vector<CardCode> main_deck = main_decks_.at(deck_name);
    std::vector<CardCode> extra_deck = extra_decks_.at(deck_name);

    if (verbose_) {
      fmt::println("{} {}: {}, main({}), extra({})", player, nickname_[player],
        deck_name, main_deck.size(), extra_deck.size());
    }

    if (shuffle) {
      std::shuffle(main_deck.begin(), main_deck.end(), gen);
    }

    // Keep the insertion order aligned with this environment's replay writer.
    for (int i = 0; i < main_deck.size(); i++) {
      OCG_NewCard(pduel, main_deck[i], player, player, LOCATION_DECK, 0,
               POS_FACEDOWN_DEFENSE);
    }

    // add extra deck in reverse order following duel
    for (int i = int(extra_deck.size()) - 1; i >= 0; --i) {
      OCG_NewCard(pduel, extra_deck[i], player, player, LOCATION_EXTRA, 0,
               POS_FACEDOWN_DEFENSE);
    }

    return {main_deck, extra_deck, deck_name};
  }

  void next() {
    if (client_mode_) return;  // client mode: the stream arrives through ClientDuel::Feed
    while (duel_started_) {
      if (eng_flag_ == PROCESSOR_END) {
        break;
      }
      uint32_t res = OCG_Process(pduel_);
      dl_ = res & PROCESSOR_BUFFER_LEN;
      eng_flag_ = res & PROCESSOR_FLAG;

      if (dl_ == 0) {
        continue;
      }
      OCG_GetMessage(pduel_, data_);
      check_target_shortfall();
      if (!core_script_errors().empty()) {
        std::string text = core_script_errors().front();
        const size_t count = core_script_errors().size();
        core_script_errors().clear();
        throw std::runtime_error(fmt::format("core script error ({} in this buffer): {}", count, text));
      }
      if (consume_buffer()) return;
    }
    done_ = true;
    legal_actions_.clear();
  }

 protected:
  // refreshed_view/v1: refresh a player's monster zone (or one slot) as the server's query with ``flags`` does: the
  // fields it carries take the engine's values now (STATUS 0x80000 only when the flags hold it).
  void refresh_monsters(PlayerId player, uint32_t flags, int only_sequence = -1) {
    for (const Card &c : get_cards_in_location(player, LOCATION_MZONE)) {
      if (c.location_ & LOCATION_OVERLAY) continue;
      if (only_sequence >= 0 && c.sequence_ != only_sequence) continue;
      if (c.sequence_ >= 7) throw std::runtime_error("a monster zone sequence past 6");
      RefreshedStats &r = refreshed_mzone_.at(player).at(c.sequence_);
      r.valid = true;
      r.level = c.level_;
      r.attack = c.attack_;
      r.defense = c.defense_;
      if (flags & QUERY_STATUS) r.status = c.status_;
    }
  }

  void refresh_all_monsters() {
    refresh_monsters(0, 0x881fff);
    refresh_monsters(1, 0x881fff);
  }

  // The server's refreshes around one message (netduel/host_view.py deliver), for the monster zone only; ``before``
  // the message or after it. The moves and swaps carry a card's values with it first.
  void host_refresh(int msg, const uint8_t *p, size_t n, bool before) {
    auto need = [&](size_t bytes) {
      if (n < bytes) throw std::runtime_error(fmt::format("refreshed view: message {} of {} bytes", msg, n));
    };
    if (before) {
      if (msg == MSG_SELECT_IDLECMD || msg == MSG_SELECT_BATTLECMD || msg == MSG_NEW_TURN) refresh_all_monsters();
      if (msg == MSG_FLIPSUMMONING) {
        need(8);
        if (p[5] == LOCATION_MZONE) refresh_monsters(p[4], 0xf81fff, p[6]);
      }
      return;
    }
    switch (msg) {
      case MSG_NEW_PHASE:
      case MSG_CHAINED:
      case MSG_CHAIN_SOLVED:
      case MSG_CHAIN_END:
      case MSG_SUMMONED:
      case MSG_SPSUMMONED:
      case MSG_FLIPSUMMONED:
      case MSG_DAMAGE_STEP_START:
      case MSG_DAMAGE_STEP_END:
        refresh_all_monsters();
        return;
      case MSG_SHUFFLE_SET_CARD:
        need(1);
        if (p[0] == LOCATION_MZONE) {
          refresh_monsters(0, 0x181fff);
          refresh_monsters(1, 0x181fff);
        }
        return;
      case MSG_MOVE: {
        need(16);
        const uint8_t pc = p[4], pl = p[5], ps = p[6], cc = p[8], cl = p[9], cs = p[10];
        RefreshedStats carried;
        if (pl == LOCATION_MZONE && pc < 2 && ps < 7) {
          carried = refreshed_mzone_[pc][ps];
          refreshed_mzone_[pc][ps] = RefreshedStats{};
        }
        if (cl == LOCATION_MZONE && cc < 2 && cs < 7) refreshed_mzone_[cc][cs] = carried;
        if (cl != 0 && !(cl & LOCATION_OVERLAY) && (cl != pl || pc != cc) && cl == LOCATION_MZONE)
          refresh_monsters(cc, 0xf81fff, cs);
        return;
      }
      case MSG_SWAP: {
        need(16);
        const uint8_t c1 = p[4], l1 = p[5], s1 = p[6], c2 = p[12], l2 = p[13], s2 = p[14];
        if (l1 == LOCATION_MZONE && l2 == LOCATION_MZONE && c1 < 2 && c2 < 2 && s1 < 7 && s2 < 7)
          std::swap(refreshed_mzone_[c1][s1], refreshed_mzone_[c2][s2]);
        if (l1 == LOCATION_MZONE) refresh_monsters(c1, 0xf81fff, s1);
        if (l2 == LOCATION_MZONE) refresh_monsters(c2, 0xf81fff, s2);
        return;
      }
      case MSG_POS_CHANGE: {
        need(9);
        const uint8_t cc = p[4], cl = p[5], cs = p[6], pp = p[7], cp = p[8];
        if (cl == LOCATION_MZONE && (pp & POS_FACEDOWN) && (cp & POS_FACEUP)) refresh_monsters(cc, 0xf81fff, cs);
        return;
      }
      default:
        return;
    }
  }

  // Audience follows netduel/host_view.py deliver. Private opponent prompts
  // must not expire this seat's hint: its ClientDuel never receives them.
  unsigned placement_message_audience(int msg, const uint8_t *p, size_t n) const {
    auto only = [&](size_t at) -> unsigned {
      if (n <= at || p[at] > 1) throw std::runtime_error("placement context: malformed message audience");
      return 1u << p[at];
    };
    switch (msg) {
      case MSG_HINT:
        if (n < 6) throw std::runtime_error("placement context: malformed hint");
        switch (p[0]) {
          case 1: case 2: case 3: case 5: return only(1);
          case 4: case 6: case 7: case 8: case 9: case 11: return only(1) ^ 3u;
          case 10: return 3;
          default: return 0;
        }
      case MSG_SELECT_SUM: return only(1);
      case MSG_SELECT_BATTLECMD: case MSG_SELECT_IDLECMD: case MSG_SELECT_EFFECTYN:
      case MSG_SELECT_YESNO: case MSG_SELECT_OPTION: case MSG_SELECT_CHAIN:
      case MSG_SELECT_PLACE: case MSG_SELECT_DISFIELD: case MSG_SELECT_POSITION:
      case MSG_SELECT_COUNTER: case MSG_SORT_CARD: case MSG_SELECT_CARD:
      case MSG_SELECT_TRIBUTE: case MSG_SELECT_UNSELECT_CARD: case MSG_MISSED_EFFECT:
      case MSG_ROCK_PAPER_SCISSORS: case MSG_ANNOUNCE_RACE: case MSG_ANNOUNCE_ATTRIB:
      case MSG_ANNOUNCE_CARD: case MSG_ANNOUNCE_NUMBER: return only(0);
      case MSG_CONFIRM_CARDS: return n > 8 && p[8] == LOCATION_DECK ? only(0) : 3;
      case MSG_RANDOM_SELECTED: return only(0) | 2u;  // host routes to player, then player 1
      case MSG_WIN: case MSG_CONFIRM_DECKTOP: case MSG_CONFIRM_EXTRATOP:
      case MSG_SHUFFLE_DECK: case MSG_SHUFFLE_HAND: case MSG_SHUFFLE_EXTRA:
      case 34 /* MSG_REFRESH_DECK, legacy wire */: case MSG_REVERSE_DECK: case MSG_SWAP_GRAVE_DECK:
      case MSG_DECK_TOP: case MSG_SHUFFLE_SET_CARD: case MSG_NEW_TURN: case MSG_NEW_PHASE:
      case MSG_MOVE: case MSG_SWAP: case MSG_POS_CHANGE: case MSG_SET:
      case MSG_SUMMONING: case MSG_SUMMONED: case MSG_SPSUMMONING: case MSG_SPSUMMONED:
      case MSG_FLIPSUMMONING: case MSG_FLIPSUMMONED: case MSG_CHAINING: case MSG_CHAINED:
      case MSG_CHAIN_SOLVED: case MSG_CHAIN_END: case MSG_CHAIN_SOLVING:
      case MSG_CHAIN_NEGATED: case MSG_CHAIN_DISABLED: case MSG_BECOME_TARGET: case MSG_DRAW:
      case MSG_DAMAGE_STEP_START: case MSG_DAMAGE_STEP_END: case MSG_DAMAGE: case MSG_RECOVER:
      case MSG_EQUIP: case MSG_LPUPDATE: case 95 /* MSG_UNEQUIP, legacy wire */: case MSG_CARD_TARGET:
      case MSG_CANCEL_TARGET: case MSG_PAY_LPCOST: case MSG_ADD_COUNTER: case MSG_REMOVE_COUNTER:
      case MSG_ATTACK: case MSG_BATTLE: case MSG_ATTACK_DISABLED: case MSG_TOSS_COIN:
      case MSG_TOSS_DICE: case MSG_HAND_RES: case MSG_CARD_HINT: case MSG_PLAYER_HINT:
      case MSG_FIELD_DISABLED: return 3;
      default: return 0;  // dropped by the host, including CARD_SELECTED and MATCH_KILL
    }
  }

  void invalidate_placement_zone(int controller, int location, int sequence = -1) {
    for (PlayerId viewer = 0; viewer < 2; ++viewer) {
      auto &anchor = placement_source_[viewer];
      if (anchor.spec.empty()) continue;
      const auto [c, l, s, position] = spec_to_ls(viewer, anchor.spec);
      const bool field = l == LOCATION_MZONE || l == LOCATION_SZONE;
      if (c == controller && l == location && (!field || sequence < 0 || s == sequence)) anchor = {};
    }
  }

  // Called only for a newly received message, not handle_message's menu rebind
  // during clone/rollback. All state is included in the env snapshot below.
  void observe_placement_message(int msg, const uint8_t *p, size_t n) {
    const unsigned audience = placement_message_audience(msg, p, n);
    placement_prompt_cid_ = 0;
    // DISFIELD selects/blocks zones; even a card-named hint can identify the
    // effect handler, not a card being placed. This protocol supplies no such
    // placement witness, so its card reference remains explicitly unknown.
    if (msg == MSG_SELECT_PLACE && n >= 1 && p[0] < 2)
      placement_prompt_cid_ = placement_hint_[p[0]];
    for (int viewer = 0; viewer < 2; ++viewer)
      if (audience & (1u << viewer)) placement_hint_[viewer] = 0;
    if (msg == MSG_HINT && p[0] == HINT_SELECTMSG) {
      const uint32_t code = static_cast<uint32_t>(p[2]) | (static_cast<uint32_t>(p[3]) << 8) |
                            (static_cast<uint32_t>(p[4]) << 16) | (static_cast<uint32_t>(p[5]) << 24);
      const auto found = card_ids_.find(code);
      // Ordinary HINTMSG/string numbers are not card identities.
      if (code > 2000 && found != card_ids_.end() && cards_.find(code) != cards_.end())
        placement_hint_[p[1]] = found->second;
    }
    if (is_source_action_msg(msg) && n >= 1 && p[0] < 2)
      placement_source_[p[0]] = {};  // a new own command/chain question ends the old placement context
    if (msg == MSG_NEW_TURN || msg == MSG_NEW_PHASE || msg == MSG_CHAIN_END || msg == MSG_SWAP ||
        msg == MSG_CHAINING || msg == MSG_CHAINED || msg == MSG_CHAIN_SOLVING || msg == MSG_CHAIN_SOLVED ||
        msg == MSG_SET || msg == MSG_SUMMONING || msg == MSG_SUMMONED ||
        msg == MSG_SPSUMMONING || msg == MSG_SPSUMMONED || msg == MSG_FLIPSUMMONING || msg == MSG_FLIPSUMMONED) {
      placement_source_ = {};
    } else if (msg == MSG_MOVE && n >= 12) {
      invalidate_placement_zone(p[4], p[5], p[6]);
      invalidate_placement_zone(p[8], p[9], p[10]);
    } else if ((msg == MSG_SHUFFLE_HAND || msg == MSG_SHUFFLE_EXTRA || msg == MSG_SHUFFLE_DECK ||
                msg == MSG_DRAW || msg == MSG_SWAP_GRAVE_DECK) && n >= 1) {
      const int location = msg == MSG_SHUFFLE_EXTRA ? LOCATION_EXTRA :
                           msg == MSG_SHUFFLE_HAND || msg == MSG_DRAW ? LOCATION_HAND : LOCATION_DECK;
      invalidate_placement_zone(p[0], location);
      if (msg == MSG_SWAP_GRAVE_DECK) invalidate_placement_zone(p[0], LOCATION_GRAVE);
    } else if (msg == MSG_SHUFFLE_SET_CARD && n >= 1) {
      invalidate_placement_zone(0, p[0]);
      invalidate_placement_zone(1, p[0]);
    }
  }

  // The current buffer (data_, dl_): its stream digest, the public history, each message's handler, and the
  // prompts with one option answered inside. True when it stops at a prompt with more than one option (the player
  // decides); false when the buffer is consumed.
  bool consume_buffer() {
    for (int i = 0; i < dl_; ++i) {
      stream_hash_ = (stream_hash_ ^ data_[i]) * 1099511628211ULL;
    }
    history_.ConsumeBuffer(data_, static_cast<size_t>(dl_));
    history_.ShowHands({public_hand(0), public_hand(1)});
    // Message boundaries from mfenv's message splitter (the parser the
    // contract tests hold byte-exact). Each handler must start at a
    // boundary and end at its message's end; a handler that ignores its
    // message (dp_ = dl_) skips that message only, never the ones after it.
    message_starts_.clear();
    {
      int offset = 0;
      for (const auto &message : mfenv::SplitMessages(data_, static_cast<size_t>(dl_))) {
        message_starts_.push_back(offset);
        offset += 1 + static_cast<int>(message.payload.size());
      }
      message_starts_.push_back(offset);
    }
    if (shortfall_armed_)
      for (size_t k = 0; k + 1 < message_starts_.size(); ++k)
        if (data_[message_starts_[k]] == MSG_CHAINED) {
          shortfall_armed_ = false;  // the activation is chained: later shortfalls are not its targets
          chained_seen_ = true;
        }
    dp_ = 0;
    while ((dp_ != dl_) || (ms_idx_ != -1)) {
      if (ms_idx_ != -1) {
        handle_multi_select();
      } else {
        const auto at = std::find(message_starts_.begin(), message_starts_.end(), dp_);
        if (at == message_starts_.end() || at + 1 == message_starts_.end()) {
          throw std::runtime_error(fmt::format("protocol divergence: a handler starts at byte {} of {}, not at a "
                                               "message boundary", dp_, dl_));
        }
        const int start = dp_, end = *(at + 1);
        if (!client_mode_)
          host_refresh(data_[start], data_ + start + 1, static_cast<size_t>(end - start - 1), true);
        observe_placement_message(data_[start], data_ + start + 1, static_cast<size_t>(end - start - 1));
        handle_message();
        if (!client_mode_)
          host_refresh(data_[start], data_ + start + 1, static_cast<size_t>(end - start - 1), false);
        if (dp_ == dl_ && end < dl_ && legal_actions_.empty()) {
          dp_ = end;
        } else if (dp_ != end) {
          throw std::runtime_error(fmt::format("protocol divergence: {} at byte {} was parsed to byte {}, the "
                                               "message ends at byte {} of {}", msg_to_string(msg_), start, dp_,
                                               end, dl_));
        }
        if (legal_actions_.empty()) {
          continue;
        }
      }
      if ((play_mode_ == kSelfPlay) || (to_play_ == ai_player_)) {
        if (legal_actions_.size() == 1) {
          callback_(0);
          auto la = legal_actions_[0];
          la.msg_ = msg_;
          if (la.cid_ == 0 && !la.spec_.empty()) {
            la.cid_ = spec_to_card_id(la.spec_, to_play_);
          }
          update_history_actions(to_play_, la);
          record_public_event(to_play_, la);
          if (verbose_) {
            show_decision(0);
          }
        } else {
          return true;
        }
      } else {
        auto idx = players_[to_play_]->think(legal_actions_);
        callback_(idx);
        if (verbose_) {
          show_decision(idx);
        }
      }
    }
    return false;
  }

 private:
  uint8_t read_u8() { return data_[dp_++]; }

  uint16_t read_u16() {
    uint16_t v = *reinterpret_cast<uint16_t *>(data_ + dp_);
    dp_ += 2;
    return v;
  }

  uint32 read_u32() {
    uint32 v = *reinterpret_cast<uint32_t *>(data_ + dp_);
    dp_ += 4;
    return v;
  }

  uint32 q_read_u8() {
    uint8_t v = *reinterpret_cast<uint8_t *>(query_buf_ + qdp_);
    qdp_ += 1;
    return v;
  }

  uint32 q_read_u32() {
    uint32_t v = *reinterpret_cast<uint32_t *>(query_buf_ + qdp_);
    qdp_ += 4;
    return v;
  }

  CardCode get_card_code(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE;
    int32_t bl = OCG_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      throw std::runtime_error("[get_card_code] Invalid card");
    }
    qdp_ += 8;
    return q_read_u32();
  }

  Card get_card(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE | QUERY_ATTACK | QUERY_DEFENSE | QUERY_POSITION |
                    QUERY_LEVEL | QUERY_RANK | QUERY_LSCALE | QUERY_RSCALE |
                    QUERY_LINK;
    int32_t bl = OCG_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      show_deck(0);
      show_deck(1);
      show_turn();
      show_buffer();
      auto s = fmt::format("[get_card] Invalid card (bl <= 0), player: {}, loc: {}, seq: {}", player, loc, seq);
      throw std::runtime_error(s);
    }
    uint32_t f = q_read_u32();
    if (f == LEN_EMPTY) {
      return Card();
    }
    f = q_read_u32();
    CardCode code = q_read_u32();
    // client mode: a card the client may not identify arrives with code 0 (no card data)
    Card c = (client_mode_ && code == 0) ? Card() : c_get_card(code);
    uint32_t position = q_read_u32();
    c.set_location(position);
    uint32_t level = q_read_u32();
    if ((level & 0xff) > 0) {
      c.level_ = level & 0xff;
    }
    uint32_t rank = q_read_u32();
    if ((rank & 0xff) > 0) {
      c.level_ = rank & 0xff;
    }
    c.attack_ = q_read_u32();
    c.defense_ = q_read_u32();
    c.lscale_ = q_read_u32();
    c.rscale_ = q_read_u32();
    uint32_t link = q_read_u32();
    uint32_t link_marker = q_read_u32();
    if ((link & 0xff) > 0) {
      c.level_ = link & 0xff;
    }
    if (link_marker > 0) {
      c.defense_ = link_marker;
    }
    return c;
  }

  // A player's public (face-up) hand cards, (sequence, code): what its opponent's client is shown of that hand.
  std::vector<std::pair<int, int64_t>> public_hand(PlayerId player) {
    const int32_t length = OCG_QueryFieldCard(pduel_, player, LOCATION_HAND, QUERY_CODE | QUERY_POSITION, query_buf_);
    std::vector<std::pair<int, int64_t>> out;
    int32_t at = 0;
    while (at < length) {
      uint32_t size = 0, code = 0;
      std::memcpy(&size, query_buf_ + at, 4);
      if (size == LEN_EMPTY) {
        at += 4;
        continue;
      }
      if (size < 16 || at + static_cast<int32_t>(size) > length)
        throw std::runtime_error(fmt::format("malformed hand query record of {} bytes at {} of {}", size, at, length));
      std::memcpy(&code, query_buf_ + at + 8, 4);
      const uint8_t sequence = query_buf_[at + 14], position = query_buf_[at + 15];
      if (!(position & POS_FACEDOWN)) out.emplace_back(sequence, code);
      at += static_cast<int32_t>(size);
    }
    return out;
  }

  std::vector<Card> get_cards_in_location(PlayerId player, uint8_t loc) {
    int32_t flags = QUERY_CODE | QUERY_POSITION | QUERY_LEVEL | QUERY_RANK |
                    QUERY_ATTACK | QUERY_DEFENSE | QUERY_EQUIP_CARD |
                    QUERY_OVERLAY_CARD | QUERY_COUNTERS | QUERY_STATUS |
                    QUERY_LSCALE | QUERY_RSCALE | QUERY_LINK;
    int32_t bl = OCG_QueryFieldCard(pduel_, player, loc, flags, query_buf_);
    qdp_ = 0;
    std::vector<Card> cards;
    while (true) {
      if (qdp_ >= bl) {
        break;
      }
      uint32_t f = q_read_u32();
      if (f == LEN_EMPTY) {
        continue;
        ;
      }
      f = q_read_u32();
      CardCode code = q_read_u32();
      // client mode: a card the client may not identify arrives with code 0 (no card data)
      Card c = (client_mode_ && code == 0) ? Card() : c_get_card(code);

      uint8_t controller = q_read_u8();
      uint8_t location = q_read_u8();
      uint8_t sequence = q_read_u8();
      uint8_t position = q_read_u8();
      c.controler_ = controller;
      c.location_ = location;
      c.sequence_ = sequence;
      c.position_ = position;

      uint32_t level = q_read_u32();
      if ((level & 0xff) > 0) {
        c.level_ = level & 0xff;
      }
      uint32_t rank = q_read_u32();
      if ((rank & 0xff) > 0) {
        c.level_ = rank & 0xff;
      }
      c.attack_ = q_read_u32();
      c.defense_ = q_read_u32();

      if (f & QUERY_EQUIP_CARD) {
        c.equip_target_ = q_read_u32();  // audits only (public_status.h follows equips from the public messages)
      }

      uint32_t n_xyz = q_read_u32();
      for (int i = 0; i < n_xyz; ++i) {
        auto code = q_read_u32();
        Card c_ = c_get_card(code);
        c_.controler_ = controller;
        c_.location_ = location | LOCATION_OVERLAY;
        c_.sequence_ = sequence;
        c_.position_ = i;
        cards.push_back(c_);
      }

      // counters: each entry packs the counter type (low 16 bits) and its count (high 16 bits); the row's counter
      // field is the total count over all types (design item 7a)
      uint32_t n_counters = q_read_u32();
      c.counter_ = 0;
      for (uint32_t i = 0; i < n_counters; ++i) {
        c.counter_ += q_read_u32() >> 16;
      }

      c.status_ = q_read_u32();
      c.lscale_ = q_read_u32();
      c.rscale_ = q_read_u32();

      uint32_t link = q_read_u32();
      uint32_t link_marker = q_read_u32();
      if ((link & 0xff) > 0) {
        c.level_ = link & 0xff;
      }
      if (link_marker > 0) {
        c.defense_ = link_marker;
      }
      cards.push_back(c);
    }
    return cards;
  }

  std::vector<Card> read_cardlist(bool extra = false, bool extra8 = false) {
    std::vector<Card> cards;
    auto count = read_u8();
    cards.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto card = get_card(controller, loc, seq);
      if (extra) {
        if (extra8) {
          card.data_ = read_u8();
        } else {
          card.data_ = read_u32();
        }
      }
      cards.push_back(card);
    }
    return cards;
  }

  std::vector<IdleCardSpec> read_cardlist_spec(PlayerId player, bool extra = false, bool extra8 = false) {
    std::vector<IdleCardSpec> card_specs;
    auto count = read_u8();
    card_specs.reserve(count);
    for (int i = 0; i < count; ++i) {
      CardCode code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      uint32_t data = 0;
      if (extra) {
        if (extra8) {
          data = read_u8();
        } else {
          data = read_u32();
        }
      }
      card_specs.push_back({code, ls_to_spec(loc, seq, 0, player != controller), data});
    }
    return card_specs;
  }

  std::tuple<CardCode, int> unpack_desc(CardCode code, uint32_t desc) {
    if (desc < DESCRIPTION_LIMIT) {
      return {0, desc};
    }
    CardCode code_ = desc >> 4;
    int idx = desc & 0xf;
    if (idx < 0 || idx >= 14) {
      fmt::print("Code: {}, Code_: {}, Desc: {}\n", code, code_, desc);
      show_deck(0);
      show_deck(1);
      show_buffer();
      show_turn();
      throw std::runtime_error("Invalid effect index: " + std::to_string(idx));
    }
    return {code_, idx + CARD_EFFECT_OFFSET};
  }

  std::string cardlist_info_for_player(const Card &card, PlayerId pl) {
    std::string spec = card.get_spec(pl);
    if (card.location_ == LOCATION_DECK) {
      spec = "deck";
    }
    if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
      return position_to_string(card.position_) + "card (" + spec + ")";
    }
    return card.name_ + " (" + spec + ")";
  }

  void bind_own_deck_selections(std::vector<Card> &candidates, PlayerId player) {
    std::optional<std::vector<Card>> deck;
    std::set<uint32_t> used;
    for (auto &candidate : candidates) {
      if (candidate.controler_ != player || candidate.location_ != LOCATION_DECK) {
        continue;
      }
      if (!deck) deck = get_cards_in_location(player, LOCATION_DECK);
      // Prompt-local numbers are not positions in the hidden shuffled deck.
      const auto found = std::find_if(deck->begin(), deck->end(), [&](const Card &card) {
        return card.code_ == candidate.code_ && !used.count(card.sequence_);
      });
      if (found == deck->end()) {
        throw std::runtime_error("Own-deck selection identity/count mismatch");
      }
      used.insert(found->sequence_);
      candidate.sequence_ = found->sequence_;
    }
  }

  // This function does the following:
  // 1. read msg_ from data_ and update dp_
  // 2. (optional) print information if verbose_ is true
  // 3. update to_play_ and options_ if need action
  // A card a message names: in client mode a card the server did not identify to this seat arrives with code 0.
  Card msg_card(CardCode code) const { return (client_mode_ && code == 0) ? Card() : c_get_card(code); }

 protected:
  void handle_message() {
    msg_ = int(data_[dp_++]);
    legal_actions_ = {};
    // Whether this message directly follows a summon, special summon or set
    // command: the core's choice among the card's procedures comes next.
    const bool procedure_choice = std::exchange(procedure_choice_, false);
    end_reveals(msg_, data_ + dp_, static_cast<size_t>(dl_ - dp_));
    note_control_changes(msg_, data_ + dp_, static_cast<size_t>(dl_ - dp_));

    if (verbose_) {
      fmt::println("Message {}, length {}, dp {}", msg_to_string(msg_), dl_, dp_);
    }

    if (msg_ == MSG_DRAW) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto drawed = read_u8();
      std::vector<uint32> codes;
      for (int i = 0; i < drawed; ++i) {
        uint32 code = read_u32();
        codes.push_back(code & 0x7fffffff);
      }
      const auto &pl = players_[player];
      pl->notify(fmt::format("Drew {} cards:", drawed));
      for (int i = 0; i < drawed; ++i) {
        const auto &c = c_get_card(codes[i]);
        pl->notify(fmt::format("{}: {}", i + 1, c.name_));
      }
      const auto &op = players_[1 - player];
      op->notify(fmt::format("Opponent drew {} cards.", drawed));
    } else if (msg_ == MSG_TOSS_COIN || msg_ == MSG_TOSS_DICE) {
      const auto player = read_u8();
      const int count = read_u8();
      if (count < 1 || count > 5 || dp_ + count > dl_) {
        throw std::runtime_error(fmt::format(
            "Invalid {} payload: count={}, remaining={}",
            msg_to_string(msg_), count, dl_ - dp_));
      }
      std::vector<uint8_t> results;
      results.reserve(count);
      for (int i = 0; i < count; ++i) {
        const auto result = read_u8();
        const bool valid = msg_ == MSG_TOSS_COIN
            ? result <= 1
            : result >= 1 && result <= 6;
        if (!valid) {
          throw std::runtime_error(fmt::format(
              "Invalid {} result: {}", msg_to_string(msg_), result));
        }
        results.push_back(result);
      }
      record_protocol_event(player, msg_, results);
      if (verbose_) {
        fmt::println(
            "{} player={} count={} results={}",
            msg_to_string(msg_), player, count, results);
      }
      return;
    } else if (msg_ == MSG_NEW_TURN) {
      tp_ = int(read_u8());
      turn_count_++;
      if (!verbose_) {
        return;
      }
      auto& player = players_[tp_];
      player->notify("Your turn.");
      players_[1 - tp_]->notify(fmt::format("{}'s turn.", player->nickname_));
    } else if (msg_ == MSG_NEW_PHASE) {
      current_phase_ = int(read_u16());
      if (!verbose_) {
        return;
      }
      auto phase_str = phase_to_string(current_phase_);
      for (int i = 0; i < 2; ++i) {
        players_[i]->notify(fmt::format("Entering {} phase.", phase_str));
      }
    } else if (msg_ == MSG_FIELD_DISABLED) {
      const int payload_size = dl_ - dp_;
      if (payload_size < static_cast<int>(sizeof(uint32_t))) {
        throw std::runtime_error(fmt::format(
            "Truncated field_disabled payload: {}", payload_size));
      }
      disabled_field_ = read_u32();
      if (verbose_) {
        fmt::println(
            "Disabled field mask updated: player0=0x{:04x}, "
            "player1=0x{:04x}",
            disabled_field_ & 0xffff, disabled_field_ >> 16);
      }
      return;
    } else if (msg_ == MSG_MOVE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      uint32_t location = read_u32();
      uint32_t newloc = read_u32();
      uint32_t reason = read_u32();
      Card card = msg_card(code);
      card.set_location(location);
      Card cnew = msg_card(code);
      cnew.set_location(newloc);
      auto& pl = players_[card.controler_];
      auto& op = players_[1 - card.controler_];

      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto plnewspec = cnew.get_spec(false);
      auto opnewspec = cnew.get_spec(true);

      auto getspec = [&](auto& p) { return p.get() == pl.get() ? plspec : opspec; };
      auto getnewspec = [&](auto& p) {
        return p.get() == pl.get() ? plnewspec : opnewspec;
      };
      bool card_visible = true;
      if ((card.position_ & POS_FACEDOWN) && (cnew.position_ & POS_FACEDOWN)) {
        card_visible = false;
      }
      auto getvisiblename = [&](auto& p) {
        return card_visible ? card.name_ : "Face-down card";
      };

      if ((reason & REASON_DESTROY) && (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("Card {} ({}) destroyed.", plspec, card.name_));
        op->notify(fmt::format("Card {} ({}) destroyed.", opspec, card.name_));
      } else if ((card.location_ == cnew.location_) &&
                 (card.location_ & LOCATION_ONFIELD)) {
        if (card.controler_ != cnew.controler_) {
          pl->notify(
              fmt::format("Your card {} ({}) changed controller to {} and is "
                          "now located at {}.",
                          plspec, card.name_, op->nickname_, plnewspec));
          op->notify(
              fmt::format("You now control {}'s card {} ({}) and it's located "
                          "at {}.",
                          pl->nickname_, opspec, card.name_, opnewspec));
        } else {
          pl->notify(fmt::format("Your card {} ({}) switched its zone to {}.",
                                 plspec, card.name_, plnewspec));
          op->notify(fmt::format("{}'s card {} ({}) switched its zone to {}.",
                                 pl->nickname_, opspec, card.name_, opnewspec));
        }
      } else if ((reason & REASON_DISCARD) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You discarded {} ({})", plspec, card.name_));
        op->notify(fmt::format("{} discarded {} ({})", pl->nickname_, opspec,
                               card.name_));
      } else if ((card.location_ == LOCATION_REMOVED) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your banished card {} ({}) returns to the field at "
                        "{}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s banished card {} ({}) returns to the field at "
                        "{}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((card.location_ == LOCATION_GRAVE) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((cnew.location_ == LOCATION_HAND) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(
            fmt::format("Card {} ({}) returned to hand.", plspec, card.name_));
      } else if ((reason & (REASON_RELEASE | REASON_SUMMON)) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You tribute {} ({}).", plspec, card.name_));
        op->notify(fmt::format("{} tributes {} ({}).", pl->nickname_, opspec,
                               getvisiblename(op)));
      } else if ((card.location_ == (LOCATION_OVERLAY | LOCATION_MZONE)) &&
                 (cnew.location_ & LOCATION_GRAVE)) {
        pl->notify(fmt::format("You detached {}.", card.name_));
        op->notify(fmt::format("{} detached {}.", pl->nickname_, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_GRAVE)) {
        pl->notify(fmt::format("Your card {} ({}) was sent to the graveyard.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was sent to the graveyard.",
                               pl->nickname_, opspec, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_REMOVED)) {
        pl->notify(
            fmt::format("Your card {} ({}) was banished.", plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was banished.", pl->nickname_,
                               opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_DECK)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your deck.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) returned to their deck.",
                               pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_EXTRA)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your extra deck.",
                               plspec, card.name_));
        op->notify(
            fmt::format("{}'s card {} ({}) returned to their extra deck.",
                        pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ == LOCATION_DECK) &&
                 (cnew.location_ == LOCATION_SZONE) &&
                 (cnew.position_ != POS_FACEDOWN)) {
        pl->notify(fmt::format("Activating {} ({})", plnewspec, card.name_));
        op->notify(fmt::format("{} activating {} ({})", pl->nickname_, opspec,
                               cnew.name_));
      }
    } else if (msg_ == MSG_SWAP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code1 = read_u32();
      uint32_t loc1 = read_u32();
      CardCode code2 = read_u32();
      uint32_t loc2 = read_u32();
      Card cards[2];
      cards[0] = msg_card(code1);
      cards[1] = msg_card(code2);
      cards[0].set_location(loc1);
      cards[1].set_location(loc2);

      for (PlayerId pl = 0; pl < 2; pl++) {
        for (int i = 0; i < 2; i++) {
          auto c = cards[i];
          auto spec = c.get_spec(pl);
          auto plname = players_[1 - c.controler_]->nickname_;
          players_[pl]->notify("Card " + c.name_ + " swapped control towards " +
                               plname + " and is now located at " + spec + ".");
        }
      }
    } else if (msg_ == MSG_SET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      uint32_t location = read_u32();
      Card card = msg_card(code);
      card.set_location(location);
      auto c = card.controler_;
      auto& cpl = players_[c];
      auto& opl = players_[1 - c];
      cpl->notify(fmt::format("You set {} ({}) in {} position.", card.name_,
                              card.get_spec(c), card.get_position()));
      opl->notify(fmt::format("{} sets {} in {} position.", cpl->nickname_,
                              card.get_spec(PlayerId(1 - c)),
                              card.get_position()));
    } else if (msg_ == MSG_EQUIP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto c = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto pos = read_u8();
      Card card = get_card(c, loc, seq);
      c = read_u8();
      loc = read_u8();
      seq = read_u8();
      pos = read_u8();
      Card target = get_card(c, loc, seq);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto c = cardlist_info_for_player(card, pl);
        auto t = cardlist_info_for_player(target, pl);
        players_[pl]->notify(fmt::format("{} equipped to {}.", c, t));
      }
    } else if (msg_ == MSG_HINT) {
      auto hint_type = read_u8();
      auto player = read_u8();
      auto value = read_u32();

      if (hint_type == HINT_SELECTMSG && value == 501) {
        discard_hand_ = true;
      }
      // non-GUI don't need hint
      return;
      if (hint_type == HINT_SELECTMSG) {
        if (value > 2000) {
          CardCode code = value;
          players_[player]->notify(fmt::format("{} select {}",
                                               players_[player]->nickname_,
                                               c_get_card(code).name_));
        } else {
          players_[player]->notify(get_system_string(value));
        }
      } else if (hint_type == HINT_NUMBER) {
        players_[1 - player]->notify(
            fmt::format("Choice of player: {}", value));
      } else {
        fmt::println("Unknown hint type {} with value {}", hint_type, value);
      }
    } else if (msg_ == MSG_CARD_HINT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      uint8_t player = read_u8();
      uint8_t loc = read_u8();
      uint8_t seq = read_u8();
      uint8_t pos = read_u8();
      uint8_t type = read_u8();
      uint32_t value = read_u32();
      if (type == CHINT_RACE) {
        Card card = get_card(player, loc, seq);
        if (card.code_ == 0) {
          return;
        }
        std::string races_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           races_str));
        }
      } else if (type == CHINT_ATTRIBUTE) {
        Card card = get_card(player, loc, seq);
        if (card.code_ == 0) {
          return;
        }
        std::string attributes_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           attributes_str));
        }
      } else {
        fmt::println("Unknown card hint type {} with value {}", type, value);
      }
    } else if (msg_ == MSG_POS_CHANGE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = msg_card(code);
      card.set_location(read_u32());
      uint8_t prevpos = card.position_;
      card.position_ = read_u8();

      auto& pl = players_[card.controler_];
      auto& op = players_[1 - card.controler_];
      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto prevpos_str = position_to_string(prevpos);
      auto pos_str = position_to_string(card.position_);
      pl->notify("The position of card " + plspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
      op->notify("The position of card " + opspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
    } else if (msg_ == MSG_BECOME_TARGET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto u = read_u8();
      uint32_t target = read_u32();
      uint8_t tc = target & 0xff;
      uint8_t tl = (target >> 8) & 0xff;
      uint8_t tseq = (target >> 16) & 0xff;
      Card card = get_card(tc, tl, tseq);
      auto name = players_[chaining_player_]->nickname_;
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto tcname = card.name_;
        if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
          tcname = position_to_string(card.position_) + " card";
        }
        players_[pl]->notify(name + " targets " + spec + " (" + tcname + ")");
      }
    } else if (msg_ == MSG_CONFIRM_DECKTOP || msg_ == MSG_CONFIRM_EXTRATOP) {
      // an excavation: public to both players
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        const CardCode code = read_u32() & 0x7fffffff;
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        revealed_[{c, loc, seq}] = Reveal{3, code};
        if (verbose_) cards.push_back(get_card(c, loc, seq));
      }
      if (!verbose_) {
        return;
      }

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        if (pl == player) {
          p->notify(fmt::format("You reveal {} cards from your deck:", size));
        } else {
          p->notify(fmt::format("{} reveals {} cards from their deck:",
                                players_[player]->nickname_, size));
        }
        for (int i = 0; i < size; ++i) {
          p->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
        }
      }
    } else if (msg_ == MSG_RANDOM_SELECTED) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto count = read_u8();
      std::vector<Card> cards;

      for (int i = 0; i < count; ++i) {
        auto c = read_u8();
        auto loc = read_u8();
        if (loc & LOCATION_OVERLAY) {
          throw std::runtime_error("Overlay not supported for random selected");
        }
        auto seq = read_u8();
        auto pos = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        auto s = "card is";
        if (count > 1) {
          s = "cards are";
        }
        if (pl == player) {
          p->notify(fmt::format("Your {} {} randomly selected:", s, count));
        } else {
          p->notify(fmt::format("{}'s {} {} randomly selected:",
                                players_[player]->nickname_, s, count));
        }
        for (int i = 0; i < count; ++i) {
          p->notify(fmt::format("{}: {}", cards[i].get_spec(pl), cards[i].name_));
        }
      }
    } else if (msg_ == MSG_PLAYER_HINT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      dp_ += 6;
      // TODO(3): implement output
    } else if (msg_ == MSG_CANCEL_TARGET) {
      const int payload_size = dl_ - dp_;
      if (payload_size != 8) {
        throw std::runtime_error(fmt::format(
            "Invalid cancel_target payload: {}", payload_size));
      }
      dp_ = dl_;
    } else if (msg_ == MSG_CARD_TARGET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto c1 = read_u8();
      auto l1 = read_u8();
      auto s1 = read_u8();
      read_u8();
      auto c2 = read_u8();
      auto l2 = read_u8();
      auto s2 = read_u8();
      read_u8();

      Card card1 = get_card(c1, l1, s1);
      Card card2 = get_card(c2, l2, s2);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        auto spec1 = card1.get_spec(pl);
        auto spec2 = card2.get_spec(pl);
        auto c1name = card1.name_;
        auto c2name = card2.name_;
        if ((card1.controler_ != pl) && (card1.position_ & POS_FACEDOWN)) {
          c1name = position_to_string(card1.position_) + " card";
        }
        if ((card2.controler_ != pl) && (card2.position_ & POS_FACEDOWN)) {
          c2name = position_to_string(card2.position_) + " card";
        }
        p->notify(fmt::format(" {} ({}) targets {} ({})", spec1, c1name, spec2, c2name));
      }
    } else if (msg_ == MSG_CONFIRM_CARDS) {
      auto player = read_u8();
      dp_ += 1;  // maintained-core protocol (upstream #481): skip_panel
      auto size = read_u8();
      uint8_t confirm_mask = 0;
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        const CardCode code = read_u32() & 0x7fffffff;
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        if (verbose_) {
          cards.push_back(get_card(c, loc, seq));
        }
        // the host sends a confirm of deck cards only to the confirming player, any other confirm to both
        if (i == 0) confirm_mask = loc == LOCATION_DECK ? static_cast<uint8_t>(1 << player) : 3;
        Reveal &reveal = revealed_[{c, loc, seq}];
        reveal.viewers |= confirm_mask;
        reveal.code = code;
      }
      if (!verbose_) {
        return;
      }

      auto& pl = players_[player];
      auto& op = players_[1 - player];

      op->notify(fmt::format("{} shows you {} cards.", pl->nickname_, size));
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }
    } else if (msg_ == MSG_MISSED_EFFECT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      dp_ += 4;
      CardCode code = read_u32();
      Card card = msg_card(code);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto str = get_system_string(1622);
        std::string fmt_str = "[%ls]";
        str = str.replace(str.find(fmt_str), fmt_str.length(), card.name_);
        players_[pl]->notify(str);
      }
    } else if (msg_ == MSG_SORT_CARD) {
      // TODO(3): implement action
      if (!verbose_) {
        dp_ = dl_;
        resp_buf_[0] = 255;
        OCG_SetResponseb(pduel_, resp_buf_);
        return;
      }
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }
      auto& pl = players_[player];
      pl->notify(
          "Sort " + std::to_string(size) +
          " cards by entering numbers separated by spaces (c = cancel):");
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }

      fmt::println("sort card action not implemented");
      resp_buf_[0] = 255;
      OCG_SetResponseb(pduel_, resp_buf_);

      // // generate all permutations
      // std::vector<int> perm(size);
      // std::iota(perm.begin(), perm.end(), 0);
      // std::vector<std::vector<int>> perms;
      // do {
      //   auto option = std::accumulate(perm.begin(), perm.end(),
      //   std::string(),
      //                                 [&](std::string &acc, int i) {
      //                                   return acc + std::to_string(i + 1) +
      //                                   " ";
      //                                 });
      //   options_.push_back(option);
      // } while (std::next_permutation(perm.begin(), perm.end()));
      // options_.push_back("c");
      // callback_ = [this](int idx) {
      //   const auto &option = options_[idx];
      //   if (option == "c") {
      //     resp_buf_[0] = 255;
      //     OCG_SetResponseb(pduel_, resp_buf_);
      //     return;
      //   }
      //   std::istringstream iss(option);
      //   int x;
      //   int i = 0;
      //   while (iss >> x) {
      //     resp_buf_[i] = uint8_t(x);
      //     i++;
      //   }
      //   OCG_SetResponseb(pduel_, resp_buf_);
      // };
    } else if (msg_ == MSG_ADD_COUNTER) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      auto c = get_card(player, loc, seq);
      auto& pl = players_[player];
      PlayerId op_id = 1 - player;
      auto& op = players_[op_id];
      // TODO(3): counter type to string
      pl->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_REMOVE_COUNTER) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      auto c = get_card(player, loc, seq);
      auto& pl = players_[player];
      PlayerId op_id = 1 - player;
      auto& op = players_[op_id];
      pl->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_ATTACK_DISABLED) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      for (PlayerId pl = 0; pl < 2; pl++) {
        players_[pl]->notify(get_system_string(1621));
      }
    } else if (msg_ == MSG_SHUFFLE_SET_CARD) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      // TODO(3): implement output
      dp_ = dl_;
    } else if (msg_ == MSG_SWAP_GRAVE_DECK) {
      if (dp_ >= dl_) {
        throw std::runtime_error("Truncated swap_grave_deck payload");
      }
      const auto player = read_u8();
      if (player > 1) {
        throw std::runtime_error(fmt::format(
            "Invalid swap_grave_deck player: {}", player));
      }
      const auto deck_count =
          OCG_QueryFieldCount(pduel_, player, LOCATION_DECK);
      const auto grave_count =
          OCG_QueryFieldCount(pduel_, player, LOCATION_GRAVE);
      if (deck_count < 0 || deck_count > 255 ||
          grave_count < 0 || grave_count > 255) {
        throw std::runtime_error(fmt::format(
            "Invalid swap_grave_deck zone counts: player={}, deck={}, grave={}",
            player, deck_count, grave_count));
      }
      record_protocol_event(
          player, msg_,
          {static_cast<uint8_t>(deck_count),
           static_cast<uint8_t>(grave_count)});
      if (verbose_) {
        auto& pl = players_[player];
        auto& op = players_[1 - player];
        pl->notify(fmt::format(
            "Your deck and graveyard were swapped (deck {}, grave {}).",
            deck_count, grave_count));
        op->notify(fmt::format(
            "{} swapped their deck and graveyard (deck {}, grave {}).",
            pl->nickname_, deck_count, grave_count));
      }
      return;
    } else if (msg_ == MSG_SHUFFLE_DECK) {
      if (dp_ >= dl_) {
        throw std::runtime_error("Truncated shuffle_deck payload");
      }
      const auto player = read_u8();
      if (player > 1) {
        throw std::runtime_error(fmt::format(
            "Invalid shuffle_deck player: {}", player));
      }
      if (!verbose_) {
        return;
      }
      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify("You shuffled your deck.");
      op->notify(pl->nickname_ + " shuffled their deck.");
    } else if (msg_ == MSG_SHUFFLE_EXTRA) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto count = read_u8();
      for (int i = 0; i < count; ++i) {
        read_u32();
      }
      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify(fmt::format("You shuffled your extra deck ({}).", count));
      op->notify(fmt::format("{} shuffled their extra deck ({}).", pl->nickname_, count));
    } else if (msg_ == MSG_SHUFFLE_HAND) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }

      auto player = read_u8();
      dp_ = dl_;

      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify("You shuffled your hand.");
      op->notify(pl->nickname_ + " shuffled their hand.");
    } else if (msg_ == MSG_SUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_SUMMONING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = msg_card(code);
      card.set_location(read_u32());
      const auto &nickname = players_[card.controler_]->nickname_;
      for (auto& pl : players_) {
        pl->notify(nickname + " summoning " + card.name_ + " (" +
                   std::to_string(card.attack_) + "/" +
                   std::to_string(card.defense_) + ") in " +
                   card.get_position() + " position.");
      }
    } else if (msg_ == MSG_SPSUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_FLIPSUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_FLIPSUMMONING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }

      auto code = read_u32();
      auto location = read_u32();
      Card card = msg_card(code);
      card.set_location(location);

      auto& cpl = players_[card.controler_];
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        players_[1 - pl]->notify(cpl->nickname_ + " flip summons " + spec +
                                 " (" + card.name_ + ")");
      }
    } else if (msg_ == MSG_SPSUMMONING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = msg_card(code);
      card.set_location(read_u32());
      const auto &nickname = players_[card.controler_]->nickname_;
      for (PlayerId p = 0; p < 2; p++) {
        auto& pl = players_[p];
        auto pos = card.get_position();
        auto atk = std::to_string(card.attack_);
        auto def = std::to_string(card.defense_);
        std::string name = p == card.controler_ ? "You" : nickname;
        if (card.type_ & TYPE_LINK) {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + ") in " + pos + " position.");
        } else {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + "/" + def + ") in " + pos + " position.");
        }
      }
    } else if (msg_ == MSG_CHAIN_NEGATED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_DISABLED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_SOLVED) {
      dp_ = dl_;
      revealed_.clear();
    } else if (msg_ == MSG_CHAIN_SOLVING) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_END) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = msg_card(code);
      card.set_location(read_u32());
      auto tc = read_u8();
      auto tl = read_u8();
      auto ts = read_u8();
      uint32_t desc = read_u32();
      auto cs = read_u8();
      auto c = card.controler_;
      PlayerId o = 1 - c;
      chaining_player_ = c;
      players_[c]->notify("Activating " + card.get_spec(c) + " (" + card.name_ +
                          ")");
      players_[o]->notify(players_[c]->nickname_ + " activating " +
                          card.get_spec(o) + " (" + card.name_ + ")");
    } else if (msg_ == MSG_DAMAGE) {
      auto player = read_u8();
      auto amount = read_u32();
      _damage(player, amount);
    } else if (msg_ == MSG_RECOVER) {
      auto player = read_u8();
      auto amount = read_u32();
      _recover(player, amount);
    } else if (msg_ == MSG_LPUPDATE) {
      auto player = read_u8();
      auto lp = read_u32();
      if (lp >= lp_[player]) {
        _recover(player, lp - lp_[player]);
      } else {
        _damage(player, lp_[player] - lp);
      }
    } else if (msg_ == MSG_PAY_LPCOST) {
      auto player = read_u8();
      auto cost = read_u32();
      lp_[player] -= cost;
      if (!verbose_) {
        return;
      }
      auto& pl = players_[player];
      pl->notify("You pay " + std::to_string(cost) + " LP. Your LP is now " +
                 std::to_string(lp_[player]) + ".");
      players_[1 - player]->notify(
          pl->nickname_ + " pays " + std::to_string(cost) + " LP. " +
          pl->nickname_ + "'s LP is now " + std::to_string(lp_[player]) + ".");
    } else if (msg_ == MSG_ATTACK) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto attacker = read_u32();
      PlayerId ac = attacker & 0xff;
      auto aloc = (attacker >> 8) & 0xff;
      auto aseq = (attacker >> 16) & 0xff;
      auto apos = (attacker >> 24) & 0xff;
      auto target = read_u32();
      PlayerId tc = target & 0xff;
      auto tloc = (target >> 8) & 0xff;
      auto tseq = (target >> 16) & 0xff;
      auto tpos = (target >> 24) & 0xff;

      if ((ac == 0) && (aloc == 0) && (aseq == 0) && (apos == 0)) {
        return;
      }

      Card acard = get_card(ac, aloc, aseq);
      auto name = players_[ac]->nickname_;
      if ((tc == 0) && (tloc == 0) && (tseq == 0) && (tpos == 0)) {
        for (PlayerId i = 0; i < 2; i++) {
          players_[i]->notify(name + " prepares to attack with " +
                              acard.get_spec(i) + " (" + acard.name_ + ")");
        }
        return;
      }

      Card tcard = get_card(tc, tloc, tseq);
      for (PlayerId i = 0; i < 2; i++) {
        auto aspec = acard.get_spec(i);
        auto tspec = tcard.get_spec(i);
        auto tcname = tcard.name_;
        if ((tcard.controler_ != i) && (tcard.position_ & POS_FACEDOWN)) {
          tcname = tcard.get_position() + " card";
        }
        players_[i]->notify(name + " prepares to attack " + tspec + " (" +
                            tcname + ") with " + aspec + " (" + acard.name_ +
                            ")");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_START) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("begin damage");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_END) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("end damage");
      }
    } else if (msg_ == MSG_BATTLE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto attacker = read_u32();
      auto aa = read_u32();
      auto ad = read_u32();
      auto bd0 = read_u8();
      auto target = read_u32();
      auto da = read_u32();
      auto dd = read_u32();
      auto bd1 = read_u8();

      auto ac = attacker & 0xff;
      auto aloc = (attacker >> 8) & 0xff;
      auto aseq = (attacker >> 16) & 0xff;

      auto tc = target & 0xff;
      auto tloc = (target >> 8) & 0xff;
      auto tseq = (target >> 16) & 0xff;
      auto tpos = (target >> 24) & 0xff;

      Card acard = get_card(ac, aloc, aseq);
      Card tcard;
      if (tloc != 0) {
        tcard = get_card(tc, tloc, tseq);
      }
      for (int i = 0; i < 2; i++) {
        auto& pl = players_[i];
        std::string attacker_points;
        if (acard.type_ & TYPE_LINK) {
          attacker_points = std::to_string(aa);
        } else {
          attacker_points = std::to_string(aa) + "/" + std::to_string(ad);
        }
        if (tloc != 0) {
          std::string defender_points;
          if (tcard.type_ & TYPE_LINK) {
            defender_points = std::to_string(da);
          } else {
            defender_points = std::to_string(da) + "/" + std::to_string(dd);
          }
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks " +
                     tcard.name_ + " (" + defender_points + ")");
        } else {
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks");
        }
      }
    } else if (msg_ == MSG_WIN) {
      auto player = read_u8();
      auto reason = read_u8();
      auto& winner = players_[player];
      auto& loser = players_[1 - player];

      _duel_end(player, reason);

      auto l_reason = reason_to_string(reason);
      if (verbose_) {
        winner->notify("You won (" + l_reason + ").");
        loser->notify("You lost (" + l_reason + ").");
      }
    } else if (msg_ == MSG_RETRY) {
      throw std::runtime_error("Retry");
    } else if (msg_ == MSG_SELECT_BATTLECMD) {
      auto player = read_u8();
      auto activatable = read_cardlist_spec(player, true);
      auto attackable = read_cardlist_spec(player, true, true);
      bool to_m2 = read_u8();
      bool to_ep = read_u8();

      auto& pl = players_[player];
      if (verbose_) {
        pl->notify("Battle menu:");
      }
      int option = 0;
      for (const auto [code_t, spec, desc] : activatable) {
        CardCode code = code_t;
        if(code & 0x80000000) {
          code &= 0x7fffffff;
        }
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        la.option_ = option++;
        la.option_code_ = code;
        la.option_desc_ = desc;
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          std::string s = fmt::format(
            "{}: activate {}({}) [{}/{}] ({})",
            cmd_idx, c.name_, spec, c.attack_, c.defense_, c.get_effect_description(code_d, eff_idx));
        }
      }
      for (const auto [code, spec, data] : attackable) {
        bool direct_attackable = data & 0x1;
        auto act = direct_attackable ? ActionAct::DirectAttack : ActionAct::Attack;

        legal_actions_.push_back(
          LegalAction::act_spec(act, spec));
        if (verbose_) {
          auto [controller, loc, seq, pos] = spec_to_ls(player, spec);
          auto c = get_card(controller, loc, seq);
          int cmd_idx = legal_actions_.size();
          auto attack_str = direct_attackable ? "direct attack" : "attack";
          std::string s = fmt::format(
            "{}: {} {}({}) ", cmd_idx, attack_str, c.name_, spec);
          if (c.type_ & TYPE_LINK) {
            s += fmt::format("[{}]", c.attack_);
          } else {
            s += fmt::format("[{}/{}]", c.attack_, c.defense_);
          }
          pl->notify(s);
        }
      }
      if (to_m2) {
        legal_actions_.push_back(
          LegalAction::phase(ActionPhase::Main2));
        int cmd_idx = legal_actions_.size();
        if (verbose_) {
          pl->notify(fmt::format("{}: Main phase 2.", cmd_idx));
        }
      }
      if (to_ep) {
        if (!to_m2) {
          legal_actions_.push_back(
            LegalAction::phase(ActionPhase::End));
          int cmd_idx = legal_actions_.size();
          if (verbose_) {
            pl->notify(fmt::format("{}: End phase.", cmd_idx));
          }
        }
      }
      int n_activatables = activatable.size();
      int n_attackables = attackable.size();
      to_play_ = player;
      callback_ = [this, n_activatables, n_attackables, to_ep, to_m2](int idx) {
        const auto &la = legal_actions_[idx];
        if (idx < n_activatables) {
          OCG_SetResponsei(pduel_, idx << 16);
        } else if (idx < (n_activatables + n_attackables)) {
          idx = idx - n_activatables;
          OCG_SetResponsei(pduel_, (idx << 16) + 1);
        } else if ((la.phase_ == ActionPhase::End) && to_ep) {
          OCG_SetResponsei(pduel_, 3);
        } else if ((la.phase_ == ActionPhase::Main2) && to_m2) {
          OCG_SetResponsei(pduel_, 2);
        } else {
          throw std::runtime_error("Invalid option");
        }
      };
    } else if (msg_ == MSG_SELECT_UNSELECT_CARD) {
      // TODO: add feature of selected cards (also for multi select)
      auto player = read_u8();
      bool finishable = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto select_size = read_u8();

      std::vector<std::string> select_specs;
      select_specs.reserve(select_size);
      std::vector<Card> select_cards;
      select_cards.reserve(select_size);
      for (int i = 0; i < select_size; ++i) {
        const auto code = read_u32();
        const auto loc = read_u32();
        Card card = msg_card(code);
        card.set_location(loc);
        select_cards.push_back(card);
      }
      bind_own_deck_selections(select_cards, player);
      for (const auto &card : select_cards) {
        select_specs.push_back(card.get_spec(player));
      }
      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards:");
        for (int i = 0; i < select_size; ++i) {
          const auto &card = select_cards[i];
          const auto &spec = select_specs[i];
          auto s = fmt::format("{}: {}({})", i + 1, card.name_, spec);
          pl->notify(s);
        }
      }

      auto unselect_size = read_u8();

      // unselect not allowed (no regrets)
      dp_ += 8 * unselect_size;

      for (int j = 0; j < select_specs.size(); ++j) {
        legal_actions_.push_back(LegalAction::from_spec(select_specs[j]));
      }

      if (finishable) {
        legal_actions_.push_back(LegalAction::finish());
      }

      // cancelable and finishable not needed

      to_play_ = player;
      callback_ = [this](int idx) {
        if (legal_actions_[idx].finish_) {
          OCG_SetResponsei(pduel_, -1);
        } else {
          resp_buf_[0] = 1;
          resp_buf_[1] = idx;
          OCG_SetResponseb(pduel_, resp_buf_);
        }
      };

    } else if (msg_ == MSG_SELECT_CARD) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto size = read_u8();

      if (min == 0) {
        throw std::runtime_error("Min == 0 not implemented for select card");
      }

      std::vector<std::string> specs;
      specs.reserve(size);
      std::vector<Card> cards;
      cards.reserve(size);
      for (int i = 0; i < size; ++i) {
        const auto code = read_u32();
        const auto loc = read_u32();
        Card card = msg_card(code);
        card.set_location(loc);
        cards.push_back(card);
      }
      bind_own_deck_selections(cards, player);
      for (const auto &card : cards) specs.push_back(card.get_spec(player));
      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards separated by spaces:");
        for (size_t index = 0; index < cards.size(); ++index) {
          const auto &card = cards[index];
          const auto &spec = specs[index];
          const auto i = index + 1;
          if (card.controler_ != player && card.position_ & POS_FACEDOWN) {
            pl->notify(
              fmt::format("{}: {} card ({})", i, card.get_position(), spec));
          } else {
            pl->notify(
              fmt::format("{}: {} ({})", i, card.name_, spec));
          }
        }
      }

      if (discard_hand_) {
        discard_hand_ = false;
        if (current_phase_ == PHASE_END) {
          // random discard
          std::vector<int> comb(size);
          std::iota(comb.begin(), comb.end(), 0);
          std::shuffle(comb.begin(), comb.end(), gen_);
          resp_buf_[0] = min;
          for (int i = 0; i < min; ++i) {
            resp_buf_[i + 1] = comb[i];
          }
          OCG_SetResponseb(pduel_, resp_buf_);
          return;
        }
      }

      // TODO(1): use this when added to history actions
      // if ((min == max) && (max == specs.size())) {
      //   resp_buf_[0] = specs.size();
      //   for (int i = 0; i < specs.size(); ++i) {
      //     resp_buf_[i + 1] = i;
      //   }
      //   OCG_SetResponseb(pduel_, resp_buf_);
      //   return;
      // }

      init_multi_select(min, max, 0, specs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select(idx, ms_max_ == 1);
      };
    } else if (msg_ == MSG_SELECT_TRIBUTE) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto size = read_u8();

      std::vector<int> release_params;
      release_params.reserve(size);
      std::vector<std::string> specs;
      specs.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto release_param = read_u8();
          Card card = get_card(controller, loc, seq);
          cards.push_back(card);
          release_params.push_back(release_param);
        }
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) +
                   " cards to tribute separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          pl->notify(
            fmt::format("{}: {} ({})", specs.size(), card.name_, spec));
        }
      } else {
        for (int i = 0; i < size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto release_param = read_u8();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          specs.push_back(spec);

          release_params.push_back(release_param);
        }
      }

      const auto combs = tribute_combinations(release_params, min, max);
      if (combs.empty()) {
        throw std::runtime_error(fmt::format(
            "No valid select tribute combination: min={}, max={}, "
            "release_params={}",
            min, max, release_params));
      }

      init_multi_select(min, max, 0, specs, 1, combs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };
    } else if (msg_ == MSG_SELECT_SUM) {
      auto mode = read_u8();
      auto player = read_u8();
      auto val = read_u32();
      int _min = read_u8();
      int _max = read_u8();
      auto must_select_size = read_u8();

      if (mode > 1) {
        throw std::runtime_error("mode: " + std::to_string(mode) +
                                 " not implemented for MSG_SELECT_SUM");
      }
      if (must_select_size > 2) {
        throw std::runtime_error(
            " must select size: " + std::to_string(must_select_size) +
            " not implemented for MSG_SELECT_SUM");
      }

      std::vector<uint32_t> must_select_params;
      std::vector<uint32_t> select_params;
      std::vector<std::string> select_specs;
      must_select_params.reserve(must_select_size);

      if (verbose_) {
        std::vector<Card> must_select;
        must_select.reserve(must_select_size);
        for (int i = 0; i < must_select_size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();
          Card card = get_card(controller, loc, seq);
          must_select.push_back(card);
          must_select_params.push_back(param);
        }
        auto& pl = players_[player];
        pl->notify("Select cards for a target value of " +
                   std::to_string(val) + ", separated by spaces.");
        for (const auto &card : must_select) {
          auto spec = card.get_spec(player);
          pl->notify(card.name_ + " (" + spec +
                     ") must be selected, automatically selected.");
        }
      } else {
        for (int i = 0; i < must_select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          must_select_params.push_back(param);
        }
      }

      uint8_t select_size = read_u8();
      select_params.reserve(select_size);
      select_specs.reserve(select_size);

      if (verbose_) {
        std::vector<Card> select;
        select.reserve(select_size);
        for (int i = 0; i < select_size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();
          Card card = get_card(controller, loc, seq);
          select.push_back(card);
          select_params.push_back(param);
        }
        auto& pl = players_[player];
        for (const auto &card : select) {
          auto spec = card.get_spec(player);
          select_specs.push_back(spec);
          pl->notify(
            fmt::format("{}: {} ({})", select_specs.size(), card.name_, spec));
        }
      } else {
        for (int i = 0; i < select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          select_specs.push_back(spec);
          select_params.push_back(param);
        }
      }

      std::vector<std::vector<int>> combs =
          combinations_with_sum_params(
              must_select_params, select_params, val, _min, _max,
              mode == 1);
      if (combs.empty()) {
        throw std::runtime_error(fmt::format(
            "No valid MSG_SELECT_SUM combination: mode={}, target={}, "
            "min={}, max={}, must={}, candidates={}",
            mode, val, _min, _max, must_select_size, select_size));
      }
      if (combs.front().empty()) {
        resp_buf_[0] = must_select_size;
        for (int i = 0; i < must_select_size; ++i) {
          resp_buf_[i + 1] = 0;
        }
        OCG_SetResponseb(pduel_, resp_buf_);
        return;
      }

      int selection_min = _min;
      int selection_max = _max;
      if (mode == 1) {
        selection_min = select_size;
        selection_max = 0;
        for (const auto &comb : combs) {
          selection_min = std::min(
              selection_min, static_cast<int>(comb.size()));
          selection_max = std::max(
              selection_max, static_cast<int>(comb.size()));
        }
      }
      init_multi_select(
          selection_min, selection_max, must_select_size, select_specs,
          mode + 1, combs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };

    } else if (msg_ == MSG_SELECT_CHAIN) {
      auto player = read_u8();
      auto size = read_u8();
      auto spe_count = read_u8();
      // Maintained-core protocol (upstream #753): no header forced byte; each
      // entry carries its own forced flag after the EDESC flag, and cancelling
      // is refused while any entry is forced.
      bool forced = false;
      dp_ += 8;
      // auto hint_timing = read_u32();
      // auto other_timing = read_u32();

      std::vector<CardCode> codes;
      std::vector<uint32_t> descs;
      std::vector<std::string> specs;
      for (int i = 0; i < size; ++i) {
        auto flag = read_u8();
        forced = read_u8() != 0 || forced;
        CardCode code = read_u32();
        codes.push_back(code);
        PlayerId c = read_u8();
        uint8_t loc = read_u8();
        uint8_t seq = read_u8();
        uint8_t pos = read_u8();
        specs.push_back(ls_to_spec(loc, seq, pos, c != player));
        uint32_t desc = read_u32();
        descs.push_back(desc);
      }

      if ((size == 0) && (spe_count == 0)) {
        // non-GUI don't need this
        // if (verbose_) {
        //   fmt::println("keep processing");
        // }
        OCG_SetResponsei(pduel_, -1);
        return;
      }

      auto& pl = players_[player];
      auto& op = players_[1 - player];
      chaining_player_ = player;
      if (!op->seen_waiting_) {
        if (verbose_) {
          op->notify("Waiting for opponent.");
        }
        op->seen_waiting_ = true;
      }

      if (verbose_) {
        pl->notify("Select chain:");
      }

      for (int i = 0; i < size; i++) {
        CardCode code = codes[i];
        uint32_t desc = descs[i];
        auto spec = specs[i];
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        la.option_ = i;
        la.option_code_ = code;
        la.option_desc_ = desc;
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          std::string s = fmt::format(
            "{}: {}({}) ({})",
            i + 1, c.name_, spec, c.get_effect_description(code_d, eff_idx));
          pl->notify(s);
        }
      }

      if (!forced) {
        legal_actions_.push_back(LegalAction::cancel());
        if (verbose_) {
          pl->notify(fmt::format("{}: cancel", size + 1));
        }
      }
      to_play_ = player;
      callback_ = [this, forced](int idx) {
        const auto &action = legal_actions_[idx];
        if (action.act_ == ActionAct::Cancel) {
          if (forced) {
            fmt::print("cancel not allowed in forced chain\n");
            OCG_SetResponsei(pduel_, 0);
            return;
          }
          OCG_SetResponsei(pduel_, -1);
          return;
        }
        OCG_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_SELECT_YESNO) {
      auto player = read_u8();
      auto desc = read_u32();
      auto [code, eff_idx] = unpack_desc(0, desc);
      if (desc == 0) {
        show_buffer();
        auto s = fmt::format("Unknown desc {} in select_yesno", desc);
        throw std::runtime_error(s);
      }
      auto la = LegalAction::activate_spec(eff_idx, "");
      if (code != 0) {
        la.cid_ = c_get_card_id(code);
      }
      legal_actions_.push_back(la);
      if (verbose_) {
        auto& pl = players_[player];
        std::string s;
        if (code == 0) {
          s = get_system_string(eff_idx);
        } else {
          Card c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          eff_idx -= CARD_EFFECT_OFFSET;
          if (eff_idx >= c.strings_.size()) {
            throw std::runtime_error(
              fmt::format("Unknown effect {} of {}", eff_idx, c.name_));
          }
          auto str = c.strings_[eff_idx];
          if (str.empty()) {
            str = "effect " + std::to_string(eff_idx);
          }
          s = fmt::format("{} ({})", c.name_, str);
        }
        pl->notify("1: " + s);
        pl->notify("2: No");
      }
      // TODO: maybe add card id to cancel
      legal_actions_.push_back(LegalAction::cancel());
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx == 0) {
          OCG_SetResponsei(pduel_, 1);
        } else if (idx == 1) {
          OCG_SetResponsei(pduel_, 0);
        }
      };
    } else if (msg_ == MSG_SELECT_EFFECTYN) {
      auto player = read_u8();

      CardCode code = read_u32();
      auto ct = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto pos = read_u8();
      auto desc = read_u32();
      std::string spec = ls_to_spec(loc, seq, pos, ct != player);
      auto [code_d, eff_idx] = unpack_desc(code, desc);
      if (desc == 0) {
        code_d = code;
      }
      auto la = LegalAction::activate_spec(eff_idx, spec);
      if (code_d != 0) {
        la.cid_ = c_get_card_id(code_d);
      }
      // yes activates the single trigger the core asks about (or answers a script's yes/no: no activation)
      la.option_ = 0;
      la.option_code_ = code;
      la.option_desc_ = desc;
      legal_actions_.push_back(la);

      if (verbose_) {
        Card c = c_get_card(code);
        auto& pl = players_[player];
        auto name = c.name_;
        std::string s;
        if (code_d == 0) {
          s = get_system_string(desc);
          std::string fmt_str = "[%ls]";
          auto pos = find_substrs(s, fmt_str);
          if (pos.size() == 0) {
            // nothing to replace
          } else if (pos.size() == 1) {
            auto p = pos[0];
            s = s.substr(0, p) + name + s.substr(p + fmt_str.size());
          } else if (pos.size() == 2) {
            auto p1 = pos[0];
            auto p2 = pos[1];
            s = s.substr(0, p1) + spec +
                s.substr(p1 + fmt_str.size(), p2 - p1 - fmt_str.size()) + name +
                s.substr(p2 + fmt_str.size());
          } else {
            throw std::runtime_error("Unknown effectyn desc " +
                                     std::to_string(desc) + " of " + name);
          }
        } else {
          s = fmt::format(
            "{}({}) ({})", c.name_, spec, c.get_effect_description(code_d, eff_idx));
        }
        pl->notify("1: " + s);
        pl->notify("2: No");
      }

      // TODO: maybe add card info to cancel
      legal_actions_.push_back(LegalAction::cancel());
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx == 0) {
          OCG_SetResponsei(pduel_, 1);
        } else if (idx == 1) {
          OCG_SetResponsei(pduel_, 0);
        }
      };
    } else if (msg_ == MSG_SELECT_OPTION) {
      auto player = read_u8();
      auto size = read_u8();
      if (verbose_) {
        players_[player]->notify("Select an option:");
      }
      for (int i = 0; i < size; ++i) {
        auto desc = read_u32();
        auto [code, eff_idx] = unpack_desc(0, desc);
        // Description 0 is legitimate in one place: right after a summon,
        // special summon or set command the core asks which of the card's
        // procedures to use, listing each procedure effect's description, and
        // a procedure registered without SetDescription has 0 (for example
        // Scarred Nova Dragon - Burning Soul, 65541655: its own EFFECT_SPSUMMON_PROC
        // next to the Synchro procedure, 1164; operations.cpp special_summon_rule
        // and summon). The core lists such a card once per procedure in the idle
        // menu, but either row asks this question. Anywhere else a 0 is
        // unexplained and fatal.
        if (desc == 0 && !procedure_choice) {
          throw std::runtime_error(fmt::format("SELECT_OPTION offers description 0 (option {} of {}) outside a "
                                               "procedure choice", i, size));
        }
        auto la = LegalAction::activate_spec(eff_idx, "");
        if (code != 0) {
          la.cid_ = c_get_card_id(code);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          std::string s;
          if (code == 0) {
            s = get_system_string(eff_idx);
          } else {
            Card c = c_get_card(code);
            int cmd_idx = legal_actions_.size();
            eff_idx -= CARD_EFFECT_OFFSET;
            if (eff_idx >= c.strings_.size()) {
              throw std::runtime_error(
                fmt::format("Unknown effect {} of {}", eff_idx, c.name_));
            }
            auto str = c.strings_[eff_idx];
            if (str.empty()) {
              str = "effect " + std::to_string(eff_idx);
            }
            s = fmt::format("{} ({})", c.name_, str);
          }
          players_[player]->notify(std::to_string(i + 1) + ": " + s);
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        OCG_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_SELECT_IDLECMD) {
      int32_t player = read_u8();
      auto summonable_ = read_cardlist_spec(player);
      auto spsummon_ = read_cardlist_spec(player);
      auto repos_ = read_cardlist_spec(player);
      auto idle_mset_ = read_cardlist_spec(player);
      auto idle_set_ = read_cardlist_spec(player);
      auto idle_activate_ = read_cardlist_spec(player, true);
      bool to_bp_ = read_u8();
      bool to_ep_ = read_u8();
      read_u8(); // can_shuffle

      int offset = 0;

      auto& pl = players_[player];
      if (verbose_) {
        pl->notify("Select a card and action to perform.");
      }
      for (const auto &[code, spec, data] : summonable_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Summon, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Summon {} in face-up attack position", cmd_idx, name));
        }
      }
      offset += summonable_.size();
      int spsummon_offset = offset;
      for (const auto &[code, spec, data] : spsummon_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::SpSummon, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Special summon {}", cmd_idx, name));
        }
      }
      offset += spsummon_.size();
      int repos_offset = offset;
      for (const auto &[code, spec, data] : repos_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Repo, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Change position of {}", cmd_idx, name));
        }
      }
      offset += repos_.size();
      int mset_offset = offset;
      for (const auto &[code, spec, data] : idle_mset_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::MSet, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Summon {} in face-down defense position", cmd_idx, name));
        }
      }
      offset += idle_mset_.size();
      int set_offset = offset;
      for (const auto &[code, spec, data] : idle_set_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Set, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Set {}", cmd_idx, name));
        }
      }
      offset += idle_set_.size();
      int activate_offset = offset;
      int option = 0;
      for (const auto &[code_t, spec, desc] : idle_activate_) {
        CardCode code = code_t;
        if(code & 0x80000000) {
          code &= 0x7fffffff;
        }
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        la.option_ = option++;
        la.option_code_ = code;
        la.option_desc_ = desc;
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          std::string s = fmt::format(
            "{}: Activate {}({}) ({})",
            cmd_idx, c.name_, spec, c.get_effect_description(code_d, eff_idx));
          pl->notify(s);
        }
      }

      if (to_bp_) {
        legal_actions_.push_back(LegalAction::phase(ActionPhase::Battle));
        if (verbose_) {
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format("{}: Enter the battle phase.", cmd_idx));
        }
      }
      if (to_ep_) {
        if (!to_bp_) {
          legal_actions_.push_back(LegalAction::phase(ActionPhase::End));
          if (verbose_) {
            int cmd_idx = legal_actions_.size();
            pl->notify(fmt::format("{}: End phase.", cmd_idx));
          }
        }
      }

      to_play_ = player;
      callback_ = [this, spsummon_offset, repos_offset, mset_offset, set_offset,
                   activate_offset](int idx) {
        const auto &action = legal_actions_[idx];
        if (action.phase_ == ActionPhase::Battle) {
          OCG_SetResponsei(pduel_, 6);
        } else if (action.phase_ == ActionPhase::End) {
          OCG_SetResponsei(pduel_, 7);
        } else {
          auto act = action.act_;
          procedure_choice_ = act == ActionAct::Summon || act == ActionAct::SpSummon || act == ActionAct::MSet;
          if (act == ActionAct::Summon) {
            uint32_t idx_ = idx;
            OCG_SetResponsei(pduel_, idx_ << 16);
          } else if (act == ActionAct::SpSummon) {
            uint32_t idx_ = idx - spsummon_offset;
            OCG_SetResponsei(pduel_, (idx_ << 16) + 1);
          } else if (act == ActionAct::Repo) {
            uint32_t idx_ = idx - repos_offset;
            OCG_SetResponsei(pduel_, (idx_ << 16) + 2);
          } else if (act == ActionAct::MSet) {
            uint32_t idx_ = idx - mset_offset;
            OCG_SetResponsei(pduel_, (idx_ << 16) + 3);
          } else if (act == ActionAct::Set) {
            uint32_t idx_ = idx - set_offset;
            OCG_SetResponsei(pduel_, (idx_ << 16) + 4);
          } else if (act == ActionAct::Activate) {
            uint32_t idx_ = idx - activate_offset;
            OCG_SetResponsei(pduel_, (idx_ << 16) + 5);
          }
        }
      };
    } else if (msg_ == MSG_SELECT_PLACE || msg_ == MSG_SELECT_DISFIELD) {
      auto player = read_u8();
      auto count = read_u8();
      if (count == 0) {
        count = 1;
      }
      if (count != 1) {
        auto s = fmt::format("Select place count {} not implemented for {}",
                              count, msg_ == MSG_SELECT_PLACE ? "place" : "disfield");
        throw std::runtime_error(s);
      }
      auto flag = read_u32();
      auto places = flag_to_usable_places(flag);
      if (verbose_) {
        auto place_s = msg_ == MSG_SELECT_PLACE ? "place" : "disfield";
        auto s = fmt::format("Select {} for card, one of:", place_s);
        players_[player]->notify(s);
      }
      for (int i = 0; i < places.size(); ++i) {
        auto action = LegalAction::place(places[i]);
        action.cid_ = placement_prompt_cid_;
        legal_actions_.push_back(action);
        if (verbose_) {
          auto s = fmt::format("{}: {}", i + 1, action_place_to_string(places[i]));
          players_[player]->notify(s);
        }
      }
      to_play_ = player;
      callback_ = [this, player](int idx) {
        auto place = legal_actions_[idx].place_;
        int i = static_cast<int>(place);
        uint8_t plr = player;
        uint8_t loc;
        uint8_t seq;
        if (
          i >= static_cast<int>(ActionPlace::MZone1) &&
          i <= static_cast<int>(ActionPlace::MZone7)) {
          loc = LOCATION_MZONE;
          seq = i - static_cast<int>(ActionPlace::MZone1);
        } else if (
          i >= static_cast<int>(ActionPlace::SZone1) &&
          i <= static_cast<int>(ActionPlace::SZone8)) {
          loc = LOCATION_SZONE;
          seq = i - static_cast<int>(ActionPlace::SZone1);
        } else if (
          i >= static_cast<int>(ActionPlace::OpMZone1) &&
          i <= static_cast<int>(ActionPlace::OpMZone7)) {
          plr = 1 - player;
          loc = LOCATION_MZONE;
          seq = i - static_cast<int>(ActionPlace::OpMZone1);
        } else if (
          i >= static_cast<int>(ActionPlace::OpSZone1) &&
          i <= static_cast<int>(ActionPlace::OpSZone8)) {
          plr = 1 - player;
          loc = LOCATION_SZONE;
          seq = i - static_cast<int>(ActionPlace::OpSZone1);
        }
        resp_buf_[0] = plr;
        resp_buf_[1] = loc;
        resp_buf_[2] = seq;
        OCG_SetResponseb(pduel_, resp_buf_);
      };
    } else if (msg_ == MSG_SELECT_COUNTER) {
      auto player = read_u8();
      auto counter_type = read_u16();
      int counter_count = read_u16();
      int count = read_u8();
      if (count > 2) {
        throw std::runtime_error("Select counter count " +
                                 std::to_string(count) + " not implemented");
      }
      auto& pl = players_[player];
      if (verbose_) {
        pl->notify(fmt::format("Type new {} for {} card(s), separated by spaces.", "UNKNOWN_COUNTER", count));
      }
      std::vector<int> counters;
      counters.reserve(count);
      for (int i = 0; i < count; ++i) {
        auto code = read_u32();
        auto controller = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        auto counter = read_u16();
        counters.push_back(counter & 0xffff);

        if (verbose_) {
          pl->notify(c_get_card(code).name_ + ": " + std::to_string(counter));
        }
        // auto spec = ls_to_spec(loc, seq, 0, controller != player);
        // options_.push_back(spec);
      }
      // TODO(2): implement action
      n_counters_ = count;
      uint16_t resp1 = static_cast<uint16_t>(std::min(counter_count, counters[0]));
      memcpy(resp_buf_, &resp1, 2);
      counter_count -= counters[0];
      if (count == 2) {
        uint16_t resp2 = 0;
        if (counter_count > 0) {
          resp2 = static_cast<uint16_t>(counter_count);
        }
        memcpy(resp_buf_ + 2, &resp2, 2);
      }
      OCG_SetResponseb(pduel_, resp_buf_);
    } else if (msg_ == MSG_ANNOUNCE_RACE) {
      auto player = read_u8();
      const int count = read_u8();
      const uint32_t available = read_u32();
      if (count != 1) {
        throw std::runtime_error(
            "Announce race count " + std::to_string(count) +
            " not implemented");
      }

      for (const auto &[race, name] : race2str) {
        if (race == RACE_NONE || (available & race) == 0) {
          continue;
        }
        LegalAction action = LegalAction::number(race_to_id(race));
        action.response_ = race;
        legal_actions_.push_back(action);
      }
      if (legal_actions_.empty()) {
        throw std::runtime_error(fmt::format(
            "Announce race has no legal choices for mask 0x{:08x}",
            available));
      }

      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select 1 race:");
        for (int i = 0; i < legal_actions_.size(); ++i) {
          const auto race = legal_actions_[i].response_;
          pl->notify(fmt::format(
              "{}: {}", i + 1, race2str.at(race)));
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        OCG_SetResponsei(pduel_, legal_actions_[idx].response_);
      };
    } else if (msg_ == MSG_ANNOUNCE_NUMBER) {
      auto player = read_u8();
      int count = read_u8();
      std::vector<uint32_t> numbers;
      for (int i = 0; i < count; ++i) {
        const uint32_t number = read_u32();
        numbers.push_back(number);
        LegalAction action = LegalAction::number(
            announce_number_to_id(number));
        action.response_ = number;
        legal_actions_.push_back(action);
      }
      if (verbose_) {
        auto& pl = players_[player];
        std::string str = "Select a number, one of:";
        pl->notify(str);
        for (int i = 0; i < count; ++i) {
          pl->notify(fmt::format("{}: {}", i + 1, numbers[i]));
        }
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        OCG_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_ANNOUNCE_ATTRIB) {
      auto player = read_u8();
      int count = read_u8();
      auto flag = read_u32();

      int n_attrs = 7;

      std::vector<uint8_t> attrs;
      for (int i = 0; i < n_attrs; i++) {
        if (flag & (1 << i)) {
          attrs.push_back(i + 1);
        }
      }
      // TODO(2): implement action
      if (count != 1) {
        throw std::runtime_error("Announce attrib count " +
                                 std::to_string(count) + " not implemented");
      }

      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(count) +
                   " attributes separated by spaces:");
        for (int i = 0; i < attrs.size(); i++) {
          pl->notify(fmt::format("{}: {}", i + 1, attribute_to_string(1 << (attrs[i] - 1))));
        }
      }

      // auto combs = combinations(attrs.size(), count);
      for (int i = 0; i < attrs.size(); i++) {
        legal_actions_.push_back(LegalAction::attribute(1 << (attrs[i] - 1)));
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        const auto &action = legal_actions_[idx];
        uint32_t resp = 0;
        resp |= action.attribute_;
        OCG_SetResponsei(pduel_, resp);
      };
    } else if (msg_ == MSG_ANNOUNCE_CARD) {
      auto player = read_u8();
      int count = read_u8();

      std::vector<uint32_t> opcodes;
      opcodes.reserve(count);
      for (int i = 0; i < count; i++) {
        opcodes.push_back(read_u32());
      }

      const auto codes = announce_candidates(player, opcodes);

      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select 1 card from the following cards:");
        for (int i = 0; i < codes.size(); i++) {
          pl->notify(fmt::format("{}: {}", i + 1, c_get_card(codes[i]).name_));
        }
      }

      for (auto code : codes) {
        LegalAction la;
        la.cid_ = c_get_card_id(code);
        la.response_ = code;
        legal_actions_.push_back(la);
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        const auto &action = legal_actions_[idx];
        uint32_t resp = action.response_;
        OCG_SetResponsei(pduel_, resp);
      };
    } else if (msg_ == MSG_SELECT_POSITION) {
      auto player = read_u8();
      auto code = read_u32();
      auto valid_pos = read_u8();
      CardId cid = c_get_card_id(code);

      if (verbose_) {
        auto& pl = players_[player];
        auto card = c_get_card(code);
        pl->notify("Select position for " + card.name_ + ":");
      }

      for (auto pos : {POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK,
                       POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE}) {
        if (valid_pos & pos) {
          LegalAction la;
          la.cid_ = cid;
          la.position_ = pos;
          legal_actions_.push_back(la);
          int cmd_idx = legal_actions_.size();
          if (verbose_) {
            auto& pl = players_[player];
            pl->notify(fmt::format("{}: {}", cmd_idx, position_to_string(pos)));
          }
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        uint8_t pos = legal_actions_[idx].position_;
        OCG_SetResponsei(pduel_, pos);
      };
    } else {
      show_deck(0);
      show_deck(1);
      show_buffer();
      throw std::runtime_error(
        fmt::format("Unknown message {}, length {}, dp {}",
        msg_to_string(msg_), dl_, dp_));
    }
  }

  void _damage(uint8_t player, uint32_t amount) {
    lp_[player] -= amount;
    if (verbose_) {
      auto& lp = players_[player];
      lp->notify(fmt::format("Your lp decreased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp decreased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _recover(uint8_t player, uint32_t amount) {
    lp_[player] += amount;
    if (verbose_) {
      auto& lp = players_[player];
      lp->notify(fmt::format("Your lp increased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp increased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _duel_end(uint8_t player, uint8_t reason) {
    winner_ = player;
    win_reason_ = reason;
    // a search duel keeps the ended core duel restorable; its owner ends it (search_api.h)
    if (!keep_core_after_end_) OCG_EndDuel(pduel_);

    duel_started_ = false;
  }

 public:
  // Every assignable member a decision point's continuation reads (illegal-activation frames here, search_api.h
  // snapshots with the scripted driver's members). C arrays and the history action buffers are copied separately.
  // A member missing from the list shows up as a difference in the restore-continuity test
  // (tests/test_search_api.py), which compares every observation key and the message stream.
#define DUEL_ENV_FIELDS(X)                                                                                       \
  X(main_deck0_) X(main_deck1_) X(extra_deck0_) X(extra_deck1_) X(play_mode_) X(ai_player_) X(done_)            \
  X(step_count_) X(duel_started_) X(eng_flag_) X(disabled_field_) X(winner_) X(win_reason_) X(tp_)              \
  X(current_phase_) X(turn_count_) X(msg_) X(legal_actions_) X(to_play_) X(callback_) X(dp_) X(dl_)             \
  X(chaining_player_) X(ha_p_1_) X(ha_p_2_) X(revealed_) X(public_events_) X(history_) X(obs_view_)            \
  X(repro_responses_) X(repro_actions_) X(message_starts_) X(duel_seed_) X(procedure_choice_)                  \
  X(announce_truncated_) X(announce_empty_union_) X(announce_fixed_) X(decisions_turn_) X(turn_decisions_)      \
  X(step_limited_) X(guards_) X(visible_) X(guard_prompt_) X(guard_info_) X(pending_source_spec_)              \
  X(pending_source_cid_) X(pending_source_effect_) X(placement_source_) \
  X(placement_hint_) X(placement_prompt_cid_) X(ms_idx_) X(ms_mode_)        \
  X(ms_min_) X(ms_max_) X(ms_must_) X(ms_specs_) X(ms_combs_) X(ms_spec2idx_) X(ms_r_idxs_) X(discard_hand_)    \
  X(n_counters_) X(gen_) X(duel_gen_) X(ret_reward_) X(ret_win_reason_) X(candidate_cache_) X(own_recipe_rows_)  \
  X(room_format_) X(room_era_) X(start_gen_) X(start_duel_gen_) X(stream_hash_) X(control_changed_) X(replaced_) \
  X(refreshed_mzone_) X(activation_frames_) X(illegal_held_) X(illegal_info_) X(illegal_log_) X(shortfall_seen_)
#define DUEL_ENV_ARRAYS(X) X(deck_name_) X(lp_) X(data_) X(resp_buf_)

  struct EnvState {
#define DUEL_ENV_DECLARE(m) decltype(DuelEnvImpl::m) m;
    DUEL_ENV_FIELDS(DUEL_ENV_DECLARE)
#undef DUEL_ENV_DECLARE
#define DUEL_ENV_DECLARE_ARRAY(m) std::remove_all_extents_t<decltype(DuelEnvImpl::m)> m[std::extent_v<decltype(DuelEnvImpl::m)>];
    DUEL_ENV_ARRAYS(DUEL_ENV_DECLARE_ARRAY)
#undef DUEL_ENV_DECLARE_ARRAY
    std::vector<uint8_t> history_actions_1, history_actions_2;
  };

  EnvState save_env() const {
    EnvState env;
#define DUEL_ENV_COPY_OUT(m) env.m = m;
    DUEL_ENV_FIELDS(DUEL_ENV_COPY_OUT)
#undef DUEL_ENV_COPY_OUT
#define DUEL_ENV_COPY_ARRAY_OUT(m) std::copy(std::begin(m), std::end(m), std::begin(env.m));
    DUEL_ENV_ARRAYS(DUEL_ENV_COPY_ARRAY_OUT)
#undef DUEL_ENV_COPY_ARRAY_OUT
    env.history_actions_1 = Bytes(history_actions_1_);
    env.history_actions_2 = Bytes(history_actions_2_);
    return env;
  }

  void load_env(const EnvState &env) {
#define DUEL_ENV_COPY_IN(m) m = env.m;
    DUEL_ENV_FIELDS(DUEL_ENV_COPY_IN)
#undef DUEL_ENV_COPY_IN
#define DUEL_ENV_COPY_ARRAY_IN(m) std::copy(std::begin(env.m), std::end(env.m), std::begin(m));
    DUEL_ENV_ARRAYS(DUEL_ENV_COPY_ARRAY_IN)
#undef DUEL_ENV_COPY_ARRAY_IN
    SetBytes(history_actions_1_, env.history_actions_1);
    SetBytes(history_actions_2_, env.history_actions_2);
  }

  static std::vector<uint8_t> Bytes(const TArray<uint8_t> &array) {
    const auto *data = static_cast<const uint8_t *>(array.Data());
    return std::vector<uint8_t>(data, data + array.size * array.element_size);
  }
  static void SetBytes(TArray<uint8_t> &array, const std::vector<uint8_t> &bytes) {
    if (bytes.size() != array.size * array.element_size) throw std::runtime_error("history action buffer size changed");
    std::memcpy(array.Data(), bytes.data(), bytes.size());
  }

 protected:
  // An illegal-activation frame (step): the duel at a prompt before a response -- the core arena copy and the env
  // state -- with the response's row, whether it was a decision and whose, the scripted response log's length and
  // the replay file's length.
  struct Frame {
    Frame(void *core_, EnvState env_, int row_, bool decision_, PlayerId player_, size_t responses_, long replay_)
        : core(core_), env(std::move(env_)), row(row_), decision(decision_), player(player_),
          response_log(responses_), replay_pos(replay_) {}
    ~Frame() { duel_snapshot_free(core); }
    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;
    void *core;
    EnvState env;
    int row;
    bool decision;
    PlayerId player;
    size_t response_log;
    long replay_pos;
  };

  std::shared_ptr<const Frame> take_frame(int row, bool decision) {
    void *core = duel_snapshot(pduel_);
    if (core == nullptr) throw std::runtime_error("duel_snapshot refused an illegal-activation frame");
    long replay_pos = -1;
    if (record_ && is_recording && fp_ != nullptr) {
      std::fflush(fp_);
      replay_pos = std::ftell(fp_);
    }
    return std::make_shared<const Frame>(core, save_env(), row, decision, to_play_,
                                         response_log_ ? response_log_->size() : 0, replay_pos);
  }

  void restore_frame(const Frame &frame) {
    const int32_t rc = duel_rollback(pduel_, frame.core);
    if (rc != 0) throw std::runtime_error("duel_rollback of an illegal-activation frame returned " + std::to_string(rc));
    load_env(frame.env);
    if (response_log_) response_log_->resize(frame.response_log);
    if (frame.replay_pos >= 0) {
      std::fflush(fp_);
      if (ftruncate(fileno(fp_), frame.replay_pos) != 0 || std::fseek(fp_, frame.replay_pos, SEEK_SET) != 0)
        throw std::runtime_error("illegal-activation frame: the replay file could not be cut back");
    }
  }
};

class DuelEnv : public Env<DuelEnvSpec> {
protected:
  const int max_episode_steps_;
  const int timeout_;

  int elapsed_step_;

  std::uniform_int_distribution<uint64_t> dist_int_;

  // The pool can't be in vector, so we create multiple pools manually
  BS::thread_pool pool0_;
  BS::thread_pool pool1_;
  BS::thread_pool pool2_;
  BS::thread_pool pool3_;
  BS::thread_pool pool4_;

  const int max_timeout_{5};
 
  // DuelEnvImpl env_impl0_;
  // DuelEnvImpl env_impl1_;
  // DuelEnvImpl env_impl2_;
  // DuelEnvImpl env_impl3_;
  // DuelEnvImpl env_impl4_;
  std::vector<DuelEnvImpl> env_impls_;

  bool done_{true};

public:
  DuelEnv(const Spec &spec, int env_id)
      : Env<DuelEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        timeout_(spec.config["timeout"_]),
        pool0_(1), pool1_(1), pool2_(1), pool3_(1), pool4_(1),
        dist_int_(0, 0xffffffff) {
    env_impls_.reserve(max_timeout_);
    env_impls_.emplace_back(spec, dist_int_(gen_));
  }

  bool IsDone() override { return done_; }

  // Exact resume of this environment (see DuelEnvImpl::export_state): the step counters, done flag and the
  // current game. Called only while no step is in flight.
  std::string ExportState() const {
    std::ostringstream out;
    out << elapsed_step_ << " " << ResumeStepCounter() << " " << (done_ ? 1 : 0) << "\n"
        << env_impls_.back().export_state();
    return out.str();
  }

  // The current game as a rollout root (DuelEnvImpl::root_record). Called only while no step is in flight.
  std::string RootRecord() const { return env_impls_.back().root_record(); }

  void ImportState(const std::string &text) {
    const auto newline = text.find('\n');
    if (newline == std::string::npos) throw std::runtime_error("malformed env state header");
    std::istringstream head(text.substr(0, newline));
    int done = 1, step_counter = -1;
    head >> elapsed_step_ >> step_counter >> done;
    if (!head) throw std::runtime_error("malformed env state header");
    RestoreStepCounter(step_counter);
    done_ = done != 0;
    env_impls_.back().import_state(text.substr(newline + 1));
  }

  BS::thread_pool& get_pool(int idx) {
    switch (idx) {
      case 0: return pool0_;
      case 1: return pool1_;
      case 2: return pool2_;
      case 3: return pool3_;
      case 4: return pool4_;
      default: throw std::runtime_error("Invalid pool index");
    }
  }

  void Reset() override {
    int idx = env_impls_.size() - 1;
    auto& pool = get_pool(idx);
    auto fut = pool.submit_task([this, idx]() {
      env_impls_[idx].reset();
    });
    if (fut.wait_for(std::chrono::seconds(timeout_)) != std::future_status::ready) {
      throw std::runtime_error("Reset timeout");
    }
    try {
      fut.get();
    } catch (const std::exception &error) {
      fmt::println(stderr, "ENV_FATAL_REPRO {}", env_impls_[idx].repro_json(error.what()));
      std::fflush(stderr);
      throw;
    }

    auto &env_impl = env_impls_[idx];
    elapsed_step_ = 0;
    done_ = false;
    State state = Allocate();
    env_impl.WriteState(state);
  }

  void Step(const Action &action) override {
    int idx = env_impls_.size() - 1;
    auto& pool = get_pool(idx);
    int action_idx = action["action"_];
    auto task = pool.submit_task([this, action_idx, idx]() {
      // Test timeout: random sleep with probability 0.01
      // if (dist_int_(gen_) % 10000 == 0) {
      //   fmt::println("Env {} sleep {}", env_id_, env_impls_.capacity());
      //   std::this_thread::sleep_for(std::chrono::seconds(5));
      //   fmt::println("Env {} after {}", env_id_, env_impls_.capacity());
      //   auto& env_impl = env_impls_[idx];
      //   env_impl.step(action_idx);
      //   std::this_thread::sleep_for(std::chrono::seconds(1));
      //   return;
      // }
      env_impls_[idx].step(action_idx);
    });
    if (task.wait_for(std::chrono::seconds(timeout_)) != std::future_status::ready) {
      // A step that does not finish is fatal (the fork replaced the env and
      // truncated the game). The worker is still inside the duel, so only
      // the seed and decks are reported.
      fmt::println(stderr, "ENV_FATAL_REPRO {{\"error\":\"step timeout after {} s\",\"seed\":{},\"env\":{}}}",
                   timeout_, env_impls_[idx].repro_seed(), env_id_);
      std::fflush(stderr);
      throw std::runtime_error("Step timeout");
    }
    try {
      task.get();
    } catch (const std::exception &error) {
      // An exception while stepping a duel is fatal (no truncation law): the
      // full repro record -- seed, decks as loaded, every response, the prompt
      // -- goes to stderr as one line, then the exception ends the process.
      fmt::println(stderr, "ENV_FATAL_REPRO {}", env_impls_[idx].repro_json(error.what()));
      std::fflush(stderr);
      throw;
    }
    auto& env_impl = env_impls_[idx];
    done_ = env_impl.done();
    State state = Allocate();
    env_impl.WriteState(state);
  }

};

using DuelEnvPool = AsyncEnvPool<DuelEnv>;

} // namespace duelenv

template <>
struct fmt::formatter<duelenv::LegalAction>: formatter<string_view> {

    // Format the LegalAction object
    template <typename FormatContext>
    auto format(const duelenv::LegalAction& action, FormatContext& ctx) const {
        std::stringstream ss;
        ss << "{";
        if (!action.spec_.empty()) {
          ss << "spec='" << action.spec_ << "', ";
        }
        if (action.cid_ != 0) {
          ss << "cid=" << action.cid_ << ", ";
        }
        if (action.act_ != duelenv::ActionAct::None) {
          ss << "act=" << duelenv::action_act_to_string(action.act_) << ", ";
        }
        if (action.phase_ != duelenv::ActionPhase::None) {
          ss << "phase=" << duelenv::action_phase_to_string(action.phase_) << ", ";
        }
        if (action.finish_) {
          ss << "finish=true, ";
        }
        if (action.position_ != 0) {
          ss << "position=" << duelenv::position_to_string(action.position_) << ", ";
        }
        if (action.effect_ != -1) {
          ss << "effect=" << action.effect_ << ", ";
        }
        if (action.number_ != 0) {
          ss << "number=" << int(action.number_) << ", ";
        }
        if (action.place_ != duelenv::ActionPlace::None) {
          ss << "place=" << duelenv::action_place_to_string(action.place_) << ", ";
        }
        if (action.attribute_ != 0) {
          ss << "attribute=" << duelenv::attribute_to_string(action.attribute_) << ", ";
        }
        std::string s = ss.str();
        if (s.back() == ' ') {
          s.pop_back();
          s.pop_back();
        }
        s.push_back('}');
        return format_to(ctx.out(), "{}", s);
    }
};

#endif // DUELPOOL_DUEL_DUEL_ENV_H_
