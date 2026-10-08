#include "messages.h"

#include <cstring>
#include <unordered_map>

#include "constants.h"

namespace mfenv {
namespace {

struct Counted {
  int head;
  std::vector<int> strides;
  int tail;
};

/* messages.py:52-109 */
const std::unordered_map<int, int>& Fixed() {
  static const std::unordered_map<int, int> table = {
      {MSG_RETRY, 0},           {MSG_HINT, 6},
      {MSG_WIN, 2},             {MSG_SELECT_EFFECTYN, 13},
      {MSG_SELECT_YESNO, 5},    {MSG_SELECT_PLACE, 6},
      {MSG_SELECT_DISFIELD, 6}, {MSG_SELECT_POSITION, 6},
      {MSG_SHUFFLE_DECK, 1},    {MSG_REFRESH_DECK, 1},
      {MSG_SWAP_GRAVE_DECK, 1}, {MSG_REVERSE_DECK, 0},
      {MSG_DECK_TOP, 6},        {MSG_NEW_TURN, 1},
      {MSG_NEW_PHASE, 2},       {MSG_MOVE, 16},
      {MSG_POS_CHANGE, 9},      {MSG_SET, 8},
      {MSG_SWAP, 16},           {MSG_FIELD_DISABLED, 4},
      {MSG_SUMMONING, 8},       {MSG_SUMMONED, 0},
      {MSG_SPSUMMONING, 8},     {MSG_SPSUMMONED, 0},
      {MSG_FLIPSUMMONING, 8},   {MSG_FLIPSUMMONED, 0},
      {MSG_CHAINING, 16},       {MSG_CHAINED, 1},
      {MSG_CHAIN_SOLVING, 1},   {MSG_CHAIN_SOLVED, 1},
      {MSG_CHAIN_END, 0},       {MSG_CHAIN_NEGATED, 1},
      {MSG_CHAIN_DISABLED, 1},  {MSG_DAMAGE, 5},
      {MSG_RECOVER, 5},         {MSG_EQUIP, 8},
      {MSG_LPUPDATE, 5},        {MSG_UNEQUIP, 4},
      {MSG_CARD_TARGET, 8},     {MSG_CANCEL_TARGET, 8},
      {MSG_PAY_LPCOST, 5},      {MSG_ADD_COUNTER, 7},
      {MSG_REMOVE_COUNTER, 7},  {MSG_ATTACK, 8},
      {MSG_BATTLE, 26},         {MSG_ATTACK_DISABLED, 0},
      {MSG_DAMAGE_STEP_START, 0}, {MSG_DAMAGE_STEP_END, 0},
      {MSG_MISSED_EFFECT, 8},   {MSG_ROCK_PAPER_SCISSORS, 1},
      {MSG_HAND_RES, 1},        {MSG_ANNOUNCE_RACE, 6},
      {MSG_ANNOUNCE_ATTRIB, 6}, {MSG_CARD_HINT, 9},
      {MSG_PLAYER_HINT, 6},     {MSG_MATCH_KILL, 4},
  };
  return table;
}

/* messages.py:113-137 */
const std::unordered_map<int, Counted>& CountedTable() {
  static const std::unordered_map<int, Counted> table = {
      {MSG_SELECT_BATTLECMD, {1, {11, 8}, 2}},
      {MSG_SELECT_IDLECMD, {1, {7, 7, 7, 7, 7, 11}, 3}},
      {MSG_SELECT_OPTION, {1, {4}, 0}},
      {MSG_SELECT_CARD, {4, {8}, 0}},
      {MSG_SELECT_TRIBUTE, {4, {8}, 0}},
      {MSG_SELECT_UNSELECT_CARD, {5, {8, 8}, 0}},
      {MSG_SELECT_COUNTER, {5, {9}, 0}},
      {MSG_SORT_CARD, {1, {7}, 0}},
      {MSG_SORT_CHAIN, {1, {7}, 0}},
      {MSG_CONFIRM_DECKTOP, {1, {7}, 0}},
      {MSG_CONFIRM_EXTRATOP, {1, {7}, 0}},
      {MSG_CONFIRM_CARDS, {2, {7}, 0}},
      {MSG_SHUFFLE_HAND, {1, {4}, 0}},
      {MSG_SHUFFLE_EXTRA, {1, {4}, 0}},
      {MSG_SHUFFLE_SET_CARD, {1, {8}, 0}},
      {MSG_CARD_SELECTED, {1, {4}, 0}},
      {MSG_RANDOM_SELECTED, {1, {4}, 0}},
      {MSG_BECOME_TARGET, {0, {4}, 0}},
      {MSG_DRAW, {1, {4}, 0}},
      {MSG_TOSS_COIN, {1, {1}, 0}},
      {MSG_TOSS_DICE, {1, {1}, 0}},
      {MSG_ANNOUNCE_CARD, {1, {4}, 0}},
      {MSG_ANNOUNCE_NUMBER, {1, {4}, 0}},
  };
  return table;
}

uint8_t At(const uint8_t* buf, size_t len, size_t i) {
  if (i >= len)
    throw UnknownMessage("message buffer truncated while measuring a message");
  return buf[i];
}

/* messages.py:140-191 */
size_t LenSelectChain(const uint8_t* buf, size_t len, size_t pos) {
  size_t count = At(buf, len, pos + 1);
  return 2 + 9 + count * 14;
}

size_t LenSelectSum(const uint8_t* buf, size_t len, size_t pos) {
  size_t p = pos + 8;
  for (int i = 0; i < 2; ++i) {
    size_t count = At(buf, len, p);
    p += 1 + count * 11;
  }
  return p - pos;
}

size_t LenTagSwap(const uint8_t* buf, size_t len, size_t pos) {
  return static_cast<size_t>(At(buf, len, pos + 2)) * 4 +
         static_cast<size_t>(At(buf, len, pos + 4)) * 4 + 9;
}

size_t LenReloadField(const uint8_t* buf, size_t len, size_t pos) {
  size_t p = pos + 1;  /* duel rule */
  for (int side = 0; side < 2; ++side) {
    p += 4;  /* lp */
    for (int i = 0; i < 7; ++i) {  /* monster zones */
      if (At(buf, len, p)) p += 2;
      p += 1;
    }
    for (int i = 0; i < 8; ++i) {  /* spell/trap zones */
      if (At(buf, len, p)) p += 1;
      p += 1;
    }
    p += 6;  /* deck / hand / grave / removed / extra / extra-summonable */
  }
  size_t count = At(buf, len, p);
  p += 1 + count * 15;
  return p - pos;
}

size_t LenCString16(const uint8_t* buf, size_t len, size_t pos) {
  uint16_t n = 0;
  if (pos + 2 > len)
    throw UnknownMessage("message buffer truncated while measuring a message");
  std::memcpy(&n, buf + pos, 2);
  return 2 + static_cast<size_t>(n) + 1;
}

size_t PayloadLength(int msg, const uint8_t* buf, size_t len, size_t pos) {
  auto fixed = Fixed().find(msg);
  if (fixed != Fixed().end()) return static_cast<size_t>(fixed->second);
  auto counted = CountedTable().find(msg);
  if (counted != CountedTable().end()) {
    size_t p = pos + counted->second.head;
    for (int stride : counted->second.strides) {
      size_t count = At(buf, len, p);
      p += 1 + count * static_cast<size_t>(stride);
    }
    return p + counted->second.tail - pos;
  }
  switch (msg) {
    case MSG_SELECT_CHAIN: return LenSelectChain(buf, len, pos);
    case MSG_SELECT_SUM: return LenSelectSum(buf, len, pos);
    case MSG_TAG_SWAP: return LenTagSwap(buf, len, pos);
    case MSG_RELOAD_FIELD: return LenReloadField(buf, len, pos);
    case MSG_AI_NAME:
    case MSG_SHOW_HINT: return LenCString16(buf, len, pos);
    default: break;
  }
  throw UnknownMessage(std::string("no length rule for message ") +
                       MessageName(msg) + " (" + std::to_string(msg) + ")");
}

}  // namespace

std::vector<Message> SplitMessages(const uint8_t* buf, size_t len) {
  std::vector<Message> out;
  size_t pos = 0;
  while (pos < len) {
    int msg = buf[pos];
    ++pos;
    size_t length = PayloadLength(msg, buf, len, pos);
    if (pos + length > len)
      throw UnknownMessage(std::string(MessageName(msg)) + " wants " +
                           std::to_string(length) + " bytes at " +
                           std::to_string(pos) + ", buffer holds " +
                           std::to_string(len - pos));
    Message message;
    message.msg = msg;
    message.payload.assign(buf + pos, buf + pos + length);
    out.push_back(std::move(message));
    pos += length;
  }
  return out;
}

std::string MessageName(int msg) {
  switch (msg) {
    case MSG_RETRY: return "MSG_RETRY";
    case MSG_HINT: return "MSG_HINT";
    case MSG_WAITING: return "MSG_WAITING";
    case MSG_START: return "MSG_START";
    case MSG_WIN: return "MSG_WIN";
    case MSG_SELECT_BATTLECMD: return "MSG_SELECT_BATTLECMD";
    case MSG_SELECT_IDLECMD: return "MSG_SELECT_IDLECMD";
    case MSG_SELECT_EFFECTYN: return "MSG_SELECT_EFFECTYN";
    case MSG_SELECT_YESNO: return "MSG_SELECT_YESNO";
    case MSG_SELECT_OPTION: return "MSG_SELECT_OPTION";
    case MSG_SELECT_CARD: return "MSG_SELECT_CARD";
    case MSG_SELECT_CHAIN: return "MSG_SELECT_CHAIN";
    case MSG_SELECT_PLACE: return "MSG_SELECT_PLACE";
    case MSG_SELECT_POSITION: return "MSG_SELECT_POSITION";
    case MSG_SELECT_TRIBUTE: return "MSG_SELECT_TRIBUTE";
    case MSG_SORT_CHAIN: return "MSG_SORT_CHAIN";
    case MSG_SELECT_COUNTER: return "MSG_SELECT_COUNTER";
    case MSG_SELECT_SUM: return "MSG_SELECT_SUM";
    case MSG_SELECT_DISFIELD: return "MSG_SELECT_DISFIELD";
    case MSG_SORT_CARD: return "MSG_SORT_CARD";
    case MSG_SELECT_UNSELECT_CARD: return "MSG_SELECT_UNSELECT_CARD";
    case MSG_ROCK_PAPER_SCISSORS: return "MSG_ROCK_PAPER_SCISSORS";
    case MSG_ANNOUNCE_RACE: return "MSG_ANNOUNCE_RACE";
    case MSG_ANNOUNCE_ATTRIB: return "MSG_ANNOUNCE_ATTRIB";
    case MSG_ANNOUNCE_CARD: return "MSG_ANNOUNCE_CARD";
    case MSG_ANNOUNCE_NUMBER: return "MSG_ANNOUNCE_NUMBER";
    case MSG_NEW_TURN: return "MSG_NEW_TURN";
    case MSG_NEW_PHASE: return "MSG_NEW_PHASE";
    case MSG_MOVE: return "MSG_MOVE";
    case MSG_DRAW: return "MSG_DRAW";
    case MSG_CHAINING: return "MSG_CHAINING";
    default: return "MSG_" + std::to_string(msg);
  }
}

}  // namespace mfenv
