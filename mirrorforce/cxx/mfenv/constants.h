/* Core message ids, locations, positions and phases.
 *
 * Transcribed from mirrorforce/mirrorforce/netduel/constants.py (which is
 * itself a transcription of ygopro-core/common.h).  The C++ environment must
 * agree with the Python action layer on every one of these numbers, so they
 * live in one header and are checked against the Python module by
 * mirrorforce/tests/test_cxx_constants.py.
 */
#ifndef MFENV_CONSTANTS_H_
#define MFENV_CONSTANTS_H_

#include <cstdint>

namespace mfenv {

/* common.h processor flags (mirrorforce/puzzle/core.py:49-56) */
constexpr uint32_t kProcessorBufferLen = 0x0FFFFFFFu;
constexpr uint32_t kProcessorFlag = 0xF0000000u;
constexpr uint32_t kProcessorWaiting = 0x10000000u;
constexpr uint32_t kProcessorEnd = 0x20000000u;
constexpr int kSizeMessageBuffer = 0x2000;
constexpr int kSizeQueryBuffer = 0x4000;
constexpr int kSizeReturnValue = 256;
constexpr int kSizeSetcode = 16;

/* locations */
constexpr int LOCATION_DECK = 0x01;
constexpr int LOCATION_HAND = 0x02;
constexpr int LOCATION_MZONE = 0x04;
constexpr int LOCATION_SZONE = 0x08;
constexpr int LOCATION_GRAVE = 0x10;
constexpr int LOCATION_REMOVED = 0x20;
constexpr int LOCATION_EXTRA = 0x40;
constexpr int LOCATION_OVERLAY = 0x80;

/* positions */
constexpr int POS_FACEUP_ATTACK = 0x1;
constexpr int POS_FACEDOWN_ATTACK = 0x2;
constexpr int POS_FACEUP_DEFENSE = 0x4;
constexpr int POS_FACEDOWN_DEFENSE = 0x8;
constexpr int POS_FACEDOWN = 0xA;
constexpr int POS_FACEUP = 0x5;
/* OR-ed into QUERY_POSITION by EFFECT_REVEAL_ONFIELD: the card is face
 * down but its identity is public anyway (constants.py:33). */
constexpr int POS_REVEAL = 0x80;

/* card types the masking rules read */
constexpr uint32_t TYPE_MONSTER = 0x1;
constexpr uint32_t TYPE_PENDULUM = 0x1000000;
constexpr uint32_t TYPE_XYZ = 0x800000;
constexpr uint32_t TYPE_LINK = 0x4000000;

/* phases */
constexpr int PHASE_DRAW = 0x01;
constexpr int PHASE_STANDBY = 0x02;
constexpr int PHASE_MAIN1 = 0x04;
constexpr int PHASE_BATTLE_START = 0x08;
constexpr int PHASE_BATTLE_STEP = 0x10;
constexpr int PHASE_DAMAGE = 0x20;
constexpr int PHASE_DAMAGE_CAL = 0x40;
constexpr int PHASE_BATTLE = 0x80;
constexpr int PHASE_MAIN2 = 0x100;
constexpr int PHASE_END = 0x200;

/* hints */
constexpr int HINT_SELECTMSG = 3;

/* messages */
constexpr int MSG_RETRY = 1;
constexpr int MSG_HINT = 2;
constexpr int MSG_WAITING = 3;
constexpr int MSG_START = 4;
constexpr int MSG_WIN = 5;
constexpr int MSG_SELECT_BATTLECMD = 10;
constexpr int MSG_SELECT_IDLECMD = 11;
constexpr int MSG_SELECT_EFFECTYN = 12;
constexpr int MSG_SELECT_YESNO = 13;
constexpr int MSG_SELECT_OPTION = 14;
constexpr int MSG_SELECT_CARD = 15;
constexpr int MSG_SELECT_CHAIN = 16;
constexpr int MSG_SELECT_PLACE = 18;
constexpr int MSG_SELECT_POSITION = 19;
constexpr int MSG_SELECT_TRIBUTE = 20;
constexpr int MSG_SORT_CHAIN = 21;
constexpr int MSG_SELECT_COUNTER = 22;
constexpr int MSG_SELECT_SUM = 23;
constexpr int MSG_SELECT_DISFIELD = 24;
constexpr int MSG_SORT_CARD = 25;
constexpr int MSG_SELECT_UNSELECT_CARD = 26;
constexpr int MSG_CONFIRM_DECKTOP = 30;
constexpr int MSG_CONFIRM_CARDS = 31;
constexpr int MSG_SHUFFLE_DECK = 32;
constexpr int MSG_SHUFFLE_HAND = 33;
constexpr int MSG_REFRESH_DECK = 34;
constexpr int MSG_SWAP_GRAVE_DECK = 35;
constexpr int MSG_SHUFFLE_SET_CARD = 36;
constexpr int MSG_REVERSE_DECK = 37;
constexpr int MSG_DECK_TOP = 38;
constexpr int MSG_SHUFFLE_EXTRA = 39;
constexpr int MSG_NEW_TURN = 40;
constexpr int MSG_NEW_PHASE = 41;
constexpr int MSG_CONFIRM_EXTRATOP = 42;
constexpr int MSG_MOVE = 50;
constexpr int MSG_POS_CHANGE = 53;
constexpr int MSG_SET = 54;
constexpr int MSG_SWAP = 55;
constexpr int MSG_FIELD_DISABLED = 56;
constexpr int MSG_SUMMONING = 60;
constexpr int MSG_SUMMONED = 61;
constexpr int MSG_SPSUMMONING = 62;
constexpr int MSG_SPSUMMONED = 63;
constexpr int MSG_FLIPSUMMONING = 64;
constexpr int MSG_FLIPSUMMONED = 65;
constexpr int MSG_CHAINING = 70;
constexpr int MSG_CHAINED = 71;
constexpr int MSG_CHAIN_SOLVING = 72;
constexpr int MSG_CHAIN_SOLVED = 73;
constexpr int MSG_CHAIN_END = 74;
constexpr int MSG_CHAIN_NEGATED = 75;
constexpr int MSG_CHAIN_DISABLED = 76;
constexpr int MSG_CARD_SELECTED = 80;
constexpr int MSG_RANDOM_SELECTED = 81;
constexpr int MSG_BECOME_TARGET = 83;
constexpr int MSG_DRAW = 90;
constexpr int MSG_DAMAGE = 91;
constexpr int MSG_RECOVER = 92;
constexpr int MSG_EQUIP = 93;
constexpr int MSG_LPUPDATE = 94;
constexpr int MSG_UNEQUIP = 95;
constexpr int MSG_CARD_TARGET = 96;
constexpr int MSG_CANCEL_TARGET = 97;
constexpr int MSG_PAY_LPCOST = 100;
constexpr int MSG_ADD_COUNTER = 101;
constexpr int MSG_REMOVE_COUNTER = 102;
constexpr int MSG_ATTACK = 110;
constexpr int MSG_BATTLE = 111;
constexpr int MSG_ATTACK_DISABLED = 112;
constexpr int MSG_DAMAGE_STEP_START = 113;
constexpr int MSG_DAMAGE_STEP_END = 114;
constexpr int MSG_MISSED_EFFECT = 120;
constexpr int MSG_BE_CHAIN_TARGET = 121;
constexpr int MSG_CREATE_RELATION = 122;
constexpr int MSG_RELEASE_RELATION = 123;
constexpr int MSG_TOSS_COIN = 130;
constexpr int MSG_TOSS_DICE = 131;
constexpr int MSG_ROCK_PAPER_SCISSORS = 132;
constexpr int MSG_HAND_RES = 133;
constexpr int MSG_ANNOUNCE_RACE = 140;
constexpr int MSG_ANNOUNCE_ATTRIB = 141;
constexpr int MSG_ANNOUNCE_CARD = 142;
constexpr int MSG_ANNOUNCE_NUMBER = 143;
constexpr int MSG_CARD_HINT = 160;
constexpr int MSG_TAG_SWAP = 161;
constexpr int MSG_RELOAD_FIELD = 162;
constexpr int MSG_AI_NAME = 163;
constexpr int MSG_SHOW_HINT = 164;
constexpr int MSG_PLAYER_HINT = 165;
constexpr int MSG_MATCH_KILL = 170;
constexpr int MSG_CUSTOM_MSG = 180;

constexpr int RACES_COUNT = 26;

/* ygoenv builds duel options as (rules << 16); rules 5 is Master Rule 5
 * (worldmodel/engine.py:78). */
constexpr uint32_t MR5_DUEL_OPTIONS = 5u << 16;

}  // namespace mfenv

#endif  /* MFENV_CONSTANTS_H_ */
