#include "history.h"

#include <algorithm>
#include <cstring>

#include "constants.h"
#include "encoding.h"
#include "sha256.h"

namespace mfenv {
namespace {

uint32_t ReadU32(const std::vector<uint8_t>& b, size_t off) {
  uint32_t v = 0;
  std::memcpy(&v, b.data() + off, 4);
  return v;
}

uint16_t ReadU16(const std::vector<uint8_t>& b, size_t off) {
  uint16_t v = 0;
  std::memcpy(&v, b.data() + off, 2);
  return v;
}

/* state.py:687-698 */
int Relative(int player, int viewer) {
  if (player != 0 && player != 1) return 2;
  return player == viewer ? 0 : 1;
}

struct At {
  int controller = 2, location = 0, sequence = 0, position = 0;
};

/* winbot_trajectory.py:691-698 */
At UnpackAt(uint32_t value) {
  At at;
  at.controller = static_cast<int>(value & 0xFF);
  at.location = static_cast<int>((value >> 8) & 0xFF);
  at.sequence = static_cast<int>((value >> 16) & 0xFF);
  at.position = static_cast<int>((value >> 24) & 0xFF);
  return at;
}

/* state.py:1235-1247 -- the core's seven-byte code+location card records. */
void PublicCardRecords(const std::vector<uint8_t>& body, size_t head, int count,
                       std::vector<std::pair<uint32_t, uint32_t>>* out) {
  size_t off = head;
  for (int i = 0; i < count; ++i) {
    if (off + 7 > body.size()) break;
    const uint32_t code = ReadU32(body, off) & 0x7FFFFFFFu;
    const uint32_t at = static_cast<uint32_t>(body[off + 4]) |
                        (static_cast<uint32_t>(body[off + 5]) << 8) |
                        (static_cast<uint32_t>(body[off + 6]) << 16);
    out->emplace_back(code, at);
    off += 7;
  }
}

/* Python state.chain_link_after_message: negation is not a boundary; END is. */
bool UpdateChainLink(const Message& m, int* link, bool include_chaining = true) {
  if (m.msg == MSG_CHAIN_SOLVING && !m.payload.empty()) {
    *link = m.payload[0];
    return true;
  }
  if (include_chaining && m.msg == MSG_CHAINING && m.payload.size() >= 16) {
    *link = m.payload[15];
    return true;
  }
  if (m.msg == MSG_CHAIN_SOLVED || m.msg == MSG_CHAIN_END) {
    *link = 0;
    return true;
  }
  return false;
}

/* state.py:1249-1447 */
void ParsePublicEvents(const std::vector<Message>& messages,
                       std::vector<PublicEvent>* out) {
  int link = 0;
  for (size_t trace_index = 0; trace_index < messages.size(); ++trace_index) {
    const size_t event_start = out->size();
    const Message& m = messages[trace_index];
    const std::vector<uint8_t>& body = m.payload;
    const int msg = m.msg;

    if (UpdateChainLink(m, &link)) continue;
    if ((msg == MSG_CONFIRM_DECKTOP || msg == MSG_CONFIRM_EXTRATOP) &&
               body.size() >= 2) {
      std::vector<std::pair<uint32_t, uint32_t>> records;
      PublicCardRecords(body, 2, body[1], &records);
      for (const auto& record : records)
        out->push_back({PUB_CONFIRM, body[0], record.first, record.second, 0, 0,
                        0, link, 3, 0});
    } else if (msg == MSG_CONFIRM_CARDS && body.size() >= 3) {
      const int player = body[0];
      std::vector<std::pair<uint32_t, uint32_t>> records;
      PublicCardRecords(body, 3, body[2], &records);
      /* a deck confirmation reaches only the requesting seat; every other zone
       * is broadcast (state.py:1274-1280) */
      const bool in_deck =
          !records.empty() && ((records[0].second >> 8) & 0xFF) == LOCATION_DECK;
      const int audience = in_deck ? (1 << player) : 3;
      for (const auto& record : records)
        out->push_back({PUB_CONFIRM, player, record.first, record.second, 0, 0,
                        0, link, audience, 0});
    } else if (msg == MSG_RANDOM_SELECTED && body.size() >= 2) {
      const int player = body[0];
      /* single_duel.cpp sends it to players[player] and re-sends it to
       * players[1]: seat 0 never receives seat 1's selection (host_view.py
       * keeps the quirk); seat 1's second copy is a transport echo. */
      const int audience = player == 0 ? 3 : 2;
      for (int k = 0; k < body[1]; ++k) {
        const size_t off = 2 + 4 * static_cast<size_t>(k);
        if (off + 4 > body.size()) break;
        out->push_back({PUB_RANDOM, player, 0, ReadU32(body, off), 0, 0, 0, link,
                        audience, 0});
      }
    } else if ((msg == MSG_TOSS_COIN || msg == MSG_TOSS_DICE) && body.size() >= 2) {
      const int kind = msg == MSG_TOSS_COIN ? PUB_COIN : PUB_DICE;
      const int count = body[1];
      for (int k = 0; k < count && 2 + k < static_cast<int>(body.size()); ++k)
        out->push_back({kind, body[0], 0, 0, 0,
                        static_cast<uint32_t>(body[2 + k]), 0, link, 3, 0});
    } else if (msg == MSG_HINT && body.size() >= 6) {
      const int hint = body[0];
      const int player = body[1];
      const uint32_t value = ReadU32(body, 2);
      int kind = 0;
      switch (hint) {
        case 4: kind = PUB_OPSELECTED; break;        /* HINT_OPSELECTED */
        case 6: kind = PUB_ANNOUNCE_RACE; break;     /* HINT_RACE */
        case 7: kind = PUB_ANNOUNCE_ATTRIB; break;   /* HINT_ATTRIB */
        case 8: kind = PUB_ANNOUNCE_CODE; break;     /* HINT_CODE */
        case 9: kind = PUB_ANNOUNCE_NUMBER; break;   /* HINT_NUMBER */
        case 11: kind = PUB_ANNOUNCE_ZONE; break;    /* HINT_ZONE */
        case 10: kind = PUB_HINT_CARD; break;        /* HINT_CARD */
        default: kind = 0; break;
      }
      if (kind) {
        const uint32_t code =
            (kind == PUB_ANNOUNCE_CODE || kind == PUB_HINT_CARD) ? value : 0;
        /* these declarations are routed to the *other* seat only; HINT_CARD is
         * the one broadcast member, since the chooser already knows the rest */
        const int audience = hint == 10 ? 3 : (1 << (1 - player));
        out->push_back({kind, player, code, 0, 0, value, 0, link, audience, 0});
      }
    } else if (msg == MSG_SHUFFLE_DECK || msg == MSG_SHUFFLE_HAND ||
               msg == MSG_SHUFFLE_EXTRA || msg == MSG_SHUFFLE_SET_CARD) {
      int kind = PUB_SHUFFLE_DECK;
      if (msg == MSG_SHUFFLE_HAND) kind = PUB_SHUFFLE_HAND;
      else if (msg == MSG_SHUFFLE_EXTRA) kind = PUB_SHUFFLE_EXTRA;
      else if (msg == MSG_SHUFFLE_SET_CARD) kind = PUB_SHUFFLE_SET;
      const int player =
          (!body.empty() && msg != MSG_SHUFFLE_SET_CARD) ? body[0] : 2;
      out->push_back({kind, player, 0, 0, 0, 0, 0, link, 3, 0});
    } else if (msg == MSG_HAND_RES && !body.empty()) {
      out->push_back({PUB_HAND_RESULT, 2, 0, 0, 0, body[0], 0, link, 3, 0});
    } else if (msg == MSG_DECK_TOP && body.size() >= 6) {
      const int player = body[0];
      const int sequence = body[1];
      const uint32_t raw_code = ReadU32(body, 2);
      /* the high bit is the public/reversed marker; otherwise the broadcast
       * intentionally carries a hidden identity */
      const uint32_t code = (raw_code & 0x80000000u) ? (raw_code & 0x7FFFFFFFu) : 0;
      const uint32_t at = static_cast<uint32_t>(player) |
                          (static_cast<uint32_t>(LOCATION_DECK) << 8) |
                          (static_cast<uint32_t>(sequence) << 16);
      out->push_back({PUB_DECK_TOP, player, code, at, 0, 0, 0, link, 3, 0});
    } else if (msg == MSG_DRAW && body.size() >= 2) {
      const int player = body[0];
      for (int k = 0; k < body[1]; ++k) {
        const size_t off = 2 + 4 * static_cast<size_t>(k);
        if (off + 4 > body.size()) break;
        const uint32_t raw_code = ReadU32(body, off);
        /* the drawing seat gets the identity; the opponent only when the core
         * marked it public */
        const int audience = (raw_code & 0x80000000u) ? 3 : (1 << player);
        out->push_back({PUB_DRAW_REVEAL, player, raw_code & 0x7FFFFFFFu, 0, 0, 0,
                        0, link, audience, 0});
      }
    } else if ((msg == MSG_SUMMONING || msg == MSG_SPSUMMONING ||
                msg == MSG_FLIPSUMMONING) && body.size() >= 8) {
      const uint32_t code = ReadU32(body, 0);
      const uint32_t at = ReadU32(body, 4);
      int kind = PUB_SUMMONING;
      if (msg == MSG_SPSUMMONING) kind = PUB_SPSUMMONING;
      else if (msg == MSG_FLIPSUMMONING) kind = PUB_FLIPSUMMONING;
      out->push_back({kind, static_cast<int>(at & 0xFF), code & 0x7FFFFFFFu, at,
                      0, 0, 0, link, 3, 0});
    } else if (msg == MSG_SET && body.size() >= 8) {
      const uint32_t at = ReadU32(body, 4);
      /* setting is public, a face-down identity is not: code stays zero */
      out->push_back({PUB_SET, static_cast<int>(at & 0xFF), 0, at, 0, 0, 0, link,
                      3, 0});
    } else if (msg == MSG_NEW_TURN) {
      out->push_back({PUB_NEW_TURN, body.empty() ? 2 : body[0], 0, 0, 0, 0, 0, 0,
                      3, 0});
    } else if (msg == MSG_SUMMONED || msg == MSG_SPSUMMONED ||
               msg == MSG_FLIPSUMMONED || msg == MSG_DAMAGE_STEP_START) {
      int kind = PUB_SUMMONED;
      if (msg == MSG_SPSUMMONED) kind = PUB_SPSUMMONED;
      else if (msg == MSG_FLIPSUMMONED) kind = PUB_FLIPSUMMONED;
      else if (msg == MSG_DAMAGE_STEP_START) kind = PUB_DAMAGE_STEP_START;
      out->push_back({kind, 2, 0, 0, 0, 0, 0, link, 3, 0});
    } else if ((msg == MSG_EQUIP || msg == MSG_CARD_TARGET ||
                msg == MSG_CANCEL_TARGET) && body.size() >= 8) {
      int kind = PUB_EQUIP;
      if (msg == MSG_CARD_TARGET) kind = PUB_CARD_TARGET;
      else if (msg == MSG_CANCEL_TARGET) kind = PUB_CANCEL_TARGET;
      out->push_back({kind, 2, 0, ReadU32(body, 0), ReadU32(body, 4), 0, 0, link,
                      3, 0});
    } else if (msg == MSG_UNEQUIP && body.size() >= 4) {
      out->push_back({PUB_UNEQUIP, 2, 0, ReadU32(body, 0), 0, 0, 0, link, 3, 0});
    } else if (msg == MSG_FIELD_DISABLED && body.size() >= 4) {
      out->push_back({PUB_FIELD_DISABLED, 2, 0, 0, 0, ReadU32(body, 0), 0, link,
                      3, 0});
    } else if ((msg == MSG_ADD_COUNTER || msg == MSG_REMOVE_COUNTER) &&
               body.size() >= 7) {
      const uint32_t counter_type = ReadU16(body, 0);
      const uint32_t at = static_cast<uint32_t>(body[2]) |
                          (static_cast<uint32_t>(body[3]) << 8) |
                          (static_cast<uint32_t>(body[4]) << 16);
      const uint32_t count = ReadU16(body, 5);
      const int kind = msg == MSG_ADD_COUNTER ? PUB_ADD_COUNTER : PUB_REMOVE_COUNTER;
      out->push_back({kind, 2, 0, at, 0, count, counter_type, link, 3, 0});
    } else if (msg == MSG_CARD_HINT && body.size() >= 9) {
      const uint32_t at = ReadU32(body, 0);
      const int hint_type = body[4];
      const uint32_t value = ReadU32(body, 5);
      /* CHINT_CARD stores another card's code as its value */
      const uint32_t code = hint_type == 2 ? value : 0;
      out->push_back({PUB_CARD_HINT, 2, code, at, 0, value,
                      static_cast<uint32_t>(hint_type), link, 3, 0});
    } else if (msg == MSG_PLAYER_HINT && body.size() >= 6) {
      out->push_back({PUB_PLAYER_HINT, body[0], 0, 0, 0, ReadU32(body, 2),
                      static_cast<uint32_t>(body[1]), link, 3, 0});
    } else if (msg == MSG_SWAP_GRAVE_DECK && !body.empty()) {
      out->push_back({PUB_SWAP_GRAVE_DECK, body[0], 0, 0, 0, 0, 0, link, 3, 0});
    } else if (msg == MSG_REVERSE_DECK) {
      out->push_back({PUB_REVERSE_DECK, 2, 0, 0, 0, 0, 0, link, 3, 0});
    } else if (msg == MSG_ATTACK_DISABLED) {
      out->push_back({PUB_ATTACK_DISABLED, 2, 0, 0, 0, 0, 0, link, 3, 0});
    } else if (msg == MSG_MISSED_EFFECT && body.size() >= 8) {
      out->push_back({PUB_MISSED_EFFECT, 2, ReadU32(body, 4), ReadU32(body, 0), 0,
                      0, 0, link, 3, 0});
    }
    for (size_t i = event_start; i < out->size(); ++i)
      (*out)[i].trace_index = static_cast<int>(trace_index);
  }
}

/* state.py:1449-1523 */
void ParseMoves(const std::vector<Message>& messages, std::vector<Move>* moves,
                std::vector<PosChange>* changes, std::vector<Draw>* draws) {
  int link = 0;
  for (size_t trace_index = 0; trace_index < messages.size(); ++trace_index) {
    const Message& m = messages[trace_index];
    if (UpdateChainLink(m, &link, false)) continue;
    if (m.msg == MSG_MOVE && m.payload.size() >= 16) {
      const uint32_t code = ReadU32(m.payload, 0);
      const uint32_t prev = ReadU32(m.payload, 4);
      const uint32_t cur = ReadU32(m.payload, 8);
      const uint32_t reason = ReadU32(m.payload, 12);
      Move move;
      move.code = code;
      move.from_controller = static_cast<int>(prev & 0xFF);
      move.from_location = static_cast<int>((prev >> 8) & 0xFF);
      move.from_sequence = static_cast<int>((prev >> 16) & 0xFF);
      move.from_position = static_cast<int>((prev >> 24) & 0xFF);
      move.to_controller = static_cast<int>(cur & 0xFF);
      move.to_location = static_cast<int>((cur >> 8) & 0xFF);
      move.to_sequence = static_cast<int>((cur >> 16) & 0xFF);
      move.to_position = static_cast<int>((cur >> 24) & 0xFF);
      move.reason = reason;
      move.link = link;
      move.trace_index = static_cast<int>(trace_index);
      moves->push_back(move);
    } else if (m.msg == MSG_POS_CHANGE && m.payload.size() >= 9) {
      PosChange change;
      change.code = ReadU32(m.payload, 0);
      change.controller = m.payload[4];
      change.location = m.payload[5];
      change.sequence = m.payload[6];
      change.previous = m.payload[7];
      change.current = m.payload[8];
      change.link = link;
      change.trace_index = static_cast<int>(trace_index);
      changes->push_back(change);
    } else if (m.msg == MSG_DRAW && m.payload.size() >= 2) {
      Draw draw;
      draw.player = m.payload[0];
      const int count = m.payload[1];
      for (int k = 0; k < count; ++k) {
        const size_t off = 2 + 4 * static_cast<size_t>(k);
        if (off + 4 > m.payload.size()) break;
        /* the top bit marks a card drawn face down to the opponent's view */
        draw.codes.push_back(ReadU32(m.payload, off) & 0x7FFFFFFFu);
      }
      draw.link = link;
      draw.trace_index = static_cast<int>(trace_index);
      draws->push_back(draw);
    }
  }
}

/* state.py:1107-1135 */
void ParseChain(const std::vector<Message>& messages, std::vector<ChainEvent>* out) {
  for (size_t trace_index = 0; trace_index < messages.size(); ++trace_index) {
    const Message& m = messages[trace_index];
    if (m.msg == MSG_CHAINING && m.payload.size() >= 16) {
      ChainEvent event;
      event.kind = CHAIN_CHAINING;
      event.link = m.payload[15];
      event.code = ReadU32(m.payload, 0);
      event.at = ReadU32(m.payload, 4);
      event.player = m.payload[8];
      event.desc = ReadU32(m.payload, 11);
      event.trace_index = static_cast<int>(trace_index);
      out->push_back(event);
    } else if (!m.payload.empty() &&
               (m.msg == MSG_CHAIN_SOLVED || m.msg == MSG_CHAIN_NEGATED ||
                m.msg == MSG_CHAIN_DISABLED || m.msg == MSG_CHAIN_SOLVING)) {
      int kind = CHAIN_SOLVED;
      if (m.msg == MSG_CHAIN_NEGATED) kind = CHAIN_NEGATED;
      else if (m.msg == MSG_CHAIN_DISABLED) kind = CHAIN_DISABLED;
      else if (m.msg == MSG_CHAIN_SOLVING) kind = CHAIN_SOLVING;
      ChainEvent event;
      event.kind = kind;
      event.link = m.payload[0];
      event.trace_index = static_cast<int>(trace_index);
      out->push_back(event);
    }
  }
}

/* state.py:1146-1174 */
void ParseTargets(const std::vector<Message>& messages,
                  std::vector<ChainTarget>* out) {
  int link = 0;
  for (size_t trace_index = 0; trace_index < messages.size(); ++trace_index) {
    const Message& m = messages[trace_index];
    if (UpdateChainLink(m, &link)) continue;
    if (m.msg == MSG_BECOME_TARGET && !m.payload.empty()) {
      const int count = m.payload[0];
      for (int i = 0; i < count; ++i) {
        const size_t off = 1 + 4 * static_cast<size_t>(i);
        if (off + 4 > m.payload.size()) break;
        out->push_back({link, ReadU32(m.payload, off),
                        static_cast<int>(trace_index)});
      }
    }
  }
}

/* state.py:1191-1217 */
void ParseLpEvents(const std::vector<Message>& messages,
                   std::vector<LpEvent>* out) {
  int link = 0;
  bool in_damage_step = false;
  for (size_t trace_index = 0; trace_index < messages.size(); ++trace_index) {
    const Message& m = messages[trace_index];
    if (UpdateChainLink(m, &link)) continue;
    if (m.msg == MSG_DAMAGE_STEP_START) { in_damage_step = true; continue; }
    if (m.msg == MSG_DAMAGE_STEP_END) { in_damage_step = false; continue; }
    int kind = -1;
    if (m.msg == MSG_DAMAGE) kind = LP_DAMAGE;
    else if (m.msg == MSG_RECOVER) kind = LP_RECOVER;
    else if (m.msg == MSG_PAY_LPCOST) kind = LP_PAY_COST;
    else if (m.msg == MSG_LPUPDATE) kind = LP_UPDATE;
    if (kind >= 0 && m.payload.size() >= 5)
      out->push_back({kind, m.payload[0], ReadU32(m.payload, 1), link,
                      in_damage_step, static_cast<int>(trace_index)});
  }
}

/* state.py:1220-1232 */
void ParseAttacks(const std::vector<Message>& messages, std::vector<Attack>* out) {
  for (size_t trace_index = 0; trace_index < messages.size(); ++trace_index) {
    const Message& m = messages[trace_index];
    if (m.msg != MSG_ATTACK || m.payload.size() < 4) continue;
    const uint32_t attacker = ReadU32(m.payload, 0);
    const uint32_t target = m.payload.size() >= 8 ? ReadU32(m.payload, 4) : 0;
    out->push_back({attacker, target, static_cast<int>(trace_index)});
  }
}

/* winbot_trajectory.py:701-725 */
void FillSemanticFields(const SemanticTable& table, int64_t code,
                        int64_t effect_row, bool effect_exact,
                        HistoryTokenRecord* row) {
  const int64_t card_row =
      code ? table.CardRow(static_cast<uint32_t>(code)) : 0;
  row->semantic_visible = code != 0;
  row->card_code = row->semantic_visible ? code : 0;
  row->card_row = card_row;
  if (card_row) {
    row->card_effect_rows = table.EffectRowsForCard(static_cast<uint32_t>(code));
    row->effect_row = effect_row;
    row->effect_exact = effect_exact;
  } else {
    row->card_effect_rows.clear();
    row->effect_row = 0;
    row->effect_exact = false;
  }
}

int Clamp(int value, int low, int high) {
  return std::max(low, std::min(high, value));
}

struct RecordArgs {
  HistoryTokenKind kind = HistoryTokenKind::STATE_DELTA;
  HistorySubtype subtype = HistorySubtype::PUBLIC_RESULT;
  int turn = 0;
  int64_t trace = 0;
  int64_t code = 0;
  int64_t effect_row = 0;
  bool effect_exact = false;
  int player = 2;
  At from_at;
  At to_at;
  int phase = 0;
  int link = 0;
  int event_kind = 0;
  int64_t value = 0;
  int64_t detail = 0;
  int64_t reason = 0;
  int64_t amount = 0;
  int64_t count = 0;
  int64_t flags = 0;
  std::vector<std::pair<std::string, std::string>> payload;
};

/* winbot_trajectory.py:1004-1060 */
HistoryTokenRecord MakeRecord(int viewer, const RecordArgs& args,
                              const SemanticTable& table) {
  HistoryTokenRecord row;
  row.kind = static_cast<int>(args.kind);
  row.subtype = static_cast<int>(args.subtype);
  row.turn = args.turn;
  row.trace_index = args.trace;
  row.player_relative = Relative(args.player, viewer);
  row.from_controller_relative = Relative(args.from_at.controller, viewer);
  row.from_location = args.from_at.location;
  row.from_sequence = Clamp(args.from_at.sequence, 0, 255);
  row.to_controller_relative = Relative(args.to_at.controller, viewer);
  row.to_location = args.to_at.location;
  row.to_sequence = Clamp(args.to_at.sequence, 0, 255);
  row.phase = args.phase;
  row.link = Clamp(args.link, 0, 255);
  row.public_event_kind = Clamp(args.event_kind, 0, 63);
  row.value = args.value;
  row.detail = args.detail;
  row.reason = args.reason;
  row.amount = args.amount;
  row.count = args.count;
  row.from_position = args.from_at.position;
  row.to_position = args.to_at.position;
  row.flags = args.flags;
  row.payload = args.payload;
  FillSemanticFields(table, args.code, args.effect_row, args.effect_exact, &row);
  return row;
}

/* state.py:426-448 -- sanitize a god-view transition code exactly as
 * single_duel.cpp routes it.  Applying the same deterministic rule to the
 * offline payload is what makes the two history streams identical. */
/* state.py observer_event_code -- single_duel.cpp's MSG_SPSUMMONING copy
 * for the non-controller loses a face-down summon's code unless the core
 * marked it POS_REVEAL (host_view.py should_hide_facedown_code).  Normal and
 * flip summons are broadcast unrewritten and are never face-down. */
int64_t ObserverEventCode(int kind, int64_t payload_code, int viewer, uint32_t at) {
  const int64_t code = payload_code & 0x7FFFFFFF;
  if (!code || kind != PUB_SPSUMMONING) return code;
  const int controller = static_cast<int>(at & 0xFF);
  const int position = static_cast<int>((at >> 24) & 0xFF);
  if (controller == viewer) return code;
  if ((position & POS_FACEDOWN) != 0 && (position & 0x80) == 0) return 0;
  return code;
}

int64_t ObserverTransitionCode(bool is_move, int64_t payload_code, int viewer,
                               const At& to_at) {
  int64_t code = payload_code & 0x7FFFFFFF;
  if (!code) return 0;
  if (!is_move) return code;  /* MSG_POS_CHANGE is broadcast unrewritten */
  if (to_at.controller == viewer) return code;
  const bool hidden_facedown =
      (to_at.position & POS_FACEDOWN) != 0 && (to_at.position & 0x80) == 0;
  if (!(to_at.location & (LOCATION_GRAVE | LOCATION_OVERLAY)) &&
      ((to_at.location & (LOCATION_DECK | LOCATION_HAND)) || hidden_facedown))
    return 0;
  return code;
}

std::string Num(int64_t value) { return std::to_string(value); }

std::string AtList(const At& at) {
  return "[" + Num(at.controller) + "," + Num(at.location) + "," +
         Num(at.sequence) + "," + Num(at.position) + "]";
}

}  // namespace

IntervalEvents ParseInterval(const std::vector<Message>& messages) {
  IntervalEvents out;
  ParseMoves(messages, &out.moves, &out.pos_changes, &out.draws);
  ParseChain(messages, &out.chain);
  ParseTargets(messages, &out.targets);
  ParseAttacks(messages, &out.attacks);
  ParseLpEvents(messages, &out.lp_events);
  ParsePublicEvents(messages, &out.public_events);
  return out;
}

IntervalRecords RecordsByTrace(const IntervalEvents& events,
                               int64_t message_cursor, int current_turn,
                               const SemanticTable& table) {
  /* (viewer, local trace) -> (priority, insertion order, record).  Public
   * events take priority 0 and everything else 1, and the sort is stable, so
   * within one message the order is the order the parsers produced
   * (winbot_trajectory.py:1067-1068, :1302-1307). */
  struct Slot {
    int priority;
    int order;
    HistoryTokenRecord row;
  };
  std::map<int, std::map<int, std::vector<Slot>>> staged;
  int order = 0;
  auto add = [&staged, &order](int viewer, int trace, int priority,
                               HistoryTokenRecord row) {
    staged[viewer][trace].push_back(Slot{priority, order++, std::move(row)});
  };

  for (const PublicEvent& event : events.public_events) {
    if (event.kind == PUB_NEW_TURN) continue;  /* the boundary token covers it */
    for (int viewer = 0; viewer < 2; ++viewer) {
      if (!(event.audience & (1 << viewer))) continue;
      const int64_t code = ObserverEventCode(event.kind, event.code, viewer, event.at);
      RecordArgs args;
      args.kind = HistoryTokenKind::PUBLIC_EVENT;
      args.subtype = HistorySubtype::PUBLIC_RESULT;
      args.turn = current_turn;
      args.trace = message_cursor + event.trace_index;
      args.code = code;
      args.player = event.player;
      args.from_at = UnpackAt(event.at);
      args.to_at = UnpackAt(event.target);
      args.link = event.link;
      args.event_kind = event.kind;
      args.value = event.value;
      args.detail = event.detail;
      args.payload = {
          {"at", Num(event.at)},
          {"code", Num(code)},
          {"detail", Num(event.detail)},
          {"event_kind", Num(event.kind)},
          {"link", Num(event.link)},
          {"player", Num(event.player)},
          {"target", Num(event.target)},
          {"value", Num(event.value)},
      };
      add(viewer, event.trace_index, 0, MakeRecord(viewer, args, table));
    }
  }

  for (const Move& move : events.moves) {
    for (int viewer = 0; viewer < 2; ++viewer) {
      int to_position = move.to_position;
      if (move.to_location & 0x0C /* LOCATION_ONFIELD */) to_position &= ~0x80;
      At to_at{move.to_controller, move.to_location, move.to_sequence, to_position};
      At raw_to{move.to_controller, move.to_location, move.to_sequence,
                move.to_position};
      const int64_t code =
          ObserverTransitionCode(true, move.code, viewer, raw_to);
      RecordArgs args;
      args.kind = HistoryTokenKind::STATE_DELTA;
      args.subtype = HistorySubtype::MOVE;
      args.turn = current_turn;
      args.trace = message_cursor + move.trace_index;
      args.code = code;
      args.from_at = At{move.from_controller, move.from_location,
                        move.from_sequence, move.from_position};
      args.to_at = to_at;
      args.link = move.link;
      args.reason = move.reason;
      args.payload = {
          {"code", Num(code)},
          {"from", "[" + Num(move.from_controller) + "," +
                       Num(move.from_location) + "," + Num(move.from_sequence) +
                       "," + Num(move.from_position) + "]"},
          {"link", Num(move.link)},
          {"reason", Num(move.reason)},
          {"to", "[" + Num(move.to_controller) + "," + Num(move.to_location) +
                     "," + Num(move.to_sequence) + "," + Num(to_position) + "]"},
      };
      add(viewer, move.trace_index, 1, MakeRecord(viewer, args, table));
    }
  }

  for (const Draw& draw : events.draws) {
    for (int viewer = 0; viewer < 2; ++viewer) {
      /* identities arrive only through audience-filtered PUBLIC_EVENT records;
       * this count token never carries raw draw codes */
      RecordArgs args;
      args.kind = HistoryTokenKind::STATE_DELTA;
      args.subtype = HistorySubtype::DRAW;
      args.turn = current_turn;
      args.trace = message_cursor + draw.trace_index;
      args.player = draw.player;
      args.from_at = At{draw.player, LOCATION_DECK, 0, 0};
      args.to_at = At{draw.player, LOCATION_HAND, 0, 0};
      args.link = draw.link;
      args.count = draw.count();
      args.payload = {
          {"count", Num(draw.count())},
          {"link", Num(draw.link)},
          {"player", Num(draw.player)},
      };
      add(viewer, draw.trace_index, 1, MakeRecord(viewer, args, table));
    }
  }

  for (const PosChange& change : events.pos_changes) {
    for (int viewer = 0; viewer < 2; ++viewer) {
      At to_at{change.controller, change.location, change.sequence,
               change.current};
      const int64_t code =
          ObserverTransitionCode(false, change.code, viewer, to_at);
      RecordArgs args;
      args.kind = HistoryTokenKind::STATE_DELTA;
      args.subtype = HistorySubtype::POSITION;
      args.turn = current_turn;
      args.trace = message_cursor + change.trace_index;
      args.code = code;
      args.from_at = At{change.controller, change.location, change.sequence,
                        change.previous};
      args.to_at = to_at;
      args.link = change.link;
      args.payload = {
          {"at", "[" + Num(change.controller) + "," + Num(change.location) +
                     "," + Num(change.sequence) + "]"},
          {"code", Num(code)},
          {"current", Num(change.current)},
          {"link", Num(change.link)},
          {"previous", Num(change.previous)},
      };
      add(viewer, change.trace_index, 1, MakeRecord(viewer, args, table));
    }
  }

  for (const ChainEvent& event : events.chain) {
    int64_t effect_row = 0;
    bool exact = false;
    if (event.kind == CHAIN_CHAINING && event.code) {
      LegalAction probe;
      probe.code = event.code;
      probe.desc = event.desc;
      const ActionSemanticResolution resolved = table.Resolve(probe);
      effect_row = resolved.effect_row;
      exact = resolved.effect_exact;
    }
    for (int viewer = 0; viewer < 2; ++viewer) {
      RecordArgs args;
      args.kind = HistoryTokenKind::STATE_DELTA;
      args.subtype = HistorySubtype::CHAIN;
      args.turn = current_turn;
      args.trace = message_cursor + event.trace_index;
      args.code = event.code;
      args.effect_row = effect_row;
      args.effect_exact = exact;
      args.player = event.player;
      args.from_at = UnpackAt(event.at);
      args.link = event.link;
      args.value = event.kind;
      args.detail = event.desc;
      args.payload = {
          {"at", Num(event.at)},
          {"code", Num(event.code)},
          {"desc", Num(event.desc)},
          {"kind", Num(event.kind)},
          {"link", Num(event.link)},
          {"player", Num(event.player)},
      };
      add(viewer, event.trace_index, 1, MakeRecord(viewer, args, table));
    }
  }

  for (const ChainTarget& event : events.targets) {
    for (int viewer = 0; viewer < 2; ++viewer) {
      RecordArgs args;
      args.kind = HistoryTokenKind::STATE_DELTA;
      args.subtype = HistorySubtype::TARGET;
      args.turn = current_turn;
      args.trace = message_cursor + event.trace_index;
      args.to_at = UnpackAt(event.at);
      args.link = event.link;
      args.payload = {{"at", Num(event.at)}, {"link", Num(event.link)}};
      add(viewer, event.trace_index, 1, MakeRecord(viewer, args, table));
    }
  }

  for (const Attack& event : events.attacks) {
    for (int viewer = 0; viewer < 2; ++viewer) {
      RecordArgs args;
      args.kind = HistoryTokenKind::STATE_DELTA;
      args.subtype = HistorySubtype::ATTACK;
      args.turn = current_turn;
      args.trace = message_cursor + event.trace_index;
      args.from_at = UnpackAt(event.attacker);
      args.to_at = UnpackAt(event.target);
      args.payload = {{"attacker", Num(event.attacker)},
                      {"target", Num(event.target)}};
      add(viewer, event.trace_index, 1, MakeRecord(viewer, args, table));
    }
  }

  for (const LpEvent& event : events.lp_events) {
    for (int viewer = 0; viewer < 2; ++viewer) {
      RecordArgs args;
      args.kind = HistoryTokenKind::STATE_DELTA;
      args.subtype = HistorySubtype::LP;
      args.turn = current_turn;
      args.trace = message_cursor + event.trace_index;
      args.player = event.player;
      args.link = event.link;
      args.value = event.kind;
      args.amount = event.amount;
      args.flags = event.in_damage_step ? 1 : 0;
      args.payload = {
          {"amount", Num(event.amount)},
          {"in_damage_step", event.in_damage_step ? "true" : "false"},
          {"kind", Num(event.kind)},
          {"link", Num(event.link)},
          {"player", Num(event.player)},
      };
      add(viewer, event.trace_index, 1, MakeRecord(viewer, args, table));
    }
  }

  IntervalRecords out;
  for (auto& viewer_entry : staged) {
    for (auto& trace_entry : viewer_entry.second) {
      std::vector<Slot>& slots = trace_entry.second;
      std::stable_sort(slots.begin(), slots.end(),
                       [](const Slot& a, const Slot& b) {
                         return a.priority < b.priority;
                       });
      std::vector<HistoryTokenRecord> rows;
      rows.reserve(slots.size());
      for (Slot& slot : slots) rows.push_back(std::move(slot.row));
      out.by_viewer[viewer_entry.first][trace_entry.first] = std::move(rows);
    }
  }
  return out;
}

namespace {
/* The interval walk both token streams share: at every message the boundary
 * token first (NEW_TURN also advances the turn), then that message's facts,
 * carrying the turn in force after it (winbot_trajectory.py
 * ``interval_history_events``). */
IntervalStream WalkInterval(const std::vector<Message>& messages, const IntervalRecords& records,
                            int64_t message_cursor, int current_turn, const SemanticTable& table,
                            std::map<int, std::vector<uint8_t>>* starts_turn) {
  IntervalStream out;
  if (starts_turn != nullptr) { (*starts_turn)[0]; (*starts_turn)[1]; }
  out.by_viewer[0];
  out.by_viewer[1];
  int turn = current_turn;

  for (size_t local = 0; local < messages.size(); ++local) {
    const Message& m = messages[local];
    const int64_t global_trace = message_cursor + static_cast<int64_t>(local);

    if (m.msg == MSG_NEW_TURN) {
      /* the counter advances before this index's facts are emitted, so they
       * belong to the new turn (winbot_trajectory.py:1348-1361) */
      ++turn;
      const int player = m.payload.empty() ? 2 : m.payload[0];
      for (int viewer = 0; viewer < 2; ++viewer) {
        RecordArgs args;
        args.kind = HistoryTokenKind::BOUNDARY;
        args.subtype = HistorySubtype::NEW_TURN;
        args.turn = turn;
        args.trace = global_trace;
        args.player = player;
        args.payload = {{"player", Num(player)}, {"turn", Num(turn)}};
        out.by_viewer[viewer].push_back(MakeRecord(viewer, args, table));
        if (starts_turn != nullptr) (*starts_turn)[viewer].push_back(1);
      }
    } else if (m.msg == MSG_NEW_PHASE) {
      const int phase =
          m.payload.size() >= 2 ? static_cast<int>(ReadU16(m.payload, 0)) : 0;
      for (int viewer = 0; viewer < 2; ++viewer) {
        RecordArgs args;
        args.kind = HistoryTokenKind::BOUNDARY;
        args.subtype = HistorySubtype::NEW_PHASE;
        args.turn = turn;
        args.trace = global_trace;
        args.phase = phase;
        args.payload = {{"phase", Num(phase)}};
        out.by_viewer[viewer].push_back(MakeRecord(viewer, args, table));
        if (starts_turn != nullptr) (*starts_turn)[viewer].push_back(0);
      }
    } else if (m.msg == MSG_WIN) {
      const int winner = m.payload.empty() ? 2 : m.payload[0];
      const int reason = m.payload.size() > 1 ? m.payload[1] : 0;
      for (int viewer = 0; viewer < 2; ++viewer) {
        RecordArgs args;
        args.kind = HistoryTokenKind::BOUNDARY;
        args.subtype = HistorySubtype::TERMINAL;
        args.turn = turn;
        args.trace = global_trace;
        args.player = winner;
        args.reason = reason;
        args.payload = {{"reason", Num(reason)}, {"winner", Num(winner)}};
        out.by_viewer[viewer].push_back(MakeRecord(viewer, args, table));
        if (starts_turn != nullptr) (*starts_turn)[viewer].push_back(0);
      }
    }

    for (int viewer = 0; viewer < 2; ++viewer) {
      auto by_trace = records.by_viewer.find(viewer);
      if (by_trace == records.by_viewer.end()) continue;
      auto rows = by_trace->second.find(static_cast<int>(local));
      if (rows == by_trace->second.end()) continue;
      for (HistoryTokenRecord row : rows->second) {
        row.turn = turn;
        out.by_viewer[viewer].push_back(std::move(row));
        if (starts_turn != nullptr) (*starts_turn)[viewer].push_back(0);
      }
    }
  }
  out.current_turn = turn;
  return out;
}
}  // namespace

IntervalStream IntervalTokens(const std::vector<Message>& messages,
                              int64_t message_cursor, int current_turn,
                              const SemanticTable& table) {
  const IntervalEvents events = ParseInterval(messages);
  const IntervalRecords records =
      RecordsByTrace(events, message_cursor, current_turn, table);
  return WalkInterval(messages, records, message_cursor, current_turn, table, nullptr);
}

void RetagInterval(const std::vector<Message>& messages, ChainCarry* carry, IntervalEvents* events) {
  std::vector<int> chain, settlement;
  std::vector<uint8_t> damage;
  chain.reserve(messages.size());
  settlement.reserve(messages.size());
  damage.reserve(messages.size());
  for (const Message& m : messages) {
    UpdateChainLink(m, &carry->chain_link, true);
    UpdateChainLink(m, &carry->settlement_link, false);
    if (m.msg == MSG_DAMAGE_STEP_START) carry->in_damage_step = true;
    chain.push_back(carry->chain_link);
    settlement.push_back(carry->settlement_link);
    damage.push_back(carry->in_damage_step ? 1 : 0);
    if (m.msg == MSG_DAMAGE_STEP_END) carry->in_damage_step = false;
    if (m.msg == MSG_START || m.msg == MSG_NEW_TURN) {
      /* a chain or damage step cannot cross a game or turn boundary */
      carry->chain_link = carry->settlement_link = 0;
      carry->in_damage_step = false;
    }
  }
  for (Move& row : events->moves) row.link = settlement[row.trace_index];
  for (PosChange& row : events->pos_changes) row.link = settlement[row.trace_index];
  for (Draw& row : events->draws) row.link = settlement[row.trace_index];
  for (PublicEvent& row : events->public_events) row.link = chain[row.trace_index];
  for (ChainTarget& row : events->targets) row.link = chain[row.trace_index];
  for (LpEvent& row : events->lp_events) {
    row.link = chain[row.trace_index];
    row.in_damage_step = damage[row.trace_index] != 0;
  }
}

CarriedInterval CarriedIntervalTokens(const std::vector<Message>& messages, int64_t message_cursor,
                                      int* current_turn, ChainCarry* carry, const SemanticTable& table) {
  IntervalEvents events = ParseInterval(messages);
  RetagInterval(messages, carry, &events);
  const IntervalRecords records = RecordsByTrace(events, message_cursor, *current_turn, table);
  CarriedInterval out;
  IntervalStream stream = WalkInterval(messages, records, message_cursor, *current_turn, table, &out.starts_turn);
  out.by_viewer = std::move(stream.by_viewer);
  *current_turn = stream.current_turn;
  return out;
}

std::map<std::pair<int, int>, int> RoundtripOccupancy(
    const std::map<std::pair<int, int>, int>& before_counts,
    const IntervalEvents& events) {
  std::map<std::pair<int, int>, int> occupancy = before_counts;

  /* (trace, priority, kind, index); priority orders the three kinds at one
   * trace index and a stable sort keeps insertion order within a kind */
  struct Item { int trace; int priority; int kind; size_t index; };
  std::vector<Item> items;
  for (size_t i = 0; i < events.moves.size(); ++i)
    items.push_back({events.moves[i].trace_index, 0, 0, i});
  for (size_t i = 0; i < events.draws.size(); ++i)
    items.push_back({events.draws[i].trace_index, 1, 1, i});
  for (size_t i = 0; i < events.public_events.size(); ++i)
    if (events.public_events[i].kind == PUB_SWAP_GRAVE_DECK)
      items.push_back({events.public_events[i].trace_index, 2, 2, i});
  std::stable_sort(items.begin(), items.end(),
                   [](const Item& a, const Item& b) {
                     if (a.trace != b.trace) return a.trace < b.trace;
                     return a.priority < b.priority;
                   });

  for (const Item& item : items) {
    if (item.kind == 0) {
      const Move& move = events.moves[item.index];
      if (move.from_location && !(move.from_location & LOCATION_OVERLAY))
        occupancy[{move.from_controller, move.from_location}] -= 1;
      if (move.to_location && !(move.to_location & LOCATION_OVERLAY))
        occupancy[{move.to_controller, move.to_location}] += 1;
    } else if (item.kind == 1) {
      const Draw& draw = events.draws[item.index];
      occupancy[{draw.player, LOCATION_DECK}] -= draw.count();
      occupancy[{draw.player, LOCATION_HAND}] += draw.count();
    } else {
      const PublicEvent& event = events.public_events[item.index];
      const int deck = occupancy[{event.player, LOCATION_DECK}];
      const int grave = occupancy[{event.player, LOCATION_GRAVE}];
      occupancy[{event.player, LOCATION_DECK}] = grave;
      occupancy[{event.player, LOCATION_GRAVE}] = deck;
    }
  }
  return occupancy;
}

std::string CanonicalHistoryJson(const HistoryTokenRecord& r) {
  /* Field names in code-point order, because Python hashes
   * ``json.dumps(..., sort_keys=True)`` and the training loader rejects a batch
   * whose fingerprint moved.  The list is written out sorted rather than
   * sorted at run time so a new field cannot be added without noticing. */
  std::string out = "{";
  auto key = [&out](const char* name, bool first) {
    if (!first) out += ',';
    AppendJsonString(name, &out);
    out += ':';
  };
  key("amount", true);           out += Num(r.amount);
  key("card_code", false);       out += Num(r.card_code);
  key("card_effect_rows", false);
  out += '[';
  for (size_t i = 0; i < r.card_effect_rows.size(); ++i) {
    if (i) out += ',';
    out += Num(r.card_effect_rows[i]);
  }
  out += ']';
  key("card_row", false);        out += Num(r.card_row);
  key("count", false);           out += Num(r.count);
  key("decision_index", false);  out += Num(r.decision_index);
  key("detail", false);          out += Num(r.detail);
  key("effect_exact", false);    out += r.effect_exact ? "true" : "false";
  key("effect_row", false);      out += Num(r.effect_row);
  // R2 (2026-09-25): the record's public entity. Only the Python annotation
  // (history_entities) fills it, so a native record always carries 0 here.
  key("entity", false);          out += "0";
  key("flags", false);           out += Num(r.flags);
  key("from_controller_relative", false); out += Num(r.from_controller_relative);
  key("from_location", false);   out += Num(r.from_location);
  key("from_position", false);   out += Num(r.from_position);
  key("from_sequence", false);   out += Num(r.from_sequence);
  key("kind", false);            out += Num(r.kind);
  key("link", false);            out += Num(r.link);
  key("payload", false);
  out += '{';
  for (size_t i = 0; i < r.payload.size(); ++i) {
    if (i) out += ',';
    AppendJsonString(r.payload[i].first, &out);
    out += ':';
    out += r.payload[i].second;
  }
  out += '}';
  key("phase", false);           out += Num(r.phase);
  key("player_relative", false); out += Num(r.player_relative);
  key("public_event_kind", false); out += Num(r.public_event_kind);
  key("reason", false);          out += Num(r.reason);
  key("semantic_visible", false); out += r.semantic_visible ? "true" : "false";
  key("subtype", false);         out += Num(r.subtype);
  key("to_controller_relative", false); out += Num(r.to_controller_relative);
  key("to_location", false);     out += Num(r.to_location);
  key("to_position", false);     out += Num(r.to_position);
  key("to_sequence", false);     out += Num(r.to_sequence);
  key("trace_index", false);     out += Num(r.trace_index);
  key("turn", false);            out += Num(r.turn);
  key("value", false);           out += Num(r.value);
  out += '}';
  (void)&AtList;
  return out;
}

HistoryPrefixDigest::HistoryPrefixDigest() = default;

void HistoryPrefixDigest::Append(const HistoryTokenRecord& record) {
  buffered_ += CanonicalHistoryJson(record);
}

std::string HistoryPrefixDigest::Hex() const { return Sha256Hex(buffered_); }

}  // namespace mfenv
