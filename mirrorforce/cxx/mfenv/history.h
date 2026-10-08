/* The observer-visible history stream: events, tokens and the prefix digest.
 *
 * Port of the message-interval half of the trajectory exporter:
 *
 *   parse_moves / parse_public_events / parse_chain /
 *   parse_targets / parse_attacks / parse_lp_events   worldmodel/state.py
 *   _records_by_trace / _record / boundary tokens     common/winbot_trajectory.py
 *   _canonical / history_prefix_fingerprints          common/winbot_trajectory.py
 *
 * Three things here are load-bearing and none of them announce themselves when
 * wrong.
 *
 * **Audience.** The in-process core writes one omniscient message buffer; the
 * real server routes or masks some messages per seat.  ``PublicEvent::audience``
 * is that routing, kept in the corpus so a seat's history cannot contain an
 * opponent-only confirmation (state.py:976-990).  A deck confirmation goes to
 * the requesting seat only; a draw's identity goes to the drawing seat unless
 * the core marked it public.
 *
 * **Ordering.** Tokens are emitted per raw message index, and within one index
 * public events come before everything else (priority 0 vs 1,
 * winbot_trajectory.py:1067-1068).  ``trace_index`` is
 * ``message_cursor + local index``, so the C++ message count has to equal
 * ``split_messages``' -- which phase (1) already establishes.
 *
 * **The canonical JSON.** The prefix fingerprint is a running sha256 over
 * ``json.dumps(asdict(record), sort_keys=True, separators=(",",":"),
 * ensure_ascii=True)``.  Unlike every other encoder in this port, the keys are
 * *sorted*, not fixed-order, because that is what the Python side hashes and the
 * training loader rejects a batch whose fingerprint moved
 * (winbot_trajectory.py:564-566).
 */
#ifndef MFENV_HISTORY_H_
#define MFENV_HISTORY_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "messages.h"
#include "semantics.h"

namespace mfenv {

/* semantic_model.py:877-882 */
enum class HistoryTokenKind : int {
  ACTION = 0,
  PRIVATE_CHOICE = 1,
  PUBLIC_EVENT = 2,
  STATE_DELTA = 3,
  BOUNDARY = 4,
};

/* winbot_trajectory.py:119-133 */
enum class HistorySubtype : int {
  ACTION_SELECTED = 0,
  PRIVATE_SELECTED = 1,
  PUBLIC_RESULT = 2,
  MOVE = 3,
  DRAW = 4,
  POSITION = 5,
  CHAIN = 6,
  TARGET = 7,
  ATTACK = 8,
  LP = 9,
  NEGATED = 10,
  NEW_TURN = 11,
  NEW_PHASE = 12,
  TERMINAL = 13,
};

/* state.py:934-973 -- the public event vocabulary. */
enum PublicEventKind {
  PUB_CONFIRM = 1, PUB_RANDOM = 2, PUB_COIN = 3, PUB_DICE = 4,
  PUB_ANNOUNCE_RACE = 5, PUB_ANNOUNCE_ATTRIB = 6, PUB_ANNOUNCE_CODE = 7,
  PUB_ANNOUNCE_NUMBER = 8, PUB_ANNOUNCE_ZONE = 9, PUB_OPSELECTED = 10,
  PUB_SHUFFLE_DECK = 11, PUB_SHUFFLE_HAND = 12, PUB_SHUFFLE_EXTRA = 13,
  PUB_SHUFFLE_SET = 14, PUB_HAND_RESULT = 15, PUB_DECK_TOP = 16,
  PUB_EQUIP = 17, PUB_UNEQUIP = 18, PUB_CARD_TARGET = 19,
  PUB_CANCEL_TARGET = 20, PUB_FIELD_DISABLED = 21, PUB_ADD_COUNTER = 22,
  PUB_REMOVE_COUNTER = 23, PUB_HINT_CARD = 24, PUB_CARD_HINT = 25,
  PUB_PLAYER_HINT = 26, PUB_SWAP_GRAVE_DECK = 27, PUB_REVERSE_DECK = 28,
  PUB_DRAW_REVEAL = 29, PUB_ATTACK_DISABLED = 30, PUB_MISSED_EFFECT = 31,
  PUB_SUMMONING = 32, PUB_SPSUMMONING = 33, PUB_FLIPSUMMONING = 34,
  PUB_SET = 35, PUB_NEW_TURN = 36, PUB_SUMMONED = 37, PUB_SPSUMMONED = 38,
  PUB_FLIPSUMMONED = 39, PUB_DAMAGE_STEP_START = 40,
};

/* state.py:893-919 */
enum ChainKind {
  CHAIN_CHAINING = 0, CHAIN_SOLVED = 1, CHAIN_NEGATED = 2,
  CHAIN_DISABLED = 3, CHAIN_SOLVING = 4,
};

/* state.py:917 */
enum LpKind { LP_DAMAGE = 0, LP_RECOVER = 1, LP_PAY_COST = 2, LP_UPDATE = 3 };

struct Move {
  uint32_t code = 0;
  int from_controller = 0, from_location = 0, from_sequence = 0, from_position = 0;
  int to_controller = 0, to_location = 0, to_sequence = 0, to_position = 0;
  uint32_t reason = 0;
  int link = 0;
  int trace_index = 0;
};

struct PosChange {
  uint32_t code = 0;
  int controller = 0, location = 0, sequence = 0, previous = 0, current = 0;
  int link = 0;
  int trace_index = 0;
};

struct Draw {
  int player = 0;
  std::vector<uint32_t> codes;
  int link = 0;
  int trace_index = 0;
  int count() const { return static_cast<int>(codes.size()); }
};

struct ChainEvent {
  int kind = 0;
  int link = 0;
  uint32_t code = 0;
  uint32_t at = 0;
  uint32_t desc = 0;
  /* zero, not the "unknown seat" 2: a result message (SOLVED / NEGATED /
   * DISABLED / SOLVING) carries only its link number and Python leaves every
   * other field at zero, which the token's player_relative then reflects
   * (state.py:893-909) */
  int player = 0;
  int trace_index = 0;
};

struct ChainTarget {
  int link = 0;
  uint32_t at = 0;
  int trace_index = 0;
};

struct Attack {
  uint32_t attacker = 0;
  uint32_t target = 0;
  int trace_index = 0;
};

struct LpEvent {
  int kind = 0;
  int player = 0;
  uint32_t amount = 0;
  int link = 0;
  bool in_damage_step = false;
  int trace_index = 0;
};

struct PublicEvent {
  int kind = 0;
  int player = 2;
  uint32_t code = 0;
  uint32_t at = 0;
  uint32_t target = 0;
  uint32_t value = 0;
  uint32_t detail = 0;
  int link = 0;
  int audience = 3;
  int trace_index = 0;
};

/* What ``diff_snapshots`` extracts from one message interval.  Only the
 * message-derived half is here: the snapshot-derived fields (appeared,
 * vanished, count_delta, negated) belong to the settlement target, not to the
 * observer history. */
struct IntervalEvents {
  std::vector<Move> moves;
  std::vector<PosChange> pos_changes;
  std::vector<Draw> draws;
  std::vector<ChainEvent> chain;
  std::vector<ChainTarget> targets;
  std::vector<Attack> attacks;
  std::vector<LpEvent> lp_events;
  std::vector<PublicEvent> public_events;
};

IntervalEvents ParseInterval(const std::vector<Message>& messages);

/* winbot_trajectory.py:231-272 -- one structured observer-visible fact. */
struct HistoryTokenRecord {
  int kind = 0;
  int subtype = 0;
  int turn = 0;
  int64_t trace_index = 0;
  int player_relative = 2;
  int from_controller_relative = 2;
  int from_location = 0;
  int from_sequence = 0;
  int to_controller_relative = 2;
  int to_location = 0;
  int to_sequence = 0;
  int phase = 0;
  int link = 0;
  int public_event_kind = 0;
  int64_t value = 0;
  int64_t detail = 0;
  int64_t reason = 0;
  int64_t amount = 0;
  int64_t count = 0;
  int from_position = 0;
  int to_position = 0;
  int64_t flags = 0;
  int64_t card_code = 0;
  int64_t card_row = 0;
  std::vector<int64_t> card_effect_rows;
  int64_t effect_row = 0;
  bool effect_exact = false;
  bool semantic_visible = false;
  int decision_index = -1;
  /* ``payload`` is a canonical lossless record of the public fact.  Its keys
   * differ per subtype, so it is kept as an ordered key/value list rather than
   * a struct; the canonical JSON sorts it anyway. */
  std::vector<std::pair<std::string, std::string>> payload;
};

/* winbot_trajectory.py:1061-1308.  Returns, per viewer, the tokens produced at
 * each *local* message index, already ordered (public events first). */
struct IntervalRecords {
  /* viewer -> local trace index -> tokens */
  std::map<int, std::map<int, std::vector<HistoryTokenRecord>>> by_viewer;
};

IntervalRecords RecordsByTrace(const IntervalEvents& events, int64_t message_cursor,
                               int current_turn, const SemanticTable& table);

/* The complete observer stream for one interval, in emission order.
 *
 * ``RecordsByTrace`` gives the facts; this walks the raw messages and
 * interleaves them with the boundary tokens, which is what
 * ``_TrajectoryReplay.flush`` does (winbot_trajectory.py:1346-1399).  The order
 * matters twice over: NEW_TURN increments the turn counter *before* the facts
 * at that same message index are emitted, so those facts carry the new turn
 * ("NEW_TURN changed the active segment before its facts", :1391), and the
 * prefix fingerprint is a running digest over exactly this sequence.
 */
struct IntervalStream {
  /* viewer -> tokens, in emission order */
  std::map<int, std::vector<HistoryTokenRecord>> by_viewer;
  /* the turn counter after the walk; the caller carries it to the next call */
  int current_turn = 0;
};

IntervalStream IntervalTokens(const std::vector<Message>& messages,
                              int64_t message_cursor, int current_turn,
                              const SemanticTable& table);

/* state.py ``PublicChainContext``: the chain attribution a client carries from
 * one message interval to the next (active link, settlement link, damage step). */
struct ChainCarry {
  int chain_link = 0;
  int settlement_link = 0;
  bool in_damage_step = false;
};

/* ``PublicChainContext.retag``: give every event of an interval the attribution
 * the carried state implies, and advance the carry over the interval. */
void RetagInterval(const std::vector<Message>& messages, ChainCarry* carry, IntervalEvents* events);

/* ``MessageHistoryStream.advance`` (winbot_trajectory.py): one interval's
 * observer history with the chain attribution carried in ``carry``; each
 * viewer's tokens in emission order, and which of them open a turn segment
 * (the NEW_TURN boundary). ``current_turn`` is advanced. */
struct CarriedInterval {
  std::map<int, std::vector<HistoryTokenRecord>> by_viewer;
  std::map<int, std::vector<uint8_t>> starts_turn;
};
CarriedInterval CarriedIntervalTokens(const std::vector<Message>& messages, int64_t message_cursor,
                                      int* current_turn, ChainCarry* carry, const SemanticTable& table);

/* state.py:1634-1681 -- the occupancy implied by replaying the interval's
 * count-changing events onto a starting count map.
 *
 * The settlement target is only worth training on if it *determines* the next
 * board, and that is what this checks: moves, draws and zone-wide swaps applied
 * in their shared wire order must reproduce the after-snapshot's counts.  The
 * order is load-bearing -- MSG_SWAP_GRAVE_DECK is not accompanied by one move
 * per card, so applying an activation-cost move after rather than before the
 * swap gives a different answer.  Overlay traffic is excluded: a material is
 * not a zone occupant.
 */
std::map<std::pair<int, int>, int> RoundtripOccupancy(
    const std::map<std::pair<int, int>, int>& before_counts,
    const IntervalEvents& events);

/* winbot_trajectory.py:569-572 -- the record as Python's
 * ``json.dumps(asdict(record), sort_keys=True, separators=(",",":"),
 * ensure_ascii=True)``.  Keys sorted, not fixed-order. */
std::string CanonicalHistoryJson(const HistoryTokenRecord& record);

/* A running sha256 over canonical records, in order.  Advancing it is what a
 * decision's ``input_prefix_fingerprint`` is: the digest of everything this
 * viewer had observed before that decision. */
class HistoryPrefixDigest {
 public:
  HistoryPrefixDigest();
  void Append(const HistoryTokenRecord& record);
  /* hex digest of everything appended so far */
  std::string Hex() const;

 private:
  std::string buffered_;
};

}  // namespace mfenv

#endif  /* MFENV_HISTORY_H_ */
