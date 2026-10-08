/* One viewer's public tracker: entities, counters, hints and turn resources from its own stream.
 *
 * The native twin of ``mirrorforce/cardrules/public_trackers.py`` (the reference and the network client's
 * implementation); a contract test holds the two byte-identical on real games. It reads, in emission
 * order, only the records one viewer receives (``RecordsByTrace`` filters them by audience), the
 * viewer's own choices and the raw ``MSG_SWAP`` messages, which produce no history record. No core card
 * id, hidden layout or server-side effect table is read, so two hidden truths behind one public stream
 * give one tracker.
 *
 * Place logic is the ``EntityTracker`` (``mirrorforce_history_entities/v2``). The rules on top:
 * ``PUB_MISSED_EFFECT`` is ignored (the server sends it to the controller only, the parser marks it
 * public); Xyz materials are keyed by their host's location and sequence and follow its every move (left
 * the materials an Xyz summon attaches in the extra deck behind); an own choice in the opponent's hand, deck
 * or extra deck names canonical coordinates and so
 * no entity; a hand shuffle keeps the group (side, hand, code) of each renewed entity it knew; counters,
 * card hints and player hints follow the reference client (duelclient.cpp); per-turn and per-duel counts
 * of summons, activations and attacks reset at each new turn.
 *
 * A plain value type: copying it is a branch's fork, so search can snapshot it with its instance.
 * ``Export`` throws instead of guessing when the tracker and the public state disagree.
 */
#ifndef MFENV_PUBLIC_TRACKER_H_
#define MFENV_PUBLIC_TRACKER_H_

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "history.h"

namespace mfenv {

constexpr const char* kPublicTrackerSchema = "mfenv_public_tracker/v1";
constexpr int kTrackerTurnCounts = 6;
constexpr int kTrackerCardWidth = 16;
constexpr int kTrackerLinkWidth = 7;
constexpr int kTrackerLedgerWidth = 2 * (kTrackerTurnCounts + 1) + 3;
constexpr int kTrackerActivationWidth = 7;
constexpr int kTrackerHintWidth = 3;
constexpr size_t kTrackerActivationLimit = 32;
constexpr size_t kTrackerHintLimit = 8;

struct TrackerRowAnnotation {
  int64_t subject = 0;
  int64_t counterpart = 0;
  bool effect_present = false;
  int64_t effect_code = 0;
  int64_t effect_desc = 0;
};

/* One public card row as the tracker sees it, relative to the observer. The state tokens convert to it
 * (public_tracker_tokens.h); the env builds it from its own card rows. */
struct TrackerToken {
  enum Kind : int { CARD = 0, OVERLAY_MATERIAL = 1, KNOWN_UNPOSITIONED = 2, OTHER = 3 };
  int kind = CARD;
  int side = 0;           /* 0 the observer, 1 its opponent */
  int location = 0;       /* the card's location; for a material, its host's */
  int sequence = 0;       /* the card's sequence; for a material, its host's */
  int overlay_index = 0;  /* a material's index under its host */
  int64_t code = 0;
};

/* The observer's public view the tracker exports against: its card rows in order, and each zone's public card
 * count keyed by (side, location). */
struct TrackerView {
  std::vector<TrackerToken> tokens;
  std::map<std::pair<int, int>, int64_t> zone_counts;
};

struct TrackerExport {
  std::vector<std::array<int64_t, kTrackerCardWidth>> cards;
  std::vector<std::array<int64_t, kTrackerLinkWidth>> links;
  std::array<int64_t, kTrackerLedgerWidth> ledger{};
  std::vector<std::array<int64_t, kTrackerActivationWidth>> activations;
  std::vector<std::array<int64_t, kTrackerHintWidth>> hints;
};

/* An explicit current-view seed, not a synthetic event history. Coordinates are relative to the viewer;
 * overlay position is the material index. Missing arrival/use history stays unknown/zero. */
struct TrackerSeedCard {
  int side = 0, location = 0, sequence = 0, position = 0;
  int64_t code = 0, arrival_turn = -1;
  int arrival_kind = 0, arrived_from = 0;
  int64_t activations = 0, attacks = 0, hint_kind = 0, hint_value = 0;
  std::map<int64_t, int64_t> counters, desc_hints;
};
struct TrackerSeed {
  std::vector<TrackerSeedCard> cards;
  std::array<std::array<int64_t, kTrackerTurnCounts>, 2> counts{};
  std::array<std::map<int64_t, int64_t>, 2> hints;
};

class PublicTracker {
 public:
  void SeedCurrent(int turn, const TrackerSeed& seed);
  TrackerRowAnnotation Consume(const HistoryTokenRecord& record);
  /* The viewer's own choice of the card row ``token`` (the card the choice record names; null for none). */
  TrackerRowAnnotation ConsumeOwnChoice(const HistoryTokenRecord& record, const TrackerToken* token);
  /* The code of the card the tracker places at a relative place (hand, graveyard, banished, field), or 0 when its
   * identity is not known there (never seen, or a shuffle took its place away). */
  int64_t KnownCode(int side, int location, int sequence) const;
  /* A list zone's (hand, graveyard, banished) tracked entities in sequence order, each with the code the tracker knows
   * for it (0 when unknown). */
  std::vector<std::pair<int64_t, int64_t>> ListEntities(int side, int location) const;
  /* The entity at a record's origin (0 for none) and the code the tracker knows for an entity (0 when unknown). */
  int64_t EntityFrom(const HistoryTokenRecord& record) const { return From(record); }
  int64_t CodeOf(int64_t entity) const;
  /* The location (base) the card the tracker places at a relative place came from (0 when unknown). */
  int ArrivedFrom(int side, int location, int sequence) const;
  /* The entity at a relative place (0 for none), whether the tracker holds an entity face-down, and the base location
   * an entity last arrived from (0 when unknown). */
  int64_t EntityAt(int side, int location, int sequence) const { return At(side, location, sequence, 0); }
  bool FaceDown(int64_t entity) const { return facedown_.count(entity) != 0; }
  int ArrivedFromEntity(int64_t entity) const;
  /* Per card row of ``view``, the entity the tracker follows there (0 where it follows none: decks, extra decks, the
   * opponent's hand); the places Export reads. Call after Export, which may resync a graveyard. */
  std::vector<int64_t> RowEntities(const TrackerView& view) const;
  /* ``MSG_SWAP`` between two relative places. */
  void Swap(int side_a, int location_a, int sequence_a, int side_b, int location_b, int sequence_b);
  /* Per card row of ``view``, per retained history row, and the turn ledger and tables. May refill a
   * graveyard the last grave/deck swap emptied; otherwise changes nothing. */
  TrackerExport Export(const TrackerView& view, int turn, const std::vector<TrackerRowAnnotation>& rows);

 private:
  struct EntityInfo {
    int64_t arrival_turn = -1;
    int arrival_kind = 0;
    int arrived_from = 0;
    int64_t activations = 0;
    int64_t attacks = 0;
    std::map<int64_t, int64_t> counters;
    std::map<int64_t, int64_t> desc_hints;
    int64_t hint_kind = 0;
    int64_t hint_value = 0;
  };
  using Zone = std::pair<int, int>;

  int64_t At(int side, int location, int sequence, int position) const;
  int64_t From(const HistoryTokenRecord& r) const;
  int64_t To(const HistoryTokenRecord& r) const;
  int64_t New() { return next_id_++; }
  int64_t Take(int side, int location, int sequence, int position);
  void Put(int64_t entity, int side, int location, int sequence, int position);
  void Learn(int64_t entity, int64_t code);
  void Forget(int64_t entity);
  EntityInfo& Info(int64_t entity) { return info_[entity]; }
  int64_t Move(const HistoryTokenRecord& r);
  int64_t Public(const HistoryTokenRecord& r);
  void NewTurn(int64_t turn);
  void Count(int side, int column);
  void Check(const TrackerView& view);
  std::array<int64_t, kTrackerCardWidth> CardRow(int64_t entity, int turn) const;

  int64_t next_id_ = 1;
  std::map<Zone, std::vector<int64_t>> lists_;
  std::map<Zone, std::map<int, int64_t>> fixed_;
  /* (side, host location, host sequence): materials follow every move of their host */
  std::map<std::tuple<int, int, int>, std::vector<int64_t>> overlays_;
  std::map<int64_t, int64_t> codes_;
  std::set<int64_t> facedown_;
  std::map<int, std::vector<int64_t>> reveals_;
  int64_t turn_ = 0;
  std::map<int64_t, EntityInfo> info_;
  std::map<int64_t, std::tuple<int, int, int64_t>> absorbed_;
  std::set<int> resync_;
  std::array<std::array<int64_t, kTrackerTurnCounts>, 2> counts_{};
  /* (code, desc) -> turn0, turn1, duel0, duel1, last turn, order */
  std::map<std::pair<int64_t, int64_t>, std::array<int64_t, 6>> activations_;
  int64_t order_ = 0;
  std::array<std::map<int64_t, int64_t>, 2> hints_;
  bool cant_check_grave_ = false;
};

}  // namespace mfenv

#endif  /* MFENV_PUBLIC_TRACKER_H_ */
