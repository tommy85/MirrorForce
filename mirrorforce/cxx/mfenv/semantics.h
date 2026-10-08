/* ActionSemanticResolver, ported from mirrorforce/mirrorforce/common/action_semantics.py.
 *
 * The rule this file exists to keep is the one that module's docstring states:
 * an effect is identified by the *full* description, never by
 * ``LegalAction.effect``.  That field is the printed string slot, not the Lua
 * ``RegisterEffect`` index, and the two disagree on 35.55% of activation
 * candidates (netduel/actions.py:117-121).  Resolution fails soft into row zero
 * with an explicit reason; it never guesses an index and never substitutes a
 * whole-card mean.
 *
 * The three lookup tables come from a sidecar written by
 * ``mirrorforce.cxxenv.semantics_export`` -- see that module for the layout and
 * for why the artifacts are not parsed here.  A table is immutable once loaded,
 * so every instance thread shares one.
 */
#ifndef MFENV_SEMANTICS_H_
#define MFENV_SEMANTICS_H_

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "actions.h"

namespace mfenv {

class SemanticsError : public std::runtime_error {
 public:
  explicit SemanticsError(const std::string& what) : std::runtime_error(what) {}
};

/* action_semantics.py:40-48 -- why an action did or did not get an exact row.
 * The wire values are the enum's *strings* on the Python side; the C++ side
 * keeps the same order and converts at the boundary so the comparison is on
 * the same tokens. */
enum class EffectResolution : int {
  EXACT = 0,
  NO_CARD = 1,
  NO_DESCRIPTION = 2,
  UNMAPPED_DESCRIPTION = 3,
  AMBIGUOUS_DESCRIPTION = 4,
  MISSING_EFFECT_ARTIFACT = 5,
};

const char* EffectResolutionName(EffectResolution value);

/* action_semantics.py:65-76 */
struct ActionSemanticResolution {
  int64_t passcode = 0;
  int64_t card_row = 0;
  bool card_known = false;
  bool has_effect_index = false;   /* Python's ``effect_index is None`` */
  int64_t effect_index = 0;
  int64_t effect_row = 0;
  bool effect_exact = false;
  EffectResolution resolution = EffectResolution::NO_CARD;
  std::vector<int64_t> ambiguous_candidates;
};

class SemanticTable {
 public:
  explicit SemanticTable(const std::string& path);
  /* Card rows only (no effect rows or descriptions): the table a component with its own card ids builds, such as
   * the env from its code list. */
  SemanticTable(const std::unordered_map<uint32_t, int64_t>& card_rows, const std::string& artifact_fingerprint);

  ActionSemanticResolution Resolve(const LegalAction& action) const;

  /* effect_encoder.py:604-609 -- every effect row of one card, in effect-index
   * order.  ``padded_effect_rows_for_cards`` pads these to the batch's widest
   * card, which is what the state block's effect_rows/effect_mask carry. */
  const std::vector<int64_t>& EffectRowsForCard(uint32_t code) const;
  /* cardfeat.py:945-960 -- every effect index that sets this description.
   * One element is an exact binding, several a genuine ambiguity, empty
   * unmapped.  ``_deck_choice_assignments`` uses only emptiness. */
  std::vector<int64_t> CandidatesForDesc(uint32_t code, uint32_t desc) const;
  int64_t CardRow(uint32_t code) const;

  const std::string& script_revision() const { return script_revision_; }
  const std::string& artifact_fingerprint() const { return artifact_fingerprint_; }
  int description_limit() const { return description_limit_; }
  size_t card_rows() const { return card_row_by_code_.size(); }
  size_t effect_rows() const { return effect_row_by_identity_.size(); }
  size_t desc_rows() const { return desc_unique_.size(); }
  size_t ambiguous_rows() const { return desc_ambiguous_.size(); }
  const std::string& path() const { return path_; }

 private:
  /* action_semantics.py:120-133: ``code`` first, then ``response`` -- the
   * latter is the public candidate identity of a MSG_ANNOUNCE_CARD row. */
  static int64_t Passcode(const LegalAction& action);

  std::string path_;
  std::string script_revision_;
  std::string artifact_fingerprint_;
  int description_limit_ = 10000;
  std::unordered_map<uint32_t, int64_t> card_row_by_code_;
  /* key = code << 8 | (index + 1); effect indices are small non-negative ints
   * (max 13 in the production artifacts) and the +1 keeps a zero key free */
  std::unordered_map<uint64_t, int64_t> effect_row_by_identity_;
  std::unordered_map<uint64_t, int32_t> desc_unique_;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> desc_ambiguous_;
  std::vector<int32_t> ambiguous_index_;
  std::unordered_map<uint32_t, std::vector<int64_t>> effect_rows_by_card_;
};

/* Install the process default, the table the replay path resolves against.
 * Idempotent for the same path; a different path throws, because a resolver
 * silently swapped mid-run would produce two row spaces in one corpus. */
void InstallSemantics(const std::string& path);
/* nullptr when no table has been installed; the environment then reports rows
 * as "not resolved" rather than inventing them. */
const SemanticTable* DefaultSemantics();

/* The canonical encoding of one prompt's whole menu, as a JSON array of rows.
 *
 * Each row is
 *   [spec, act, phase, finish, position, effect, number, place, attribute,
 *    race, code, response, msg, desc,          -- LegalAction's 14 fields
 *    describe, key,                            -- describe() and SnapshotAction.key
 *    passcode, card_row, card_known, effect_index, effect_row, effect_exact,
 *    resolution, ambiguous_candidates]         -- ActionSemanticResolution
 *
 * ``effect_index`` is JSON null when Python's is ``None``.  The string must be
 * byte-identical to Python's ``json.dumps(rows, ensure_ascii=True,
 * separators=(",", ":"))`` -- the digest over it is what the contract compares,
 * so an encoding difference would look like a semantic difference.
 * ``mirrorforce/mirrorforce/cxxenv/contract.py`` holds the Python twin. */
std::string EncodeMenuRows(const std::vector<LegalAction>& actions, int prompt_msg,
                           const SemanticTable* table);

/* simulator_adapter.py:106-134 -- the canonical public selector identity.
 * JSON is used as a collision-free encoding, not as a hash, so the string must
 * match Python's ``json.dumps(..., ensure_ascii=True, separators=(",", ":"))``
 * byte for byte. */
std::string SnapshotActionKey(const LegalAction& action, int prompt_msg,
                              int menu_index);

}  // namespace mfenv

#endif  /* MFENV_SEMANTICS_H_ */
