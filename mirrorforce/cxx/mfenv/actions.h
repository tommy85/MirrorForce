/* Legal actions and response encoding, ported from the Python action layer.
 *
 * Sources, ported line by line:
 *   * mirrorforce/mirrorforce/netduel/actions.py   -- LegalAction, Reader,
 *     ls_to_spec / spec_to_ls / unpack_desc / flag_to_usable_places /
 *     place_to_ls, the select_sum combination search, card_is_declarable
 *     (in cards.cc), every Selector and the whole ``parse_select`` switch
 *   * mirrorforce/mirrorforce/replay/actions.py   -- MultiSelect, the response
 *     codec (encode_responsei / encode_selection)
 *
 * The invariant this file exists to hold: for the same message bytes, the
 * option list, its order, and the bytes ``choose(i)`` returns are identical to
 * the Python ones.  An index means the same thing to the policy and to the
 * engine only if that holds.
 */
#ifndef MFENV_ACTIONS_H_
#define MFENV_ACTIONS_H_

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mfenv {

class CardPool;

/* actions.py:68 */
class UnsupportedMessage : public std::runtime_error {
 public:
  explicit UnsupportedMessage(const std::string& what) : std::runtime_error(what) {}
};

/* actions.py:72-82 */
enum class ActionAct : int {
  NONE = 0, SET = 1, REPO = 2, SPSUMMON = 3, SUMMON = 4,
  MSET = 5, ATTACK = 6, DIRECT_ATTACK = 7, ACTIVATE = 8, CANCEL = 9,
};

/* actions.py:85-89 */
enum class ActionPhase : int { NONE = 0, BATTLE = 1, MAIN2 = 2, END = 3 };

/* actions.py:92-97 */
constexpr int PLACE_MZONE1 = 1;
constexpr int PLACE_SZONE1 = 8;
constexpr int PLACE_OP_MZONE1 = 16;
constexpr int PLACE_OP_SZONE1 = 23;

constexpr int DESCRIPTION_LIMIT = 10000;
constexpr int CARD_EFFECT_OFFSET = 10010;

/* actions.py:100-147 */
struct LegalAction {
  std::string spec;
  ActionAct act = ActionAct::NONE;
  ActionPhase phase = ActionPhase::NONE;
  bool finish = false;
  int position = 0;
  int effect = -1;
  int64_t number = 0;
  int place = 0;
  uint32_t attribute = 0;
  uint32_t race = 0;
  uint32_t code = 0;
  uint32_t response = 0;
  int msg = 0;
  uint32_t desc = 0;

  std::string Describe() const;
};

/* actions.py:194-213 */
std::string LsToSpec(int loc, int seq, int pos, bool opponent);
/* actions.py:269-279 */
void PlaceToLs(int place, int player, int* controller, int* location, int* sequence);
/* actions.py:254-266 */
std::vector<int> FlagToUsablePlaces(uint32_t flag, bool reverse = false);
/* actions.py:243-251 */
void UnpackDesc(uint32_t code, uint32_t desc, uint32_t* out_code, int* out_index);
/* actions.py:303-317 */
std::vector<int64_t> GetSumParams(uint32_t param);

/* replay/actions.py:103-122 */
std::vector<uint8_t> EncodeResponsei(int64_t value);
std::vector<uint8_t> EncodeSelection(const std::vector<int>& indices, int must = 0);

/* actions.py SelectionRole -- what the chooser's already-selected cards are. */
enum class SelectionRole : int { NONE = 0, GENERIC = 1, MATERIAL = 2, TRIBUTE = 3 };

/* actions.py SELECTION_ROLE_BY_MSG: the role of a card-selection prompt, NONE
 * for every prompt that carries no selection context. */
SelectionRole SelectionRoleFor(int msg);

/* replay/actions.py:155-284 -- the ygoenv multi-select state machine. */
struct SelectionContext {
  /* The actor choice context (mfenv_actor_choice_context/v1): only a
   * MultiSelect prompt is ``active``. */
  bool active = false;
  int stage = 0, minimum = 0, maximum = 0, must = 0, mode = 0;
  std::vector<int> selected_indices;
  std::vector<std::string> specs;
  std::vector<LegalAction> candidates;
  /* actions.py PromptSelection, the model input's card-selection context (R3).
   * ``role`` is NONE for any prompt other than the four card selections.
   * ``cards`` is every card the prompt names (candidates, then the sum's
   * must-select cards or the unselect prompt's already-selected list), bound
   * to public slots once; ``selected_cards`` indexes the current selection in
   * it; the counts and values are in the units PromptSelection documents. */
  SelectionRole role = SelectionRole::NONE;
  int count_minimum = 0, count_maximum = 0;
  std::vector<LegalAction> cards;
  std::vector<int> selected_cards;
  int64_t value_target = 0, value_low = 0, value_high = 0;
};

class MultiSelect {
 public:
  MultiSelect(int min, int max, int must, std::vector<std::string> specs, int mode,
              std::vector<std::vector<int>> combs, std::vector<int> weights);

  /* Legal actions of the current sub-step.  ``has_finish`` marks the trailing
   * synthetic finish action (Python's ``None``). */
  void Options(std::vector<std::string>* specs, bool* has_finish) const;
  void Step(int option);

  bool done() const { return done_; }
  bool has_picks() const { return !r_idxs_.empty(); }
  int selected_weight() const { return SelectedWeight(); }
  const std::vector<uint8_t>& response() const { return response_; }
  int SpecIndex(const std::string& spec) const;
  SelectionContext PublicContext() const {
    SelectionContext out;
    out.active = !done_;
    out.stage = idx_;
    out.minimum = min_;
    out.maximum = max_;
    out.must = must_;
    out.mode = mode_;
    out.selected_indices = r_idxs_;
    out.specs = specs_;
    return out;
  }

 private:
  void StepMode0(const std::string* spec);
  void StepMode1(const std::string* spec);
  int SelectedWeight() const;
  bool Mode0Viable(int index) const;

  int min_, max_, must_, mode_;
  std::vector<std::string> specs_;
  std::vector<std::vector<int>> combs_;
  std::vector<int> weights_;
  std::vector<int> r_idxs_;
  /* spec2idx: Python deletes entries as cards are taken, and the deletion is
   * observable through ``options()``, so the map is modelled the same way. */
  std::unordered_map<std::string, int> spec2idx_;
  int idx_ = 0;
  bool done_ = false;
  std::vector<uint8_t> response_;
};

/* actions.py:505-528.  ``Choose`` returns true and fills ``out`` when the
 * selection is finished; false means another round on the same prompt. */
class Selector {
 public:
  virtual ~Selector() = default;
  virtual std::vector<LegalAction> Options() = 0;
  virtual bool Choose(int idx, std::vector<uint8_t>* out) = 0;
  virtual SelectionContext PublicContext() const { return {}; }
  int msg() const { return msg_; }
  int player() const { return player_; }

 protected:
  Selector(int msg, int player) : msg_(msg), player_(player) {}
  int msg_;
  int player_;
};

/* actions.py:679-698 */
struct SelectContext {
  int our_player = 0;
  int current_phase = 0;
  bool discard_hand = false;
  int max_options = 24;
  const CardPool* card_pool = nullptr;
  std::vector<uint32_t> known_codes;
  bool full_phase_menu = false;
  bool auto_end_phase_discard = true;
  bool full_card_sort_menu = false;
  bool commit_command_selections = false;
  int command_player = -1;
  /* ``random.Random(("select", seed))`` is only ever consumed by the automatic
   * end-phase discard (actions.py:906-907), which exact replay turns off
   * (winbot_teacher.py:292).  The stream is kept behind an explicit seed so the
   * ABI is honest about it (design §3.6). */
  uint64_t select_seed = 0;
};

/* actions.py:701-722 */
struct SelectResult {
  int msg = 0;
  int player = 0;
  bool has_auto_response = false;
  std::vector<uint8_t> auto_response;
  std::unique_ptr<Selector> selector;
  std::string note;
  bool complete_menu = true;
};

/* actions.py:732-1281 */
SelectResult ParseSelect(int msg, const std::vector<uint8_t>& payload,
                         SelectContext* ctx);

/* actions.py::track_command_follow_up, before parsing every game message. */
void TrackCommandFollowUp(SelectContext* ctx, int msg,
                          const std::vector<uint8_t>& body);

}  // namespace mfenv

#endif  /* MFENV_ACTIONS_H_ */
