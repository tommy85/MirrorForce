"""PV-14 world-model label factory.

Turns duels into supervised targets for the two prediction heads the world
model is built from:

* the **menu head** -- for every candidate the card-knowledge-free enumerator
  can name at a decision point, is it in the engine's legal-action menu?
* the **settlement head** -- given the state and the action actually taken,
  what does the board look like afterwards?

plus the **next-menu head**, which asks what our own options become one action
later when the opponent is forced to pass, so the target is a function of our
state and our action alone.

Everything here runs on CPU against the pinned ygopro-core through
``mirrorforce.puzzle.core``; nothing needs ygoenv, a GPU or a network.  See
``mirrorforce/v1/pv/pv14-pipeline.md`` for the measured coverage and the corpus
statistics.
"""

from __future__ import annotations

SCHEMA_VERSION = "pv14-13"
#: pv14-13 (2026-08-27): menu CARD rows gain the four runtime columns ``attribute`` / ``race`` /
#: ``link`` / ``link_marker``. ``parse_query_segments`` and ``CardState`` always decoded these four fields,
#: and the writer always dropped them; the material checks of 10.38 of the 15 cards in each Extra Deck
#: need exactly them.
#: Writer-side masking is unchanged: ``_blank`` zeroes these four for invisible cards too.
#: The version jumps from 10 straight to 13 to merge with ``export_replay_labels.OUTPUT_SCHEMA``
#: (which had reached pv14-12 on its own) into one numbering line, so the two version series cannot be confused.
#: pv14-10 (2026-08-25): sub-prompts of our side within a resolution keep the complete candidate set; the opponent's private
#: ``MSG_SELECT_*`` are removed entirely from that viewpoint's sub column, keeping only the ``DISCLOSE`` really broadcast
#: afterwards. This gives deterministic supervision of "does a selection box appear + our selectable range";
#: the actually chosen option / random reveal remains an input before consequence prediction.
#: pv14-9 (2026-08-25): menu CARD rows gain dynamic ``lscale/rscale``; in the old schema
#:   query/StateSnapshot already read the scales, but the writer dropped the column and the model could not see it.
#: pv14-8 (2026-08-24): menu rows gain the ban-list hash and the per-card 0/1/2 limit; the model consumes
#:   the generalizable per-card content, and the hash only checks source / online consistency.
#: pv14-7 (2026-08-24): sub-answers gain ``stage`` / ``link``, telling cost/target choices in the activation procedure
#:   from choices the operation coroutine asks at resolution; the latter enter the token stream only after the matching link's
#:   ``MSG_CHAIN_SOLVING``, so they cannot leak to higher links.
#: pv14-6 (2026-08-24): sub-answers within a resolution gain player/code/location/value; settle rows gain the action's
#: code and location; a public disclosure stream with an audience is added. Seat sequences can therefore consume the opponent's public actions,
#: random results, reveals and declarations, while refusing confirmations of the raw core that belong only to the other seat.
#: pv14-3 (2026-08-23) relative to pv14-2: the settle table gains `mv_link` / `pc_link` and explicit
#:   `dr_player` / `dr_count` / `dr_link`, splitting a whole chain's deltas back per link.
#:   Attribution rule = the window from `MSG_CHAIN_SOLVING(k)` to `MSG_CHAIN_SOLVED(k)`;
#:   `MSG_CHAIN_NEGATED` is not a boundary (it is sent while another effect resolves).
#:   `ch_kind` gains the value 4 = solving.
#: pv14-2 (2026-08-23), three changes relative to pv14-1:
#:   * ``ORIGIN_LOCATIONS[SUMMON]`` gains ``LOCATION_MZONE``;
#:   * the settle table gains ``sub_choice`` / ``sub_desc``, so a resolution is a function of
#:     (state, action, sub-answers);
#:   * the settle table gains six ``ch_*`` columns recording the chain structure within the interval and per-link negation marks.

__all__ = ["SCHEMA_VERSION"]
