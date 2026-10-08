"""Hidden-target deferral: which activation defers, what the chain made public, and where it sat."""

import struct
from types import SimpleNamespace

from mirrorforce.netduel import constants as C
from mirrorforce.common.client_hidden_target import activation, origin_slots, revealed

OPPONENT = 1
SHIZUKU = 90673288


def chaining(code, controller=OPPONENT, location=C.LOCATION_MZONE, sequence=0):
    return bytes([C.MSG_CHAINING]) + struct.pack("<I", code) + bytes([controller, location, sequence, 0]) + bytes(8)


def move(code, before, after):
    return bytes([C.MSG_MOVE]) + struct.pack("<I", code) + bytes([*before, 0, *after, 0]) + struct.pack("<I", 0)


def confirm(player, cards):
    body = b"".join(struct.pack("<I", code) + bytes(place) for code, place in cards)
    return bytes([C.MSG_CONFIRM_CARDS, player, 0, len(cards)]) + body


def own_chain_prompt():
    return bytes([C.MSG_SELECT_CHAIN, 0]) + bytes(12)


def follower(packets):
    return SimpleNamespace(viewer=0, packets=list(packets), deferred_action=None, own=[])


def test_an_activation_already_found_to_need_no_hidden_card_is_not_deferred_again():
    from mirrorforce.common.client_hidden_target import begin
    packets = [bytes([C.MSG_WAITING]), chaining(SHIZUKU), own_chain_prompt()]
    fresh = SimpleNamespace(**{**vars(follower(packets)), "hidden_target_skip": None})
    assert begin(fresh, (None, None, 0, 0), 0).chaining_index == 1
    skipped = SimpleNamespace(**{**vars(follower(packets)), "hidden_target_skip": 1})
    assert begin(skipped, (None, None, 0, 0), 0) is None


def test_only_an_opponent_public_activation_before_our_next_prompt_defers():
    packets = [bytes([C.MSG_HINT, 1, 0]), chaining(SHIZUKU), own_chain_prompt()]
    assert activation(follower(packets), 0) == (1, (OPPONENT, C.LOCATION_MZONE, 0, SHIZUKU))
    assert activation(follower([own_chain_prompt(), *packets]), 0) is None
    assert activation(follower([chaining(SHIZUKU, controller=0)]), 0) is None
    assert activation(follower([chaining(0)]), 0) is None


def test_the_chain_reveals_moves_with_codes_and_confirmed_hidden_moves_until_it_ends():
    searched = move(SHIZUKU, (OPPONENT, C.LOCATION_DECK, 7), (OPPONENT, C.LOCATION_HAND, 3))
    set_face_down = move(0, (OPPONENT, C.LOCATION_HAND, 1), (OPPONENT, C.LOCATION_SZONE, 2))
    shown = confirm(0, [(12345678, (OPPONENT, C.LOCATION_SZONE, 2))])
    ours = move(55555555, (0, C.LOCATION_HAND, 0), (0, C.LOCATION_GRAVE, 0))
    # An earlier chain's end and our own prompt precede the activation; the activated spell left the hand.
    activated = move(35726888, (OPPONENT, C.LOCATION_HAND, 2), (OPPONENT, C.LOCATION_SZONE, 0))
    before = [bytes([C.MSG_CHAIN_END]), activated]
    packets = [*before, chaining(35726888, location=C.LOCATION_SZONE), searched, ours, set_face_down, shown,
               bytes([C.MSG_CHAIN_END])]
    state = SimpleNamespace(origin_cursor=0, chaining_index=2)
    assert revealed(follower(packets[:-1]), state) is None
    assert revealed(follower(packets), state) == [(1, C.LOCATION_HAND, 2, 35726888), (3, C.LOCATION_DECK, 7, SHIZUKU),
                                                  (5, C.LOCATION_HAND, 1, 12345678)]
    # Only what the batch received counts.
    assert revealed(follower(packets), SimpleNamespace(origin_cursor=2, chaining_index=2))[0][0] == 3
    # A chain that made nothing hidden public ends with no reveal; the batch is then followed the ordinary way.
    assert revealed(follower([bytes([C.MSG_CHAIN_END]), chaining(SHIZUKU), ours, bytes([C.MSG_CHAIN_END])]),
                    SimpleNamespace(origin_cursor=0, chaining_index=1)) == []
    # The duel was won while the chain resolved: no chain end comes, and no later packet can show more.
    assert revealed(follower([*packets[:-1], bytes([C.MSG_WIN, 0, 1])]), state) == revealed(follower(packets), state)


def test_reveals_trace_back_through_draws_and_public_moves_to_their_batch_start_slots():
    discard = move(44444444, (OPPONENT, C.LOCATION_HAND, 0), (OPPONENT, C.LOCATION_GRAVE, 0))
    packets = [bytes([C.MSG_DRAW, OPPONENT, 1]) + struct.pack("<I", 0), discard, chaining(SHIZUKU),
               move(SHIZUKU, (OPPONENT, C.LOCATION_DECK, 2), (OPPONENT, C.LOCATION_HAND, 3)),
               move(22222222, (OPPONENT, C.LOCATION_HAND, 1), (OPPONENT, C.LOCATION_GRAVE, 1))]
    reveals = [(3, C.LOCATION_DECK, 2, SHIZUKU), (4, C.LOCATION_HAND, 1, 22222222)]
    sizes = {C.LOCATION_HAND: 3, C.LOCATION_DECK: 5, C.LOCATION_EXTRA: 0}
    state = SimpleNamespace(origin_cursor=0)
    # The draw took deck slot 4; the discard shifted hand slots 1 and 2 down by one.
    assert origin_slots(follower(packets), state, reveals, sizes) == [
        (OPPONENT, C.LOCATION_DECK, 2, SHIZUKU), (OPPONENT, C.LOCATION_HAND, 2, 22222222)]
    # A drawn card has no slot at the batch start, and a shuffle loses the trace: neither card took part in what
    # the batch start allowed, so its entry is None and the replay's own reveal fixes place it.
    drawn = [(4, C.LOCATION_HAND, 2, 22222222)]
    assert origin_slots(follower(packets[:4] + [move(22222222, (OPPONENT, C.LOCATION_HAND, 2),
                                                     (OPPONENT, C.LOCATION_GRAVE, 1))]), state, drawn, sizes) == [None]
    shuffled = [packets[0], bytes([C.MSG_SHUFFLE_DECK, OPPONENT]), *packets[1:]]
    assert origin_slots(follower(shuffled), state, [(4, C.LOCATION_DECK, 2, SHIZUKU)], sizes) == [None]


def _category_follower(monkeypatch, zones, offers):
    """A follower whose opponent zones are ``zones`` ({location: [code, ...]}); ``offers(cards)`` stands for the
    proof run that tells whether the window offers the activation with those cards placed."""
    from mirrorforce.common import client_hidden_target, client_shadow
    monkeypatch.setattr(client_shadow, "card_code",
                        lambda core, pduel, player, location, sequence: zones[location][sequence])
    trials = []

    def offered(follower, origin, cards, code, description=None):
        assert description is None  # the stub's chain packet is no link: any effect of the card
        # Each trial holds its own candidate as a category fact of the activation while it runs.
        if cards:
            assert follower.category_constraints[-1] == (cards[-1][0], cards[-1][1], (cards[-1][3],), 1, 40,
                                                         ((cards[-1][1], (cards[-1][3],)),), origin[2])
        trials.append(tuple(cards))
        return offers(cards)

    monkeypatch.setattr(client_hidden_target, "_offered", offered)
    kinds = {SPELL: 0x2, LINK: 0x1 | 0x4000000, OTHER_SPELL: 0x2}
    return SimpleNamespace(
        viewer=0, local=SimpleNamespace(pduel=1), category_constraints=[], hidden_target_pool=(SPELL, LINK, OTHER_SPELL),
        packets=[bytes([C.MSG_WAITING])] * 41,
        core=SimpleNamespace(query_field_count=lambda pduel, player, location: len(zones.get(location, ())),
                             card_pool=lambda: SimpleNamespace(cards={code: SimpleNamespace(type=kind)
                                                                      for code, kind in kinds.items()}))), trials


SPELL, LINK, OTHER_SPELL = 9726840, 63288573, 99550630


def test_an_activation_legal_only_with_an_unseen_extra_deck_card_names_its_candidates_there(monkeypatch):
    from mirrorforce.common.client_hidden_target import _category
    from mirrorforce.common.client_shadow import BLANK_EXTRA, BLANK_MAIN
    zones = {C.LOCATION_DECK: [SPELL, BLANK_MAIN], C.LOCATION_EXTRA: [BLANK_EXTRA], C.LOCATION_HAND: [BLANK_MAIN]}
    follower, trials = _category_follower(monkeypatch, zones, lambda cards: any(card[3] == LINK for card in cards))
    # The deck's known card and the extra-deck kind decide which candidates each zone's blank is tried with.
    assert _category(follower, _state(38, 3), [], SHIZUKU, 40) == (OPPONENT, C.LOCATION_EXTRA, 0, (LINK,),
                                                          ((C.LOCATION_EXTRA, (LINK,)),))
    assert trials == [(), ((OPPONENT, C.LOCATION_DECK, 1, SPELL),), ((OPPONENT, C.LOCATION_DECK, 1, OTHER_SPELL),),
                      ((OPPONENT, C.LOCATION_EXTRA, 0, LINK),),
                      ((OPPONENT, C.LOCATION_HAND, 0, SPELL),), ((OPPONENT, C.LOCATION_HAND, 0, OTHER_SPELL),)]
    assert follower.category_constraints == []


def test_candidates_in_two_hidden_zones_make_a_fact_of_either_zone_with_the_proxy_in_the_first(monkeypatch):
    from mirrorforce.common.client_hidden_target import _category
    from mirrorforce.common.client_shadow import BLANK_MAIN
    zones = {C.LOCATION_DECK: [BLANK_MAIN], C.LOCATION_HAND: [BLANK_MAIN]}
    # A Tuner summoned "from the hand or Deck": either zone holding one makes the activation legal.
    follower, _trials = _category_follower(monkeypatch, zones, lambda cards: any(card[3] == SPELL for card in cards))
    assert _category(follower, _state(38, 3), [], SHIZUKU, 40) == (
        OPPONENT, C.LOCATION_DECK, 0, (SPELL,), ((C.LOCATION_DECK, (SPELL,)), (C.LOCATION_HAND, (SPELL,))))
    assert follower.category_constraints == []
    # Offered without any unseen card: no category at all.
    follower, trials = _category_follower(monkeypatch, zones, lambda cards: True)
    assert _category(follower, _state(38, 3), [], SHIZUKU, 40) is None and trials == [()]


def test_a_category_fact_keeps_every_zone_of_its_fact_and_writes_nothing(monkeypatch):
    import pytest
    from mirrorforce.common.client_shadow import BlankClientSync
    from mirrorforce.common.client_sync import SyncError, SyncStats
    follower = SimpleNamespace(category_constraints=[], stats=SyncStats())
    either = ((C.LOCATION_DECK, (SPELL, OTHER_SPELL)), (C.LOCATION_HAND, (SPELL,)))
    BlankClientSync._category_fact(follower, OPPONENT, C.LOCATION_DECK, (SPELL, OTHER_SPELL), 40, either, 38)
    assert follower.category_constraints == [(OPPONENT, C.LOCATION_DECK, (SPELL, OTHER_SPELL), 1, 40, either, 38)]
    assert follower.stats.categories_open == 1
    with pytest.raises(SyncError, match="outside its first zone"):
        BlankClientSync._category_fact(follower, OPPONENT, C.LOCATION_HAND, (SPELL,), 40, either, 38)


def test_a_chain_prompt_offers_only_its_links_not_a_reset_naming_the_same_card():
    from mirrorforce.puzzle.messages import Message
    from mirrorforce.common.client_hidden_target import _offers

    def entry(kind, code):
        return bytes([kind, kind != 0]) + struct.pack("<I", code) + bytes([0, C.LOCATION_MZONE, 3, 1]) + bytes(4)

    def prompt(*entries):
        return Message(C.MSG_SELECT_CHAIN, bytes([0, len(entries), 0]) + bytes(8) + b"".join(entries))

    maxx = 23434538
    # The End Phase returns a lent monster (a reset entry); a hand trap beside it makes the core ask.
    assert not _offers(prompt(entry(2, SHIZUKU), entry(0, maxx)), SHIZUKU)
    assert _offers(prompt(entry(2, SHIZUKU), entry(0, maxx)), maxx)
    assert _offers(prompt(entry(0, SHIZUKU)), SHIZUKU)


def test_an_offer_names_the_activated_effect_not_only_the_card():
    from mirrorforce.puzzle.messages import Message
    from mirrorforce.common.client_hidden_target import _offers

    def record(code, desc):
        return struct.pack("<I", code) + bytes([OPPONENT, C.LOCATION_HAND, 0]) + struct.pack("<I", desc)

    # A card with two effects: the menu offering one of them does not offer the other.
    menu = Message(C.MSG_SELECT_BATTLECMD, bytes([OPPONENT, 1]) + record(SHIZUKU, SHIZUKU << 4 | 1) + bytes([0, 0, 0]))
    assert _offers(menu, SHIZUKU, SHIZUKU << 4 | 1) and _offers(menu, SHIZUKU)
    assert not _offers(menu, SHIZUKU, SHIZUKU << 4)


def test_the_packets_from_an_index_that_go_straight_to_an_opponent_chain_link_are_where_a_hidden_target_may_wait():
    from mirrorforce.common.client_shadow import BlankClientSync
    hint = bytes([C.MSG_HINT, 1, OPPONENT]) + bytes(4)
    packets = [bytes([C.MSG_WAITING]), hint, chaining(SHIZUKU), bytes([C.MSG_WAITING]), bytes([C.MSG_NEW_PHASE, 0, 2])]
    follower = SimpleNamespace(packets=packets, viewer=0)
    # Prompts and hints aside (which WAITING is whose is not known): the real opponent activated a card.
    assert BlankClientSync._link_at(follower, 0) and BlankClientSync._link_at(follower, 2)
    assert not BlankClientSync._link_at(follower, 3)  # the phase moved on
    follower.viewer = OPPONENT
    assert not BlankClientSync._link_at(follower, 0)  # the seat's own link


def _state(cursor, answered):
    """A follower state as the follower holds it: (snapshot, python state, cursor, answered)."""
    return (object(), None, cursor, answered)


def test_the_construction_candidates_are_the_older_batch_starts_held_back_to_the_oldest_answer_state():
    from mirrorforce.common.client_shadow import BlankClientSync
    # Three opponent prompts in a row start their batches at one cursor with one answer count: taken in turn.
    before, taking, window = _state(88, 5), _state(88, 5), _state(88, 5)
    first = _state(90, 5)
    choices = [SimpleNamespace(origin=_state(40, 2)), SimpleNamespace(origin=_state(84, 5)),
               SimpleNamespace(origin=None), SimpleNamespace(origin=before), SimpleNamespace(origin=taking),
               SimpleNamespace(origin=window)]
    answer_states = [_state(50, 3), _state(70, 4), _state(80, 5)]
    own_origin = _state(60, 3)
    loaded = []
    follower = SimpleNamespace(choices=choices, own_origin=own_origin, answer_states=answer_states,
                               _load=loaded.append, _save=lambda: loaded[-1])
    copies = BlankClientSync._older_batch_starts(follower, first)
    # Newest first, back to the oldest answer state; ``first`` (the batch start taken last) is not among them.
    assert copies == (window, taking, before, choices[1].origin, answer_states[2], answer_states[1], own_origin,
                      answer_states[0])
    assert loaded[-1] is first  # the follower is left where the deferral began
    # From an opponent prompt's own batch start: only what was taken before it.
    assert BlankClientSync._older_batch_starts(follower, taking) == (before, choices[1].origin, answer_states[2],
                                                                    answer_states[1], own_origin, answer_states[0])


def _construction_follower(monkeypatch):
    """A follower whose category trials and replays are recorded."""
    from mirrorforce.common import client_hidden_target
    from mirrorforce.common.client_hidden_target import Construction, ForcedActivation
    oldest, middle, newest = _state(70, 4), _state(84, 5), _state(90, 5)
    log = []
    monkeypatch.setattr(client_hidden_target, "_replay_from", lambda follower, origin: log.append(("replay", origin)))
    forced = ForcedActivation(100, OPPONENT, SHIZUKU, SHIZUKU << 4)
    # The fact read at the newest batch start (90).
    fact = (OPPONENT, C.LOCATION_DECK, 3, (SPELL,), ((C.LOCATION_DECK, (SPELL,)),), 90)
    follower = SimpleNamespace(construction=Construction(forced, (oldest, middle, newest), fact=fact),
                               hidden_target_pool=(SPELL,),
                               hidden_target_skip=None, origin=None, own_origin=None, choices=[], answer_states=[],
                               _free=lambda state: log.append(("free", state)),
                               _force_activation=lambda state: log.append(("force", state)),
                               _category_fact=lambda *fact: log.append(("fact", fact)))
    return follower, forced, (oldest, middle, newest), log


def test_a_forced_activation_is_replayed_from_each_batch_start_in_turn(monkeypatch):
    import pytest
    from mirrorforce.common.client_hidden_target import Construction, HiddenTargetError, construct
    from mirrorforce.common.client_shadow import BlankClientSync
    follower, forced, (oldest, middle, newest), log = _construction_follower(monkeypatch)
    # Forced from the oldest batch start; the fact the activation's legality gives (read at the newest) is only
    # claimed for the particles, after each replay.
    construct(follower)
    assert log == [("replay", oldest), ("force", forced),
                   ("fact", (OPPONENT, C.LOCATION_DECK, (SPELL,), 100, ((C.LOCATION_DECK, (SPELL,)),), 90))]
    assert follower.construction.cursor == 70 and follower.construction.candidates == (middle, newest)
    # A divergence before the activation is written goes back to the next; after it, the construction is proved.
    follower._constructing = lambda index: BlankClientSync._constructing(follower, index)
    assert follower._constructing(100)
    construct(follower)
    construct(follower)
    assert [entry for entry in log if entry[0] == "replay"] == [("replay", oldest), ("replay", middle),
                                                               ("replay", newest)]
    with pytest.raises(HiddenTargetError):
        construct(follower)
    assert follower.construction is None
    follower.construction = Construction(forced, (middle,), cursor=90)
    assert not follower._constructing(101) and follower.construction is None and log[-1] == ("free", middle)


def test_each_replay_starts_from_the_bookkeeping_the_construction_began_with():
    from mirrorforce.common.client_hidden_target import Construction, ForcedActivation, _replay_from
    started = [SimpleNamespace(begin=begin, saved=_state(begin, answered), origin=_state(begin - 1, answered))
               for begin, answered in ((60, 3), (75, 4), (88, 5))]
    states = (_state(58, 3), _state(72, 4))
    facts = ((OPPONENT, C.LOCATION_DECK, (SPELL,), 1, 65, ()), (OPPONENT, C.LOCATION_DECK, (SPELL,), 1, 80, ()))
    origin = _state(78, 4)
    forced = ForcedActivation(100, OPPONENT, SHIZUKU, SHIZUKU << 4)
    # An older replay given up: it released the choices from 60 on, and read a choice, a state and a fact of its own.
    mine = SimpleNamespace(begin=77, saved=_state(77, 4), origin=_state(76, 4))
    log = []
    follower = SimpleNamespace(construction=Construction(forced, (), tuple(started), states, facts),
                               choices=[mine], answer_states=[states[0], _state(70, 4)], category_constraints=[facts[0]],
                               parked=object(), _drop_origin=lambda: log.append("origin"),
                               _release=lambda choice: log.append(("release", choice.begin)),
                               _free=lambda state: log.append(("free", state[2])),
                               _load=lambda state: log.append(("load", state[2])))
    _replay_from(follower, origin)
    # What the replay given up read goes; the next one has what was held when the construction began, up to 78.
    assert log == ["origin", ("release", 77), ("free", 70), ("load", 78), ("free", 78)]
    assert follower.choices == started[:2] and follower.answer_states == list(states)
    assert follower.category_constraints == [facts[0]] and follower.parked is None


def _forcing_follower(monkeypatch, offered):
    from mirrorforce.common import client_hidden_target
    from mirrorforce.common.client_shadow import BlankClientSync
    monkeypatch.setattr(client_hidden_target, "_offered", lambda follower, origin, cards, code, description=None: offered)
    constructed, read_at = [], []
    monkeypatch.setattr(client_hidden_target, "construct", lambda follower: constructed.append(follower.construction))
    monkeypatch.setattr(client_hidden_target, "_category", lambda follower, origin, placed, code, chaining: (
        read_at.append(origin), (OPPONENT, C.LOCATION_DECK, 3, (SPELL,), ((C.LOCATION_DECK, (SPELL,)),)))[1])
    packets = [bytes([C.MSG_WAITING])] * 100 + [chaining(SHIZUKU), own_chain_prompt()]
    follower = SimpleNamespace(packets=packets, viewer=0, deferred_action=None, hidden_target_skip=None,
                               constructions_tried=frozenset(), construction=None, freed=[], choices=[],
                               answer_states=[], category_constraints=[], hidden_target_pool=(SPELL,), read_at=read_at)
    older = (_state(84, 5), _state(70, 4))  # newest first, as the follower holds them
    follower._older_batch_starts = lambda first: older
    follower._free = follower.freed.append
    follower._force_construction = lambda first, index: BlankClientSync._force_construction(follower, first, index)
    return follower, older, constructed


def test_a_forced_activation_is_constructed_oldest_batch_start_first_and_never_twice_from_it(monkeypatch):
    import pytest
    from mirrorforce.common.client_hidden_target import Reconstructed
    from mirrorforce.common.client_sync import SyncError
    follower, older, constructed = _forcing_follower(monkeypatch, offered=False)
    first = _state(90, 5)
    with pytest.raises(Reconstructed):
        follower._force_construction(first, 99)
    assert constructed[0].candidates == (older[1], older[0], first)
    # The fact the activation's legality gives is read once, at the newest batch start: its first opponent window
    # is the activation's.
    assert follower.read_at == [first]
    assert constructed[0].fact == (OPPONENT, C.LOCATION_DECK, 3, (SPELL,), ((C.LOCATION_DECK, (SPELL,)),), 90)
    # Met again from the same oldest batch start: no window the stream allows took it.
    with pytest.raises(SyncError, match="forced again"):
        follower._force_construction(_state(90, 5), 99)
    assert follower.freed == list(older)


def test_an_activation_the_newest_batch_start_offers_unforced_is_the_ordinary_followers(monkeypatch):
    follower, _older, constructed = _forcing_follower(monkeypatch, offered=True)
    # Its legality waits for no hidden card: nothing is constructed, the divergence is raised to the ordinary
    # follower, and the activation is not constructed a second time.
    assert follower._force_construction(_state(90, 5), 99) is False
    assert constructed == [] and follower.hidden_target_skip == 100
    assert follower._force_construction(_state(90, 5), 99) is False


def test_a_category_fact_reaches_the_root_until_a_card_it_may_speak_of_leaves_its_zones():
    from mirrorforce.common.client_hidden_target import root_claims
    from mirrorforce.common.stage_a_joint_belief_runtime import ZoneClaim
    deck_only = ((C.LOCATION_DECK, (SPELL, OTHER_SPELL)),)
    either = ((C.LOCATION_DECK, (SPELL,)), (C.LOCATION_HAND, (SPELL,)))

    def claims(packets, facts, cursor=None, since=None):
        rows = [(OPPONENT, zones[0][0], zones[0][1], 1, chain, zones, chain if since is None else since)
                for chain, zones in facts]
        return root_claims({"packets": packets, "cursor": len(packets) if cursor is None else cursor,
                            "category_constraints": rows}, 0)

    quiet = [chaining(SHIZUKU), bytes([C.MSG_CHAIN_END])]
    assert claims(quiet, [(0, deck_only)]) == (ZoneClaim((C.LOCATION_DECK,), frozenset({SPELL, OTHER_SPELL})),)
    # A card of another name leaving the deck in the open says nothing about the candidates.
    milled = quiet + [move(LINK, (OPPONENT, C.LOCATION_DECK, 3), (OPPONENT, C.LOCATION_GRAVE, 0))]
    assert len(claims(milled, [(0, deck_only)])) == 1
    # An unseen card or a candidate leaving it may be the one: the fact is gone.
    unseen = quiet + [move(0, (OPPONENT, C.LOCATION_DECK, 3), (OPPONENT, C.LOCATION_SZONE, 1))]
    shown = quiet + [move(SPELL, (OPPONENT, C.LOCATION_DECK, 3), (OPPONENT, C.LOCATION_HAND, 0))]
    drawn = quiet + [bytes([C.MSG_DRAW, OPPONENT, 1]) + bytes(4)]
    assert claims(unseen, [(0, deck_only)]) == claims(shown, [(0, deck_only)]) == claims(drawn, [(0, deck_only)]) == ()
    # A draw keeps "the deck or the hand"; a move between its zones too.
    assert len(claims(drawn, [(0, either)])) == len(claims(shown, [(0, either)])) == 1
    # An activation after the root, and the seat's own facts, say nothing here.
    assert claims(quiet, [(1, deck_only)], cursor=1) == ()
    assert root_claims({"packets": quiet, "cursor": 2, "category_constraints": [(0, 1, (SPELL,), 1, 0, deck_only, 0)]},
                       0) == ()
    # A fact speaks of its zones from the batch start it was read at: a spell activated from the hand leaves it before
    # its chain link shows, and the fact that it was there ends with it (random seat 1204/0).
    hand_only = ((C.LOCATION_HAND, (SPELL,)),)
    activated = [move(SPELL, (OPPONENT, C.LOCATION_HAND, 1), (OPPONENT, C.LOCATION_SZONE, 2)), chaining(SPELL),
                 bytes([C.MSG_CHAIN_END])]
    assert claims(activated, [(1, hand_only)], since=0) == ()
    assert len(claims(activated, [(1, hand_only)], since=1)) == 1
