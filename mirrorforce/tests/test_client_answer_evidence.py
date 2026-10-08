"""Opponent answers read from the received packets: which option each local prompt's records show was taken."""

import struct

from mirrorforce.netduel import constants as C
from mirrorforce.common.client_answer_evidence import read

VIEWER, OPPONENT = 0, 1
BLANK = 999000001
X, Y, Z = 26077387, 63166095, 98338152


def card(code, place, extra=b""):
    return struct.pack("<I", code) + bytes(place) + extra


def idle(summon=(), spsummon=(), repos=(), mset=(), sset=(), activate=(), bp=1, ep=1, shuffle=0):
    body = bytes([OPPONENT])
    for group in (summon, spsummon, repos, mset, sset):
        body += bytes([len(group)]) + b"".join(card(code, place) for code, place in group)
    body += bytes([len(activate)]) + b"".join(card(code, place, struct.pack("<I", desc)) for code, place, desc in activate)
    return body + bytes([bp, ep, shuffle])


def move(code, before, after, reason=0):
    return bytes([C.MSG_MOVE]) + struct.pack("<I", code) + bytes([*before, 0x0a, *after, 0x05]) + struct.pack("<I", reason)


def chaining(code, place, desc):
    return bytes([C.MSG_CHAINING]) + struct.pack("<I", code) + bytes([*place, 5, *place]) + struct.pack("<I", desc) + b"\x01"


def summoning(code, place):
    return bytes([C.MSG_SUMMONING]) + struct.pack("<I", code) + bytes([*place, 1])


def own_chain_prompt():
    return bytes([C.MSG_SELECT_CHAIN, VIEWER, 0, 0]) + bytes(8)


WAIT = bytes([C.MSG_WAITING])


def answer(msg, payload, after):
    return read(msg, payload, [WAIT, *after], 0, VIEWER)


def test_idle_commands_are_read_from_the_summon_the_activation_the_set_and_the_phase():
    menu = idle(summon=[(X, (OPPONENT, C.LOCATION_HAND, 2))], sset=[(BLANK, (OPPONENT, C.LOCATION_HAND, 1))],
                activate=[(Y, (OPPONENT, C.LOCATION_HAND, 0), Y << 4), (Y, (OPPONENT, C.LOCATION_HAND, 0), Y << 4 | 1)])
    summon = answer(C.MSG_SELECT_IDLECMD, menu, [move(X, (OPPONENT, C.LOCATION_HAND, 2), (OPPONENT, C.LOCATION_MZONE, 3)),
                                                 summoning(X, (OPPONENT, C.LOCATION_MZONE, 3))])
    assert summon.answers == (struct.pack("<i", 0 << 16 | 0),)
    # A spell moves from the hand before its chaining: the second effect of the card that left hand slot 0.
    spell = answer(C.MSG_SELECT_IDLECMD, menu, [move(Y, (OPPONENT, C.LOCATION_HAND, 0), (OPPONENT, C.LOCATION_SZONE, 1)),
                                                chaining(Y, (OPPONENT, C.LOCATION_SZONE, 1), Y << 4 | 1)])
    assert spell.answers == (struct.pack("<i", 1 << 16 | 5),)
    # A face-down set from the hand shows no code: the blank that could be set.
    setting = answer(C.MSG_SELECT_IDLECMD, menu, [move(0, (OPPONENT, C.LOCATION_HAND, 1), (OPPONENT, C.LOCATION_SZONE, 2)),
                                                  bytes([C.MSG_SET]) + bytes(4) + bytes([OPPONENT, C.LOCATION_SZONE, 2, 0x0a])])
    assert setting.answers == (struct.pack("<i", 0 << 16 | 4),)
    # A hidden hand card set in the spell/trap zones where the local menu offers no set: the set proxy goes there.
    no_set = idle(summon=[(X, (OPPONENT, C.LOCATION_HAND, 2))])
    proxy = answer(C.MSG_SELECT_IDLECMD, no_set, [move(0, (OPPONENT, C.LOCATION_HAND, 0), (OPPONENT, C.LOCATION_SZONE, 2)),
                                                  bytes([C.MSG_SET]) + bytes(4) + bytes([OPPONENT, C.LOCATION_SZONE, 2, 0x0a])])
    assert proxy.placements == ((OPPONENT, C.LOCATION_HAND, 0, 999000003),)
    # Closing the Main Phase writes its end hint, then the seat's chain window; the phase entered after that
    # window tells the Battle Phase from the End Phase, so before it both answers stay open.
    late = answer(C.MSG_SELECT_IDLECMD, menu, [event_hint(23), own_chain_prompt()])
    assert late.answers == (struct.pack("<i", 6), struct.pack("<i", 7))
    ended = answer(C.MSG_SELECT_IDLECMD, menu, [event_hint(23), new_phase(0x200)])
    assert ended.answers == (struct.pack("<i", 7),)
    battle = answer(C.MSG_SELECT_IDLECMD, menu, [event_hint(23), new_phase(0x08)])
    assert battle.answers == (struct.pack("<i", 6),)
    # A phase change without that window is no phase answer of this command.
    assert answer(C.MSG_SELECT_IDLECMD, menu, [new_phase(0x200)]).answers == ()


def menu_without_phase_hint():
    return idle(summon=[(X, (OPPONENT, C.LOCATION_HAND, 2))])


def event_hint(value):
    return bytes([C.MSG_HINT, C.HINT_EVENT, VIEWER]) + struct.pack("<I", value)


def new_phase(phase):
    return bytes([C.MSG_NEW_PHASE]) + struct.pack("<H", phase)


def test_the_battle_phase_exit_shows_where_main_phase_2_begins():
    menu = bytes([OPPONENT, 0, 0, 1, 1])  # no activation, no attacker, Main Phase 2 and End Phase offered
    m2, ep = struct.pack("<i", 2), struct.pack("<i", 3)
    # Both exits write the Battle Step's end hint and the seat's chain window, then enter Main Phase 2.
    assert answer(C.MSG_SELECT_BATTLECMD, menu, [event_hint(29), own_chain_prompt()]).answers == (m2, ep)
    assert answer(C.MSG_SELECT_BATTLECMD, menu, [event_hint(29), new_phase(0x100)]).answers == (m2, ep)
    # Main Phase 2 then asks the turn player, unless the End Phase was chosen: straight to its end hint. Which
    # prompts the opponent was asked shows nothing (the server asks it prompts the local duel never asks): the end
    # hint next is the End Phase (or Main Phase 2 closed at once, the same duel), an action there Main Phase 2.
    assert answer(C.MSG_SELECT_BATTLECMD, menu, [event_hint(29), new_phase(0x100), WAIT]).answers == (m2, ep)
    assert answer(C.MSG_SELECT_BATTLECMD, menu, [event_hint(29), new_phase(0x100), WAIT, event_hint(23)]).answers \
        == (ep,)
    assert answer(C.MSG_SELECT_BATTLECMD, menu, [event_hint(29), new_phase(0x100), WAIT,
                                                 summoning(X, (OPPONENT, C.LOCATION_MZONE, 2))]).answers == (m2,)
    # A second Battle Phase: either exit leads to the same duel.
    assert answer(C.MSG_SELECT_BATTLECMD, menu, [event_hint(29), new_phase(0x08)]).answers == (m2,)

    # An attack declared and cancelled at its target selection leaves nothing behind: the command asked again closed
    # the Battle Step, which is what this one does.
    attacker = bytes([OPPONENT, 0, 1]) + struct.pack("<I", X) + bytes([OPPONENT, C.LOCATION_MZONE, 5, 0]) + bytes([1, 1])
    for after in ([WAIT, WAIT, event_hint(29)], [WAIT, event_hint(29)], [event_hint(29)]):
        assert answer(C.MSG_SELECT_BATTLECMD, attacker, after).answers == (m2, ep)
    targets = bytes([OPPONENT, 1, 1, 1, 1]) + struct.pack("<I", Y) + bytes([VIEWER, C.LOCATION_MZONE, 2, 5])
    assert answer(C.MSG_SELECT_CARD, targets, [WAIT, event_hint(29)]).answers == (struct.pack("<i", -1),)
    # Not cancelable: no reading.
    assert answer(C.MSG_SELECT_CARD, bytes([OPPONENT, 0]) + targets[2:], [WAIT, event_hint(29)]).answers == ()


def test_an_evidenced_card_the_local_menu_lacks_asks_for_its_identity_first():
    menu = idle(activate=[(BLANK, (OPPONENT, C.LOCATION_HAND, 0), 0)])
    reading = answer(C.MSG_SELECT_IDLECMD, menu, [chaining(Z, (OPPONENT, C.LOCATION_HAND, 0), Z << 4)])
    assert reading.answers == () and reading.placements == ((OPPONENT, C.LOCATION_HAND, 0, Z),)


def chain_prompt(*records):
    """``(code, place, desc, forced)`` records, optionally with their kind last (1: an operation, 2: a reset)."""
    body = bytes([OPPONENT, len(records), 0]) + bytes(8)
    for code, place, desc, forced, *kind in records:
        body += bytes([kind[0] if kind else 0, forced]) + struct.pack("<I", code) + bytes([*place, 0x0a]) \
            + struct.pack("<I", desc)
    return body


def test_a_chain_answer_is_the_chaining_that_follows_or_a_pass():
    prompt = chain_prompt((Y, (OPPONENT, C.LOCATION_SZONE, 2), Y << 4, 0), (X, (OPPONENT, C.LOCATION_MZONE, 0), X << 4, 0))
    flipped = answer(C.MSG_SELECT_CHAIN, prompt, [
        bytes([C.MSG_POS_CHANGE]) + struct.pack("<I", Y) + bytes([OPPONENT, C.LOCATION_SZONE, 2, 0x0a, 0x05]),
        chaining(Y, (OPPONENT, C.LOCATION_SZONE, 2), Y << 4)])
    # A set card turns face up by its own activation. The one WAITING before it is this prompt's: chained here.
    assert flipped.answers == (struct.pack("<i", 0),) and not flipped.open
    monster = answer(C.MSG_SELECT_CHAIN, prompt, [chaining(X, (OPPONENT, C.LOCATION_MZONE, 0), X << 4)])
    assert monster.answers == (struct.pack("<i", 1),) and not monster.open
    # A second WAITING before the link: chained here after a window a hidden card opened, or passed here and chained
    # in a later window of the same timing; only the consequences tell, and both stay open.
    later = answer(C.MSG_SELECT_CHAIN, prompt, [WAIT, chaining(X, (OPPONENT, C.LOCATION_MZONE, 0), X << 4)])
    assert later.answers == (struct.pack("<i", 1), struct.pack("<i", -1)) and later.by_consequence and later.open
    passed = answer(C.MSG_SELECT_CHAIN, prompt, [bytes([C.MSG_CHAIN_SOLVING, 1])])
    assert passed.answers == (struct.pack("<i", -1),)
    forced = chain_prompt((Y, (OPPONENT, C.LOCATION_SZONE, 2), Y << 4, 1))
    assert answer(C.MSG_SELECT_CHAIN, forced, [bytes([C.MSG_CHAIN_SOLVING, 1])]).answers == ()
    # A link this window does not offer came from a later prompt (a pass here, to be proved by its consequences), or
    # needs a hidden card; a window with a forced entry cannot have been passed.
    elsewhere = chaining(Z, (OPPONENT, C.LOCATION_SZONE, 4), Z << 4)
    unoffered = answer(C.MSG_SELECT_CHAIN, prompt, [WAIT, elsewhere])
    assert unoffered.answers == (struct.pack("<i", -1),) and unoffered.by_consequence \
        and unoffered.placements == ((OPPONENT, C.LOCATION_SZONE, 4, Z),)
    assert answer(C.MSG_SELECT_CHAIN, forced, [WAIT, elsewhere]).answers == ()


def test_a_forced_entry_that_chains_nothing_is_taken_when_no_chain_link_follows():
    # The End Phase returns a monster whose control was lent until then: a forced reset entry beside an optional
    # trigger. No link follows, so the reset was taken (a forced entry may not be passed).
    lent = (X, (OPPONENT, C.LOCATION_MZONE, 0), 0, 1, 2)
    trigger = (Y, (OPPONENT, C.LOCATION_MZONE, 1), Y << 4, 0)
    prompt = chain_prompt(trigger, lent)
    reading = answer(C.MSG_SELECT_CHAIN, prompt, [bytes([C.MSG_HINT, 3, VIEWER]) + struct.pack("<I", X)])
    assert reading.answers == (struct.pack("<i", 1),) and not reading.by_consequence
    assert answer(C.MSG_SELECT_CHAIN, prompt, [chaining(Y, (OPPONENT, C.LOCATION_MZONE, 1), Y << 4)]).answers \
        == (struct.pack("<i", 0),)
    # Two such entries: which one the engine tells from their own consequences.
    operation = (Z, (OPPONENT, C.LOCATION_SZONE, 0), Z << 4, 1, 1)
    both = answer(C.MSG_SELECT_CHAIN, chain_prompt(lent, operation), [bytes([C.MSG_CHAIN_SOLVING, 1])])
    assert both.answers == (struct.pack("<i", 0), struct.pack("<i", 1)) and both.by_consequence


def select_card(lowest, highest, *cards):
    return bytes([OPPONENT, 0, lowest, highest, len(cards)]) + b"".join(
        struct.pack("<I", code) + bytes([*place, 0x05]) for code, place in cards)


def test_selected_cards_are_the_ones_the_next_operation_acts_on():
    graveyard = [(X, (OPPONENT, C.LOCATION_GRAVE, index)) for index in range(5)]
    targets = answer(C.MSG_SELECT_CARD, select_card(1, 3, *graveyard), [
        bytes([C.MSG_BECOME_TARGET, 2]) + bytes([OPPONENT, C.LOCATION_GRAVE, 4, 5]) + bytes([OPPONENT, C.LOCATION_GRAVE, 1, 5])])
    assert targets.answers == (bytes([2, 4, 1]),)
    # Two cards banished one after the other: the second move names its slot after the first left.
    banished = answer(C.MSG_SELECT_CARD, select_card(2, 2, *graveyard), [
        move(X, (OPPONENT, C.LOCATION_GRAVE, 1), (OPPONENT, C.LOCATION_REMOVED, 0)),
        move(X, (OPPONENT, C.LOCATION_GRAVE, 2), (OPPONENT, C.LOCATION_REMOVED, 1))])
    assert banished.answers == (bytes([2, 1, 3]),)
    # A deck card searched face down and then confirmed: the confirmation names it; deck cards are listed by a
    # running number, so the identity, not the deck slot, picks it.
    deck = [(BLANK, (OPPONENT, C.LOCATION_DECK, 0)), (Z, (OPPONENT, C.LOCATION_DECK, 1))]
    searched = answer(C.MSG_SELECT_CARD, select_card(1, 1, *deck), [
        move(0, (OPPONENT, C.LOCATION_DECK, 18), (OPPONENT, C.LOCATION_HAND, 1)),
        bytes([C.MSG_CONFIRM_CARDS, VIEWER, 0, 1]) + struct.pack("<I", Z) + bytes([OPPONENT, C.LOCATION_HAND, 1])])
    assert searched.answers == (bytes([1, 1]),)
    # A shown identity the local deck does not hold yet goes to the deck slot first.
    missing = answer(C.MSG_SELECT_CARD, select_card(1, 1, deck[0]), [
        move(Y, (OPPONENT, C.LOCATION_DECK, 7), (OPPONENT, C.LOCATION_HAND, 1))])
    assert missing.placements == ((OPPONENT, C.LOCATION_DECK, 7, Y),)
    # An attack target: the attack names the chosen monster.
    monsters = [(X, (VIEWER, C.LOCATION_MZONE, 1)), (X, (VIEWER, C.LOCATION_MZONE, 4))]
    attacked = answer(C.MSG_SELECT_CARD, select_card(1, 1, *monsters), [
        bytes([C.MSG_ATTACK, OPPONENT, C.LOCATION_MZONE, 2, 1, VIEWER, C.LOCATION_MZONE, 4, 8])])
    assert attacked.answers == (bytes([1, 1]),)
    # An operation on cards the prompt did not list is no reading.
    elsewhere = answer(C.MSG_SELECT_CARD, select_card(1, 1, *graveyard), [
        move(X, (OPPONENT, C.LOCATION_HAND, 0), (OPPONENT, C.LOCATION_GRAVE, 5))])
    assert elsewhere.answers == () and elsewhere.reason


def test_the_place_and_the_position_come_from_the_moved_card():
    place = answer(C.MSG_SELECT_PLACE, bytes([OPPONENT, 1]) + struct.pack("<I", 0xFFFFE0FF & ~(1 << 11)), [
        move(Y, (OPPONENT, C.LOCATION_HAND, 0), (OPPONENT, C.LOCATION_SZONE, 3))])
    assert place.answers == (bytes([OPPONENT, C.LOCATION_SZONE, 3]),)
    closed = answer(C.MSG_SELECT_PLACE, bytes([OPPONENT, 1]) + struct.pack("<I", 0xFFFFFFFF), [
        move(Y, (OPPONENT, C.LOCATION_HAND, 0), (OPPONENT, C.LOCATION_SZONE, 3))])
    assert closed.answers == ()  # a zone the prompt did not offer is no reading
    position = answer(C.MSG_SELECT_POSITION, bytes([OPPONENT]) + struct.pack("<I", X) + bytes([0x5]), [
        bytes([C.MSG_SPSUMMONING]) + struct.pack("<I", X) + bytes([OPPONENT, C.LOCATION_MZONE, 2, 0x4])])
    assert position.answers == (struct.pack("<i", 4),)


def test_a_lone_optional_activation_is_taken_when_its_chain_link_comes_first_and_declined_otherwise():
    for description in (0, 221):
        question = bytes([OPPONENT]) + struct.pack("<I", X) + bytes([OPPONENT, C.LOCATION_MZONE, 2, 5]) + struct.pack("<I", description)
        taken = answer(C.MSG_SELECT_EFFECTYN, question, [chaining(X, (OPPONENT, C.LOCATION_MZONE, 2), X << 4)])
        assert taken.answers == (struct.pack("<i", 1),)
        declined = answer(C.MSG_SELECT_EFFECTYN, question, [WAIT, own_chain_prompt()])
        assert declined.answers == (struct.pack("<i", 0),)
    # Any other question's answer shows only in its script's operation: no reading.
    other = bytes([OPPONENT]) + struct.pack("<I", X) + bytes([OPPONENT, C.LOCATION_MZONE, 2, 5]) + struct.pack("<I", 94)
    assert answer(C.MSG_SELECT_EFFECTYN, other, [WAIT, own_chain_prompt()]).answers == ()


def test_the_direct_attack_question_is_read_from_the_attack():
    question = bytes([OPPONENT]) + struct.pack("<I", 31)
    direct = bytes([C.MSG_ATTACK, OPPONENT, C.LOCATION_MZONE, 6, 1]) + bytes(4)
    assert answer(C.MSG_SELECT_YESNO, question, [direct]).answers == (struct.pack("<i", 1),)
    targeted = bytes([C.MSG_ATTACK, OPPONENT, C.LOCATION_MZONE, 6, 1, VIEWER, C.LOCATION_MZONE, 2, 1])
    assert answer(C.MSG_SELECT_YESNO, question, [WAIT, WAIT, targeted]).answers == (struct.pack("<i", 0),)
    # A prompt next says nothing (it may be one the local duel never asks): only the attack tells.
    assert answer(C.MSG_SELECT_YESNO, question, [WAIT, WAIT, direct]).answers == (struct.pack("<i", 1),)
    assert answer(C.MSG_SELECT_YESNO, question, [WAIT]).answers == ()
    assert answer(C.MSG_SELECT_YESNO, bytes([OPPONENT]) + struct.pack("<I", 93), [WAIT]).answers == ()


def test_an_attack_replay_goes_on_with_the_same_attacker_or_back_to_the_battle_command():
    question = bytes([OPPONENT]) + struct.pack("<I", 30)
    first = bytes([C.MSG_ATTACK, OPPONENT, C.LOCATION_MZONE, 6, 1]) + bytes([VIEWER, C.LOCATION_MZONE, 2, 1])
    again = bytes([C.MSG_ATTACK, OPPONENT, C.LOCATION_MZONE, 6, 1]) + bytes([VIEWER, C.LOCATION_MZONE, 3, 1])
    other = bytes([C.MSG_ATTACK, OPPONENT, C.LOCATION_MZONE, 1, 1]) + bytes(4)
    def replay(after):
        return read(C.MSG_SELECT_YESNO, question, [first, WAIT, *after], 1, VIEWER).answers
    assert replay([WAIT, again]) == (struct.pack("<i", 1),)
    assert replay([bytes([C.MSG_ATTACK, OPPONENT, C.LOCATION_MZONE, 6, 1]) + bytes(4)]) == (struct.pack("<i", 1),)
    assert replay([WAIT, event_hint(29)]) == (struct.pack("<i", 0),)
    assert replay([WAIT, other]) == (struct.pack("<i", 0),)


def test_cards_set_one_after_another_are_one_operation():
    graveyard = [(X, (OPPONENT, C.LOCATION_GRAVE, index)) for index in range(6)]
    prompt = select_card(3, 3, *graveyard)
    set_ = lambda sequence: bytes([C.MSG_SET]) + bytes(4) + bytes([OPPONENT, C.LOCATION_SZONE, sequence, 0x0a])
    events = [WAIT, move(X, (OPPONENT, C.LOCATION_GRAVE, 4), (OPPONENT, C.LOCATION_SZONE, 1)), set_(1),
              WAIT, move(X, (OPPONENT, C.LOCATION_GRAVE, 1), (OPPONENT, C.LOCATION_SZONE, 2)), set_(2),
              move(X, (OPPONENT, C.LOCATION_GRAVE, 3), (OPPONENT, C.LOCATION_SZONE, 3)), set_(3)]
    # The third card left graveyard slot 3 after slot 4 and slot 1 had gone: it was slot 5 when chosen.
    assert answer(C.MSG_SELECT_CARD, prompt, events).answers == (bytes([3, 4, 1, 5]),)


def test_each_zone_prompt_of_cards_set_together_takes_the_first_move_it_still_offers():
    set_ = lambda sequence: bytes([C.MSG_SET]) + bytes(4) + bytes([OPPONENT, C.LOCATION_SZONE, sequence, 0x0a])
    moves = [move(X, (OPPONENT, C.LOCATION_GRAVE, 4), (OPPONENT, C.LOCATION_SZONE, 1)), set_(1),
             move(X, (OPPONENT, C.LOCATION_GRAVE, 1), (OPPONENT, C.LOCATION_SZONE, 2)), set_(2)]
    second = bytes([OPPONENT, 1]) + struct.pack("<I", 0xFFFFE0FF | 1 << 9)  # zone 1 taken, zones 0 and 2-4 offered
    assert answer(C.MSG_SELECT_PLACE, second, [WAIT, *moves]).answers == (bytes([OPPONENT, C.LOCATION_SZONE, 2]),)


def test_a_hidden_card_chosen_from_a_hand_shuffled_before_it_moves_is_any_placeholder():
    hand = [(X, (OPPONENT, C.LOCATION_HAND, 0)), (BLANK, (OPPONENT, C.LOCATION_HAND, 1)), (BLANK, (OPPONENT, C.LOCATION_HAND, 2))]
    shuffle = bytes([C.MSG_SHUFFLE_HAND, OPPONENT, 3]) + bytes(12)
    hidden = move(0, (OPPONENT, C.LOCATION_HAND, 0), (OPPONENT, C.LOCATION_MZONE, 2))
    reading = answer(C.MSG_SELECT_CARD, select_card(1, 1, *hand), [shuffle, WAIT, hidden])
    # Any placeholder (the first stands for them all), or the known card: only later consequences tell them apart.
    assert reading.answers == (bytes([1, 1]), bytes([1, 0])) and reading.by_consequence


def test_an_attack_belongs_to_the_command_whose_prompts_lead_to_it():
    records = struct.pack("<I", X) + bytes([OPPONENT, C.LOCATION_MZONE, 4, 0]) + struct.pack("<I", Y) + bytes([OPPONENT, C.LOCATION_MZONE, 5, 0])
    menu = bytes([OPPONENT, 0, 2]) + records + bytes([1, 1])
    by_second = bytes([C.MSG_ATTACK, OPPONENT, C.LOCATION_MZONE, 5, 1, VIEWER, C.LOCATION_MZONE, 4, 1])
    # The declared attacker is the command's, however many prompts came first (an attack cancelled at its target
    # and the command asked again leave nothing behind: declaring this one from here is the same duel).
    for waits in (1, 3):
        assert answer(C.MSG_SELECT_BATTLECMD, menu, [WAIT] * waits + [by_second]).answers \
            == (struct.pack("<i", 1 << 16 | 1),)
    # The attack target selection: the target the attack names.
    targets = bytes([OPPONENT, 2, 1, 1, 1]) + struct.pack("<I", Z) + bytes([VIEWER, C.LOCATION_MZONE, 4, 5])
    for waits in (0, 2):
        assert answer(C.MSG_SELECT_CARD, targets, [WAIT] * waits + [by_second]).answers == (bytes([1, 0]),)
    # A direct attack names none of them: the selection was given up, which only its consequences can prove.
    direct = answer(C.MSG_SELECT_CARD, targets, [WAIT, by_second[:5] + bytes(4)])
    assert direct.answers == (struct.pack("<i", -1),) and direct.by_consequence


def test_a_spell_activated_from_the_hand_in_the_battle_phase_asks_its_zone_first():
    attacker = struct.pack("<I", X) + bytes([OPPONENT, C.LOCATION_MZONE, 4, 0])
    spell = struct.pack("<I", Y) + bytes([OPPONENT, C.LOCATION_HAND, 2]) + struct.pack("<I", Y << 4)
    menu = bytes([OPPONENT, 1]) + spell + bytes([1]) + attacker + bytes([1, 1])
    moved = move(Y, (OPPONENT, C.LOCATION_HAND, 2), (OPPONENT, C.LOCATION_SZONE, 2))
    activated = chaining(Y, (OPPONENT, C.LOCATION_SZONE, 2), Y << 4)
    # The zone prompt, the move to it and the link: the activation, though an attacker was offered.
    assert answer(C.MSG_SELECT_BATTLECMD, menu, [WAIT, moved, activated]).answers == (struct.pack("<i", 0),)
    # A prompt with no link behind it is still an attack's target.
    by_attacker = bytes([C.MSG_ATTACK, OPPONENT, C.LOCATION_MZONE, 4, 1, VIEWER, C.LOCATION_MZONE, 4, 1])
    assert answer(C.MSG_SELECT_BATTLECMD, menu, [WAIT, by_attacker]).answers == (struct.pack("<i", 0 << 16 | 1),)


def test_cards_shown_while_an_effect_resolves_are_joined_by_the_hand_cards_its_operation_takes():
    listed = [(X, (OPPONENT, C.LOCATION_HAND, 0)), (Y, (OPPONENT, C.LOCATION_MZONE, 3)), (Z, (OPPONENT, C.LOCATION_SZONE, 0))]
    prompt = bytes([OPPONENT, 0, 1, 3, len(listed)]) + b"".join(struct.pack("<I", code) + bytes([*place, 0x0a])
                                                              for code, place in listed)
    hint = bytes([C.MSG_BECOME_TARGET, 2, OPPONENT, C.LOCATION_MZONE, 3, 1, OPPONENT, C.LOCATION_SZONE, 0, 5])
    shown = bytes([C.MSG_CONFIRM_CARDS, VIEWER, 0, 1]) + struct.pack("<I", X) + bytes([OPPONENT, C.LOCATION_HAND, 0])
    to_deck = move(X, (OPPONENT, C.LOCATION_HAND, 0), (OPPONENT, C.LOCATION_DECK, 0))
    # The field cards are shown by the hint, the hand card by its confirmation: all three were chosen.
    reading = answer(C.MSG_SELECT_CARD, prompt, [hint, shown, to_deck])
    assert reading.answers == (bytes([3, 1, 2, 0]),)
    # A chain link's targets stay the targets only.
    linked = answer(C.MSG_SELECT_CARD, prompt, [hint, bytes([C.MSG_CHAINED, 1])])
    assert linked.answers == (bytes([2, 1, 2]),)


def test_a_chain_link_after_another_card_moved_and_another_prompt_is_not_this_prompts():
    prompt = chain_prompt((Y, (OPPONENT, C.LOCATION_GRAVE, 7), Y << 4, 0))
    destroyed = move(X, (OPPONENT, C.LOCATION_MZONE, 6), (OPPONENT, C.LOCATION_GRAVE, 9))
    later = answer(C.MSG_SELECT_CHAIN, prompt, [destroyed, WAIT, chaining(Y, (OPPONENT, C.LOCATION_GRAVE, 7), Y << 4)])
    assert later.answers == (struct.pack("<i", -1),)
    # A spell chosen a zone and moved there before its chain link is this prompt's activation.
    spell = chain_prompt((Y, (OPPONENT, C.LOCATION_HAND, 0), Y << 4, 0))
    placed = answer(C.MSG_SELECT_CHAIN, spell, [WAIT, move(Y, (OPPONENT, C.LOCATION_HAND, 0), (OPPONENT, C.LOCATION_SZONE, 2)),
                                                chaining(Y, (OPPONENT, C.LOCATION_SZONE, 2), Y << 4)])
    assert placed.answers == (struct.pack("<i", 0), struct.pack("<i", -1)) and placed.open


def test_a_cancelable_selection_is_the_cards_acted_on_and_cancelled_only_when_none_is():
    field = [(X, (OPPONENT, C.LOCATION_MZONE, 3)), (Y, (OPPONENT, C.LOCATION_MZONE, 4))]
    prompt = bytes([OPPONENT, 1, 1, 1, 2]) + b"".join(struct.pack("<I", code) + bytes([*place, 1]) for code, place in field)
    released = move(X, (OPPONENT, C.LOCATION_MZONE, 3), (OPPONENT, C.LOCATION_GRAVE, 0))
    # Given up and asked again for the same card, or taken at once: the same duel.
    for after in ([WAIT, released], [released]):
        reading = answer(C.MSG_SELECT_TRIBUTE, prompt, after)
        assert reading.answers == (bytes([1, 0]),) and not reading.by_consequence
    # Nothing acts on a listed card: given up, to be proved by its consequences.
    elsewhere = move(Z, (OPPONENT, C.LOCATION_HAND, 0), (OPPONENT, C.LOCATION_GRAVE, 0))
    reading = answer(C.MSG_SELECT_TRIBUTE, prompt, [WAIT, elsewhere])
    assert reading.answers == (struct.pack("<i", -1),) and reading.by_consequence


def test_each_target_selection_owns_only_its_own_become_target():
    field = [(X, (OPPONENT, C.LOCATION_MZONE, 5)), (Y, (OPPONENT, C.LOCATION_MZONE, 2))]
    first = bytes([C.MSG_BECOME_TARGET, 1, OPPONENT, C.LOCATION_MZONE, 5, 1])
    second = bytes([C.MSG_BECOME_TARGET, 1, OPPONENT, C.LOCATION_MZONE, 2, 1])
    assert answer(C.MSG_SELECT_CARD, select_card(1, 1, *field), [first, WAIT, second]).answers == (bytes([1, 0]),)
