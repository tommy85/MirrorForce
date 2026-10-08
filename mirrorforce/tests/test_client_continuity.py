"""Search continuity: the seat's public path, the worlds kept at its later menus, and the claims a kept world must
still honor where a later root takes it (design 10.13)."""

from collections import Counter
import struct

from mirrorforce.netduel import constants as C
from mirrorforce.search.belief import Evidence
from mirrorforce.common.client_continuity import (Continuation, Kept, LineTrace, Waypoint, kept_world_breach, path_digest,
                                              public_tokens)
from mirrorforce.common.stage_a_joint_belief_runtime import JointDraw, ZoneClaim
from mirrorforce.worldmodel.engine import Message

SZ, MZ, HAND, DECK = C.LOCATION_SZONE, C.LOCATION_MZONE, C.LOCATION_HAND, C.LOCATION_DECK


def _draw(player, *codes):
    return Message(C.MSG_DRAW, bytes([player, len(codes)]) + b"".join(struct.pack("<I", code) for code in codes))


def _chain_prompt(player):
    return Message(C.MSG_SELECT_CHAIN, bytes([player, 0, 0, 0]) + bytes(8))


def test_the_path_is_what_the_seat_receives_and_the_waits_it_sees():
    viewer = 0
    # The opponent's draws differ only in the cards the seat never sees: one path.
    assert public_tokens([_draw(1, 111, 222)], viewer) == public_tokens([_draw(1, 333, 444)], viewer)
    # The seat's own draw shows its cards.
    assert public_tokens([_draw(0, 111)], viewer) != public_tokens([_draw(0, 333)], viewer)
    # The opponent asked to chain is a WAITING the seat sees; the seat's own prompt is its bytes.
    assert public_tokens([_chain_prompt(1)], viewer) == [b"W"]
    own = public_tokens([_chain_prompt(0)], viewer)[0]
    assert own != b"W" and own.endswith(bytes([C.MSG_SELECT_CHAIN, 0, 0, 0, 0]) + bytes(8))
    # A path is its tokens in order, their boundaries included.
    assert path_digest([b"ab", b"c"]) != path_digest([b"a", b"bc"]) != path_digest([b"c", b"ab"])


def _world(hand=(), deck=(), facedown=(), extra=()):
    return JointDraw(hand=tuple(hand), deck=tuple(deck), facedown=tuple(facedown), extra=tuple(extra))


def test_a_kept_world_measures_its_path_from_where_its_root_began_and_continues_past_a_reused_sample():
    world = _world(hand=(1,))
    trace = LineTrace(tokens=[b"a", b"b", b"c", b"d"], waypoints=[Waypoint(2, world, 3), Waypoint(4, world, None)],
                      value=0.5)
    first = Kept(trace, 0, 0, 0.25)
    assert first.digest == path_digest([b"a", b"b"]) and first.sample == (3, 0.5)
    # The next root starts where this waypoint is: the later waypoints are measured from there.
    (later,) = first.later(0.125)
    assert later.digest == path_digest([b"c", b"d"]) and later.weight == 0.125 and later.sample is None
    # A line cut at its decision cap has no rollout to reuse.
    assert Kept(LineTrace([b"a"], [Waypoint(1, world, 2)], None), 0, 0, 1.0).sample is None
    continuation = Continuation(prompt=4, cursor=1, turn=3, kept=(first,))
    messages = [_draw(1, 9), _chain_prompt(1), _chain_prompt(1)]
    assert continuation.matching(messages, 0, 4) == ()  # another turn
    assert continuation.matching(messages, 0, 3) == ()  # another path
    trace.tokens[:2] = public_tokens(messages[1:], 0)
    assert continuation.matching(messages, 0, 3) == (first,)


def test_a_kept_world_must_honor_every_public_claim_where_it_is_taken():
    evidence = Evidence(deck_list=Counter({11: 3, 12: 3, 13: 3, 14: 3}), disclosed_hand=Counter({14: 1}), hand_size=3,
                        deck_size=5, facedown_slots=2, facedown_slot_keys=((SZ, 0), (SZ, 1)),
                        facedown_sampling_keys=((SZ, 0), (SZ, 1)), hand_categories=(frozenset({11, 12}),),
                        facedown_categories=(((SZ, 0), frozenset({13})),), unpositioned_facedown=((SZ, 12),),
                        unpositioned_facedown_categories=((SZ, frozenset({13, 12})), (SZ, frozenset({12}))))
    claims = (ZoneClaim((HAND, DECK), frozenset({11})),)
    good = _world(hand=(14, 11, 13), deck=(11, 12, 12, 13, 14), facedown=((SZ, 0, 13), (SZ, 1, 12), (MZ, 0, 99)))
    assert kept_world_breach(evidence, claims, good) is None
    cases = {
        "facedown_category": _world(hand=good.hand, deck=good.deck, facedown=((SZ, 0, 12), (SZ, 1, 13))),
        "disclosed_hand": _world(hand=(11, 13, 13), deck=good.deck, facedown=good.facedown),
        "hand_category": _world(hand=(14, 13, 13), deck=good.deck, facedown=good.facedown),
        "facedown_slot": _world(hand=good.hand, deck=good.deck, facedown=((SZ, 0, 13),)),
        "unpositioned_identity": _world(hand=good.hand, deck=good.deck, facedown=((SZ, 0, 13), (SZ, 1, 11))),
        "zone_claim": _world(hand=(14, 12, 13), deck=(12, 12, 13, 13, 14), facedown=good.facedown),
    }
    for reason, world in cases.items():
        assert kept_world_breach(evidence, claims, world) == reason, reason
    # Two position-free categories need two distinct witnesses: one 12 serves the second, the 13 the first.
    single = Evidence(**{**vars(evidence), "unpositioned_facedown": (),
                         "unpositioned_facedown_categories": ((SZ, frozenset({12})), (SZ, frozenset({12})))})
    assert kept_world_breach(single, claims, good) == "unpositioned_category"
    assert kept_world_breach(single, claims, _world(hand=good.hand, deck=good.deck,
                                                    facedown=((SZ, 0, 13), (SZ, 1, 12)))) == "unpositioned_category"
    fixed = Evidence(**{**vars(single), "fixed_field_identities": ((SZ, 12),)})
    assert kept_world_breach(fixed, claims, good) is None


def test_a_kept_world_with_a_known_identity_in_a_fresh_slot_breaks_the_claims():
    evidence = Evidence(deck_list=Counter({11: 3, 12: 3, 13: 3}), hand_size=1, deck_size=4, facedown_slots=2,
                        facedown_slot_keys=((SZ, 0), (SZ, 1)), facedown_sampling_keys=((SZ, 0), (SZ, 1)),
                        unpositioned_facedown=((SZ, 12),), fresh_facedown_keys=((SZ, 1),))
    kept = _world(hand=(11,), deck=(11, 13, 13, 12), facedown=((SZ, 0, 12), (SZ, 1, 13)))
    assert kept_world_breach(evidence, (), kept) is None
    # The 12 known among the Spell/Trap cards is not the card set later in slot 1.
    moved = _world(hand=(11,), deck=(11, 13, 13, 13), facedown=((SZ, 0, 12), (SZ, 1, 12)))
    assert kept_world_breach(evidence, (), moved) is None
    only_fresh = _world(hand=(11,), deck=(11, 13, 12, 13), facedown=((SZ, 0, 13), (SZ, 1, 12)))
    assert kept_world_breach(evidence, (), only_fresh) == "unpositioned_identity"
