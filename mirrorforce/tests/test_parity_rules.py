"""The parity rules of ``probes.client_parity.check_seat`` on a scripted stand-in builder (no native module):
``obs:`` arrays, menus and responses decide parity; ``info:`` differences are reported apart."""
import numpy as np

from mirrorforce.probes import client_parity as V


class Builder:
    """Answers one prompt message (msg 11) with one decision whose arrays are ``arrays``; step returns ``response``."""

    def __init__(self, arrays, response, msg=11):
        self.arrays, self.response, self.msg, self.pending = arrays, response, msg, False

    def feed(self, msg, payload):
        self.pending = msg == self.msg
        return None

    def prompt(self):
        return (0, self.msg, [{"act": 8}, {"act": 0}]) if self.pending else None

    def observation(self):
        return self.arrays

    def step(self, index):
        self.pending = False
        return self.response

    def forced_count(self):
        return 0


def game(arrays):
    record = V.Decision(11, [{"act": 8}, {"act": 0}], arrays, 1)
    return V.Game({"deck_orders": [[1], [1]], "extra": [[], []]}, 0, ([record], []), [b"\x01"], owners=[0])


CARDS = np.zeros((3, 41), np.uint8)
CARDS[0, :3] = [0, 7, 3]           # a monster-zone card (location id 3)
CARDS[0, 12:16] = [7, 208, 3, 232]  # ATK 2000, DEF 1000
CARDS[1, :3] = [0, 9, 4]           # a spell/trap-zone card
CARDS[2, :3] = [0, 5, 3]
CARDS[2, 6] = 1                    # an Xyz material
ENV = {"obs:cards_": CARDS, "info:step_limit": np.array([3, 4, 0], np.int32)}


def check(arrays, response=b"\x01", msg=11, law=V.FRESH_QUERY):
    record = game(ENV)
    record.decisions[0][0].msg = msg
    return V.check_seat(None, record, 0, [(msg, b"")], {}, law=law,
                        client_factory=lambda native, seat, main, extra, config: Builder(arrays, response, msg))


def test_equal_inputs_pass_and_info_differences_are_reported_apart():
    report = check({**ENV, "info:step_limit": np.array([1, 4, 0], np.int32)})
    assert report["equal"] and report["checked_decisions"] == 1
    assert report["info_differences"]["info:step_limit"]["decisions"] == 1


def test_an_input_byte_or_a_response_byte_breaks_parity():
    cards = ENV["obs:cards_"].copy()
    cards[1, 2] = 9
    report = check({**ENV, "obs:cards_": cards})
    assert not report["equal"] and report["mismatches"][0]["keys"]["obs:cards_"]["first"] == [1, 2]
    assert not check(dict(ENV), response=b"\x02")["equal"]


def stale(row=0, column=12, value=11):
    cards = CARDS.copy()
    cards[row, column] = value
    return {**ENV, "obs:cards_": cards}


def test_the_card_view_allowance_is_bounded_and_counted():
    report = check(stale(), msg=13)  # a monster's ATK at a YESNO prompt: allowed and recorded
    assert report["equal"] and report["card_view_decisions"] == 1
    [row] = report["card_view"][0]["rows"]
    assert row["fields"] == {"atk_high": [7, 11]} and row["env_atk_def"] == [2000, 1000]
    assert check(stale(column=11, value=1), msg=18)["equal"]            # disabled status at a PLACE prompt
    assert not check(stale(), msg=11)["equal"]                           # never at the command menus
    assert not check(stale(row=1), msg=13)["equal"]                      # never outside the monster zones
    assert not check(stale(row=2), msg=13)["equal"]                      # never an Xyz material
    assert not check(stale(column=9, value=4), msg=13)["equal"]          # never another column
    assert not check(stale(), msg=13, law=V.REFRESHED_VIEW)["equal"]     # nothing under the refreshed view
