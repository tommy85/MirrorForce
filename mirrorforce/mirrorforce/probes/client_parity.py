"""Client-mode parity of the observation: the env's view of a game against a client builder fed the same game.

A policy playing over the network (the WinBot arena, internal Elo matches, a YGOPro room) only has what its client
receives. ``duel_native.ClientDuel`` (cxx/duelpool) rebuilds the env's observation from that stream. This module
proves the two equal, byte for byte, on recorded games:

1. ``play_game`` plays one game on ``ScriptedDuel`` (both seats, a choice rule per prompt) and keeps, for each seat,
   every decision the env showed it: the menu rows, every ``obs:`` and ``info:`` array and the chosen row.
2. ``seat_streams`` replays the game's core responses through the in-process network host
   (``netduel.local_host.LocalHostDuel`` via ``probes.client_streams``) on a distinct copy of the module's core, giving
   each seat's received stream and which seat sent each response.
3. ``check_seat`` feeds a fresh ``ClientDuel`` one seat's stream in order. At each of its decisions, the menu and every
   array must equal the env's record for that seat's next decision; it is stepped with the env's chosen row. Every
   response it produces (``feed`` answering a prompt itself, or ``step`` completing one) must equal that seat's next
   core response. A missing, extra or reordered decision fails the game.

Not part of the tree (it imports the Python host and core). ``tests/test_client_parity.py`` and
``tools/mf_runtime_client_parity.py`` use it; run each in its own process (two copies of the core).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np

from mirrorforce.netduel import protocol as P

#: Menu row fields the policy reads (``spec`` is a debugging label, not a model input).
ROW_FIELDS = ("act", "phase", "finish", "code", "effect", "position", "number", "place", "attribute", "race")
#: Menu rows that do not advance the turn: passing a chain, cancelling, moving to the next phase.
PASSIVE_ACTS = (0, 9)


@dataclass
class Decision:
    msg: int
    rows: list
    arrays: dict
    index: int


@dataclass
class Game:
    deal: dict
    winner: int
    decisions: tuple                  # per seat, the env's decisions in order
    responses: list                   # every core response, in core order
    owners: list = field(default_factory=list)  # the seat that sent each response (from the host replay)


def row_key(row):
    return tuple(row.get(name) for name in ROW_FIELDS)


def random_rule(seed):
    rng = random.Random(seed)
    return lambda player, msg, rows: rng.randrange(len(rows))


def active_rule(seed, pass_probability=0.15):
    """Prefer rows that do something (activate, summon, set, attack), so turns run long; pass now and then."""
    rng = random.Random(seed)

    def choose(player, msg, rows):
        active = [i for i, row in enumerate(rows) if row.get("act") not in PASSIVE_ACTS and not row.get("phase")]
        if active and rng.random() >= pass_probability:
            return rng.choice(active)
        return rng.randrange(len(rows))
    return choose


def play_game(native, deal, config, choose, *, max_decisions=20000):
    """One ScriptedDuel game under ``choose(player, msg, rows) -> row``; every decision of each seat kept."""
    duel = native.ScriptedDuel(deal, dict(config))
    duel.start()
    decisions = ([], [])
    for _ in range(max_decisions):
        prompt = duel.prompt()
        if prompt is None:
            return Game(deal, duel.winner, decisions, [bytes(r) for r in duel.responses()])
        player, msg, rows = prompt
        arrays = {name: np.array(value, copy=True) for name, value in duel.observation().items()}
        index = int(choose(player, msg, rows))
        decisions[player].append(Decision(int(msg), [dict(row) for row in rows], arrays, index))
        duel.step(index)
    raise RuntimeError(f"the game did not end within {max_decisions} decisions")


def seat_streams(host_core, game, *, max_seconds=600.0):
    """Each seat's received game messages, and which seat sent each recorded response, from the in-process host."""
    from mirrorforce.probes import client_streams as CS

    queue, owners = list(game.responses), []

    class Seat(CS._Seat):
        def __init__(self, seat):
            super().__init__(queue)
            self.seat = seat

        def _handle(self, op, packet):
            if op == P.STOC.GAME_MSG and packet and packet[0] in CS.PROMPTS:
                owners.append(self.seat)
            super()._handle(op, packet)

    seats = [Seat(0), Seat(1)]
    deal = game.deal
    room = P.HostInfo(0, 0, 0, deal["duel_options"] >> 16, 0, 0, deal["start_lp"], deal["start_hand"],
                      deal["draw_count"], 0)
    host = CS.LocalHostDuel(CS.host_config(deal), host_core, seats, room)
    host.play(max_seconds=max_seconds)
    if queue:
        raise RuntimeError(f"{len(queue)} recorded responses were never asked for")
    if host.winner != game.winner:
        raise RuntimeError(f"the host replay ended with winner {host.winner}, the env with {game.winner}")
    game.owners = owners[:len(game.responses)]
    return [[(p[0], bytes(p[1:])) for p in seat.packets] for seat in seats]


def compare_arrays(expected, actual):
    """Per key: None when equal, else what differs (shape/dtype, or the number of differing elements and the first)."""
    out = {}
    for name in sorted(set(expected) | set(actual)):
        if name not in actual or name not in expected:
            out[name] = "missing from the client" if name not in actual else "not in the env"
            continue
        a, b = np.asarray(expected[name]), np.asarray(actual[name])
        if a.shape != b.shape or a.dtype != b.dtype:
            out[name] = f"shape/dtype env {a.shape} {a.dtype}, client {b.shape} {b.dtype}"
        elif a.tobytes() != b.tobytes():
            where = np.argwhere(a != b) if a.ndim else np.zeros((1, 0), int)
            first = tuple(int(i) for i in where[0]) if len(where) else ()
            out[name] = {"differing": int(len(where)), "first": list(first),
                         "env": a[first].item() if a.ndim else a.item(), "client": b[first].item() if b.ndim else b.item()}
    return out


#: Model inputs: the policy reads the ``obs:`` keys only. ``info:`` keys are the env's diagnostics (some count what no
#: client sees, e.g. ``info:step_limit`` counts both players' decisions in the turn); they are compared and reported
#: apart, and do not decide parity.
INPUT_PREFIX = "obs:"
#: Privileged arrays the env may export for training (``export_both_seats``: the other seat's view, for a critic);
#: never client inputs, so never compared.
PRIVILEGED_PREFIX = "priv:"


#: The card-view allowance while the env reads a fresh engine query for card rows (``card_view_law`` fresh_query):
#: a client holds the server's last refresh, which can lag at prompts inside a chain. Allowed differences are only
#: ``obs:cards_`` disabled status (column 11), ATK (12-13) and DEF (14-15), only in rows of cards in a monster zone
#: (column 2 == 3, not an Xyz material: column 6 == 0) that are the same card on both sides (columns 0-10 equal), only
#: at prompts other than the idle and battle command menus. Under ``refreshed_view`` nothing is allowed.
FRESH_QUERY, REFRESHED_VIEW = "fresh_query", "refreshed_view"
CARD_VIEW_COLUMNS = {11: "disabled", 12: "atk_high", 13: "atk_low", 14: "def_high", 15: "def_low"}
COMMAND_MENUS = (10, 11)  # MSG_SELECT_BATTLECMD, MSG_SELECT_IDLECMD
MZONE_ID = 3


def card_view_law(native):
    """The env's card-view law (``native.card_view_law``); a module without it predates the refreshed view."""
    name = getattr(native, "card_view_law", None)
    if name is None:
        return {"law": FRESH_QUERY, "name": FRESH_QUERY,
                "source": "module has no card_view_law (before the refreshed view)"}
    law = name.split("/")[0]  # the module names the law with its version (refreshed_view/v1)
    if law not in (FRESH_QUERY, REFRESHED_VIEW):
        raise ValueError(f"unknown card view law {name!r}")
    return {"law": law, "name": name, "source": "native.card_view_law"}


def allowed_card_view(msg, env, client):
    """The allowed card-view differences of one decision (a list of rows), or None if any difference is outside it."""
    if msg in COMMAND_MENUS or env.shape != client.shape or env.dtype != client.dtype:
        return None
    rows = []
    for row in np.unique(np.argwhere(env != client)[:, 0]):
        e, c = env[row], client[row]
        columns = [int(j) for j in np.nonzero(e != c)[0]]
        if e[2] != MZONE_ID or e[6] != 0 or any(j not in CARD_VIEW_COLUMNS for j in columns) \
                or not np.array_equal(e[:11], c[:11]):
            return None
        rows.append({"row": int(row), "card": int(e[0]) * 256 + int(e[1]), "opponent": int(e[4]),
                     "fields": {CARD_VIEW_COLUMNS[j]: [int(e[j]), int(c[j])] for j in columns},
                     "env_atk_def": [int(e[12]) * 256 + int(e[13]), int(e[14]) * 256 + int(e[15])],
                     "client_atk_def": [int(c[12]) * 256 + int(c[13]), int(c[14]) * 256 + int(c[15])]})
    return rows


def client_duel(native, seat, main, extra, config, **public_recipe):
    """The client-mode builder as deployed: ``netduel.agent_client.AgentClientDuel`` (the client's shadow board supplies
    the card view to ``native.ClientDuel``; every received game message goes to ``feed``)."""
    from mirrorforce.netduel.agent_client import AgentClientDuel
    return AgentClientDuel(native, seat, main, extra, config, **public_recipe)


def check_seat(native, game, seat, stream, config, *, stop_at=1, client_factory=client_duel, law=FRESH_QUERY,
               opponent_recipe=None):
    """Feed ``seat``'s stream to a fresh client builder and compare it with the env's record; the seat's report.

    Parity (``equal``) is the menus, every ``obs:`` array and every response, except the card-view allowance under
    ``law`` fresh_query (every allowed difference is listed under ``card_view``); ``info:`` differences are listed
    apart."""
    if law not in (FRESH_QUERY, REFRESHED_VIEW):
        raise ValueError(f"unknown card view law {law!r}")
    deal = game.deal
    public_args = {}
    if opponent_recipe is not None:
        from mirrorforce.netduel.agent_public_recipe import checked
        declared = checked(opponent_recipe)
        public_args = {"opponent_main": declared["main"], "opponent_extra": declared["extra"]}
    client = client_factory(native, seat, sorted(deal["deck_orders"][seat]), list(deal["extra"][seat]), dict(config),
                            **public_args)
    info, card_view = {}, []
    expected = [r for r, owner in zip(game.responses, game.owners) if owner == seat]
    records, sent, mismatches, k = game.decisions[seat], [], [], 0
    report = {"seat": seat, "decisions": len(records), "responses": len(expected)}

    def fail(kind, **detail):
        mismatches.append({"kind": kind, "decision": k, **detail})
        return len(mismatches) >= stop_at

    for position, (msg, payload) in enumerate(stream):
        out = client.feed(msg, payload)
        if out is not None:
            sent.append(bytes(out))
            if len(sent) > len(expected) or sent[-1] != expected[len(sent) - 1]:
                if fail("forced_response", message=position, msg=msg, client=sent[-1].hex(),
                        env=expected[len(sent) - 1].hex() if len(sent) <= len(expected) else None):
                    break
        while (prompt := client.prompt()) is not None:
            if k >= len(records):
                fail("extra_decision", message=position, msg=int(prompt[1]))
                break
            record = records[k]
            player, cmsg, rows = prompt
            if int(player) != seat or int(cmsg) != record.msg:
                if fail("prompt", message=position, env_msg=record.msg, client_msg=int(cmsg), client_player=int(player)):
                    break
            if [row_key(dict(r)) for r in rows] != [row_key(r) for r in record.rows]:
                if fail("menu", message=position, msg=record.msg, env_rows=len(record.rows), client_rows=len(rows)):
                    break
            observed = client.observation()
            # priv: keys (the both-seat export's critic inputs) exist only in the env; a client never has them
            diff = compare_arrays({k: v for k, v in record.arrays.items() if not k.startswith(PRIVILEGED_PREFIX)},
                                  observed)
            if law == FRESH_QUERY and "obs:cards_" in diff and isinstance(diff["obs:cards_"], dict):
                allowed = allowed_card_view(record.msg, np.asarray(record.arrays["obs:cards_"]),
                                            np.asarray(observed["obs:cards_"]))
                if allowed is not None:
                    card_view.append({"decision": k, "msg": record.msg, "rows": allowed})
                    del diff["obs:cards_"]
            for name in [n for n in diff if not n.startswith(INPUT_PREFIX)]:
                info.setdefault(name, {"decisions": 0, "first": {"decision": k, "msg": record.msg, **(
                    diff[name] if isinstance(diff[name], dict) else {"detail": diff[name]})}})["decisions"] += 1
                del diff[name]
            if diff:
                if fail("observation", message=position, msg=record.msg, keys=diff):
                    break
            out = client.step(record.index)
            k += 1
            if out is not None:
                sent.append(bytes(out))
                if len(sent) > len(expected) or sent[-1] != expected[len(sent) - 1]:
                    if fail("response", message=position, msg=record.msg, client=sent[-1].hex(),
                            env=expected[len(sent) - 1].hex() if len(sent) <= len(expected) else None):
                        break
        if len(mismatches) >= stop_at:
            break
    else:
        if k != len(records):
            fail("missing_decisions", seen=k, env=len(records))
        elif sent != expected:
            fail("responses", client=len(sent), env=len(expected))
    report.update(checked_decisions=k, sent=len(sent), forced=int(client.forced_count()), mismatches=mismatches,
                  info_differences=info, card_view_law=law, card_view=card_view,
                  card_view_decisions=len(card_view), equal=not mismatches)
    return report


def digest_game(game):
    """A game's env record as digests, for comparing two builds of the env (a serving module against the training
    module): per decision the seat, msg, menu rows, chosen row and the SHA-256 of every array; the responses."""
    import hashlib

    def key_digest(array):
        array = np.ascontiguousarray(array)
        return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()

    decisions = []
    for seat in (0, 1):
        for number, decision in enumerate(game.decisions[seat]):
            decisions.append({"seat": seat, "number": number, "msg": decision.msg, "index": decision.index,
                              "rows": [list(row_key(r)) for r in decision.rows],
                              "arrays": {name: key_digest(value) for name, value in sorted(decision.arrays.items())}})
    return {"deal": game.deal, "winner": game.winner, "responses": [r.hex() for r in game.responses],
            "decisions": decisions}


def replay_rule(record):
    """The choice rule that replays a digested game: each seat's chosen rows in order."""
    queues = {0: [d["index"] for d in record["decisions"] if d["seat"] == 0],
              1: [d["index"] for d in record["decisions"] if d["seat"] == 1]}

    def choose(player, msg, rows):
        if not queues[player]:
            raise RuntimeError(f"seat {player} asked for more decisions than the recorded game made")
        return queues[player].pop(0)
    return choose


def compare_digests(expected, actual):
    """The first differences between two digested games of the same deal and choices (empty when equal)."""
    out = []
    if expected["responses"] != actual["responses"]:
        out.append({"kind": "responses", "env": len(expected["responses"]), "other": len(actual["responses"])})
    if len(expected["decisions"]) != len(actual["decisions"]):
        out.append({"kind": "decisions", "env": len(expected["decisions"]), "other": len(actual["decisions"])})
    for a, b in zip(expected["decisions"], actual["decisions"]):
        if (a["seat"], a["msg"], a["rows"]) != (b["seat"], b["msg"], b["rows"]):
            out.append({"kind": "menu", "seat": a["seat"], "number": a["number"]})
        keys = sorted(k for k in set(a["arrays"]) | set(b["arrays"]) if a["arrays"].get(k) != b["arrays"].get(k))
        if keys:
            out.append({"kind": "observation", "seat": a["seat"], "number": a["number"], "keys": keys})
        if len(out) >= 5:
            break
    return out
