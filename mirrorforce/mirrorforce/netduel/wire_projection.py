"""Project one live core message through SingleDuel's host query schedule.

The result is the ``STOC_GAME_MSG`` payload for each player, including host
refreshes and ``MSG_WAITING``. Lobby packets, ``MSG_START`` metadata and timers
are outside this projection. ``project_initial_refreshes`` supplies the two
owner-only Extra Deck refreshes immediately after that metadata.

Call ``project_message`` exactly once per observed core message, before any
further ``process`` or response. Its queries use the host's cache settings;
one query is masked for both players. Querying independently for each viewer,
or projecting again during comparison, changes the core query cache and loses
fields. Other callers must not issue full diagnostic queries on this duel
between host projections: even ``use_cache=0`` updates ``card::q_cache``.

Source: ``ygopro-client/gframe/single_duel.{h,cpp}``, Analyze, WaitforResponse,
WriteUpdateData and Refresh*. No hidden query bytes escape the host masks.
Ported from branch ``sync-validation-20260904`` without behavior changes.
"""

from __future__ import annotations

import ctypes

from . import constants as C
from . import host_view

PROJECTION_SCHEMA = "mirrorforce_host_game_wire/v1"

# SingleDuel's query buffer sizes (game.h and RefreshSingle respectively).
_FIELD_BUFFER_SIZE = 0x4000
_SINGLE_BUFFER_SIZE = 0x1000
_LOCATIONS = {
    "mzone": C.LOCATION_MZONE,
    "szone": C.LOCATION_SZONE,
    "hand": C.LOCATION_HAND,
    "grave": C.LOCATION_GRAVE,
    "extra": C.LOCATION_EXTRA,
}
_WAITING_PROMPTS = frozenset((
    C.MSG_SELECT_BATTLECMD, C.MSG_SELECT_IDLECMD, C.MSG_SELECT_EFFECTYN,
    C.MSG_SELECT_YESNO, C.MSG_SELECT_OPTION, C.MSG_SELECT_CARD,
    C.MSG_SELECT_TRIBUTE, C.MSG_SELECT_UNSELECT_CARD, C.MSG_SELECT_CHAIN,
    C.MSG_SELECT_PLACE, C.MSG_SELECT_DISFIELD, C.MSG_SELECT_POSITION,
    C.MSG_SELECT_COUNTER, C.MSG_SELECT_SUM, C.MSG_SORT_CARD,
    C.MSG_ROCK_PAPER_SCISSORS, C.MSG_ANNOUNCE_RACE, C.MSG_ANNOUNCE_ATTRIB,
    C.MSG_ANNOUNCE_CARD, C.MSG_ANNOUNCE_NUMBER,
))


class InitialRefreshCore:
    """Core proxy capturing startup Extra queries immediately before start.

    Pass this proxy to ``DuelDriver``. After ``build()``, consume
    ``initial_packets`` once, then project ordinary observations against the
    same core. This preserves the host's pre-start query point without changing
    the shared Core object or duplicating DuelDriver's card-loading code.
    """

    def __init__(self, core):
        self._core = core
        self.initial_packets: dict[int, tuple[bytes, ...]] | None = None

    def __getattr__(self, name):
        return getattr(self._core, name)

    def start_duel(self, pduel, options):
        self.initial_packets = project_initial_refreshes(self._core, pduel)
        return self._core.start_duel(pduel, options)


def project_refresh(core, pduel: int, refresh: host_view.Refresh) -> dict[int, bytes]:
    """Execute one host query, then fan its masked bytes out to the players.

    The Core convenience wrapper fixes ``use_cache=0``. Use its already
    declared C entry points so the host's cache=1 defaults remain effective.
    """
    flag = refresh.flag | C.QUERY_CODE | C.QUERY_POSITION
    single = refresh.kind == "single"
    size = _SINGLE_BUFFER_SIZE if single else _FIELD_BUFFER_SIZE
    # The C ABI has no capacity argument. Keep spare room for diagnostics,
    # while rejecting a result the actual host's buffer could not represent.
    buf = ctypes.create_string_buffer(0x10000)
    if single:
        length = core._lib.query_card(pduel, refresh.player, refresh.location, refresh.sequence, flag, buf, 0)
    else:
        try:
            location = _LOCATIONS[refresh.kind]
        except KeyError as exc:
            raise ValueError(f"unknown refresh kind {refresh.kind!r}") from exc
        length = core._lib.query_field_card(pduel, refresh.player, location, flag, buf, refresh.use_cache)
    if length < 0 or length > size - (4 if single else 3):
        raise ValueError(f"host query buffer length out of range: {length}")
    query = buf.raw[:length]
    # Validate all segments before any information is exposed to a viewer.
    list(host_view._iter_segments(query, 0))
    if single:
        return host_view.mask_update_card(refresh.player, refresh.location, refresh.sequence, query)
    return host_view.mask_update_data(refresh.kind, refresh.player, location, query)


def project_initial_refreshes(core, pduel: int) -> dict[int, tuple[bytes, ...]]:
    """The owner-only Extra refreshes after MSG_START, before first process."""
    out: dict[int, list[bytes]] = {0: [], 1: []}
    for player in (0, 1):
        refresh = host_view.Refresh("extra", player, 0xe81fff)
        for viewer, raw in project_refresh(core, pduel, refresh).items():
            out[viewer].append(raw)
    return {viewer: tuple(raws) for viewer, raws in out.items()}


def project_message(core, pduel: int, msg: int, payload: bytes) -> dict[int, tuple[bytes, ...]]:
    """Return this observation's ordered packets, with a single query pass."""
    delivery = host_view.deliver(msg, payload)
    out: dict[int, list[bytes]] = {0: [], 1: []}

    def append_refreshes(refreshes) -> None:
        for refresh in refreshes:
            for viewer, raw in project_refresh(core, pduel, refresh).items():
                out[viewer].append(raw)

    append_refreshes(delivery.before)
    if msg in _WAITING_PROMPTS:
        player = payload[1] if msg == C.MSG_SELECT_SUM else payload[0]
        out[1 - player].append(bytes([C.MSG_WAITING]))
    for viewer, raw in delivery.payloads.items():
        out[viewer].extend([raw] * delivery.counts.get(viewer, 1))
    append_refreshes(delivery.after)
    return {viewer: tuple(raws) for viewer, raws in out.items()}
