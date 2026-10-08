"""The wire between a network client and the policy service (``tools/mf_runtime_policy_service.py``).

A policy plays through its client's received stream only: the client forwards every ``STOC_GAME_MSG`` it receives
to the service, and at each prompt the service answers with the exact ``CTOS_RESPONSE`` bytes. The service keeps,
per game, the client-mode observation builder (``duel_native.ClientDuel``) and the policy's recurrent state.

Frames are a 4-byte big-endian length and a UTF-8 JSON object; message payloads and responses travel as hex. Requests:

- ``{"op": "identity"}`` -> the service identity (checkpoint, weights, selection rule, native module, tables).
- ``{"op": "open", "seat": 0|1, "main": [...], "extra": [...], "seed": int}`` -> ``{"session": str}``: one game, the
  seat as the duel numbers it (0 moves first), the deck the client submitted.
  In declared-public-decklist mode, ``public_opponent_recipe`` is required and
  must follow the service identity's explicit ``opponent_recipe_mode`` (``mirror``:
  this client's own recipe; ``known``: a pinned public recipe). The request repeats
  that mode and the content-addressed declaration; never a deck order.
- ``{"op": "decide", "session": s, "messages": [[msg, hex], ...]}`` -> ``{"response": hex, "decisions": [...],
  "forced": bool}``: the messages received since the last call, ending with the prompt to answer.
- ``{"op": "close", "session": s, "messages": [...]}`` -> the session's counters; the messages after the last prompt
  (the end of the duel) are fed first.
- ``open_stream`` takes a NEW client's open fields (including seed and public recipe declaration), plus
  ``frames: [{messages: [[msg, hex], ...], response: hex}, ...]`` and trailing ``messages``. It reconstructs recurrent
  memory from synthetic client packets, forwarding once per historical non-forced sub-choice. Trailing packets
  cannot ask for a response. The reply contains the new session, stream and final-memory SHA256, counters, and each
  forward's observation/memory digests. A malformed stream registers nothing. This is a replay primitive, not the
  particle-to-packet producer: the caller must prove the stream came only from its public prefix plus a particle.
- ``public_world`` accepts exactly ``op``, this connection's ``session`` and ``expected_obs_sha256`` of its scored
  pending decision. It returns a complete public World through the shared integer-map codec, bound to that session
  and observation, plus its SHA. No viewer override, host handle, targets, model call or live-state update is allowed.
- ``open_root_view`` is available only with the explicit current-root search capability. It takes ``session``
  (the live own root), ``expected_obs_sha256``, ``view`` (a masked CurrentRootView of the hypothetical opponent),
  ``seed`` and ``memory_law: own_live_opponent_root_empty/v1``. It creates a fresh opponent observer and empty
  neural Memory without feeding past packets or consuming the own root prompt. The reply binds both root and
  hypothesis hashes. This session and its clones cannot run after the parent's pending decision is committed or
  its session is closed; explicit cleanup remains possible. The live own session is never changed.
- ``current_public_root_seed`` takes exactly ``session`` and ``expected_obs_sha256`` (besides ``op``), and returns
  only current common-public chain/lingering-effect/field-relation facts with a content hash. It does not export
  private hand identities, hints, choices, full history or neural Memory, and does not run another forward.

Socket-created sessions belong to their creating connection, including clones and replay sessions. Another
connection cannot read or change them even with a valid handle and observation SHA. Disconnect drops only the
closing connection's remaining sessions, including handles whose replies were never received; a closed handle
cannot be resumed on a new connection. Explicit close retains its original non-idempotent contract.

A reply with ``"error"`` is a failure of the request; the client raises.
"""
from __future__ import annotations

import json
import math
import socket
import struct
import time

PROTOCOL = "mirrorforce_policy_service/v1"
UNIX_PREFIX = "unix:"
_HEADER = struct.Struct(">I")
_MAX_FRAME = 64 << 20


class WireError(RuntimeError):
    """The service refused a request or the connection broke."""


class WireTransportError(WireError):
    """The connection ended; unlike a remote/protocol refusal this may admit a fresh-session recovery."""


def _remaining(sock, deadline):
    if deadline is not None:
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("an RPC deadline must be a finite monotonic time")
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("RPC absolute deadline expired")
        sock.settimeout(seconds)


def send(sock, payload, *, deadline=None):
    blob = json.dumps(payload, separators=(",", ":")).encode()
    if len(blob) > _MAX_FRAME:
        raise WireError("outgoing frame exceeds the registered maximum")
    _remaining(sock, deadline)
    sock.sendall(_HEADER.pack(len(blob)) + blob)


def _exactly(sock, n, deadline=None):
    chunks, got = [], 0
    while got < n:
        _remaining(sock, deadline)
        chunk = sock.recv(n - got)
        if not chunk:
            raise WireTransportError("connection closed mid-frame")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def recv(sock, *, deadline=None):
    (n,) = _HEADER.unpack(_exactly(sock, _HEADER.size, deadline))
    if n > _MAX_FRAME:
        raise WireError(f"frame of {n} bytes is implausible")
    return json.loads(_exactly(sock, n, deadline))


def connect(address, timeout):
    if not address.startswith(UNIX_PREFIX):
        raise WireError("the policy service listens on a Unix socket (unix:/path)")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(address[len(UNIX_PREFIX):])
    except BaseException:
        sock.close()
        raise
    return sock


def call(sock, payload, *, deadline=None):
    # One absolute budget covers serialization, send, all partial reads and the
    # final decode. A peer sending a slow byte stream cannot renew the timer.
    previous = sock.gettimeout() if deadline is not None else None
    try:
        send(sock, payload, deadline=deadline)
        reply = recv(sock, deadline=deadline)
        _remaining(sock, deadline)
    finally:
        if deadline is not None:
            try:
                sock.settimeout(previous)
            except OSError:
                pass
    if not isinstance(reply, dict):
        raise WireError("malformed reply")
    if "error" in reply:
        raise WireError(str(reply["error"]))
    return reply
