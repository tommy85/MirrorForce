"""A0's iteration buffer: one iteration's segments on the host, compressed, and the minibatches built from them.

A0 (design section 10e) collects about 4.9M decisions with one behaviour policy, then trains one epoch of minibatches
over them. The decisions do not fit on the devices (about 87 KB of observation each), so each environment's 32-step
segment is kept on the host:

- the observations (every key, time-major [T, ...]), the behaviour logits and the belief labels, as one zlib stream
  (level 1: 104x smaller on K=512 segments, 1.4 GB/s per core);
- the small arrays (dones, mains, actions, returns, advantages) as they are, and ``withdrawn`` as a view into the
  update's matrix, so a withdrawal reported later in the iteration still masks it;
- the recurrent state at the segment start, each seat's memory and chunk summaries cut to their visible tail
  (positions before it are masked by the model, so zeros there give the same outputs).

``assemble`` decompresses a minibatch's segments (in a thread pool, each straight into its rows) and lays them out per
learner device as the learner reads them: observations flattened time-major [T * S, ...] and the recurrent state
[S, ...].
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LEVEL = 1


@dataclasses.dataclass
class Layout:
    """The shapes and dtypes of one step of each packed array (the blob's order)."""
    keys: Tuple[str, ...]
    shapes: Tuple[Tuple[int, ...], ...]
    dtypes: Tuple[np.dtype, ...]

    @staticmethod
    def of(step: Dict[str, np.ndarray]) -> "Layout":
        """From one step's arrays [B, ...]: each environment's row shape."""
        keys = tuple(sorted(step))
        return Layout(keys, tuple(tuple(step[k].shape[1:]) for k in keys),
                      tuple(np.dtype(step[k].dtype) for k in keys))


@dataclasses.dataclass
class Segment:
    blob: bytes
    dones: np.ndarray  # [T] bool
    mains: np.ndarray  # [T] bool
    actions: np.ndarray  # [T] int32
    returns: np.ndarray  # [T, 3] float32
    advantages: np.ndarray  # [T] float32
    withdrawn: np.ndarray  # [T] bool, a view into the update's matrix
    rstate: Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]  # per seat: (tail, count, last, ctail, ccount, cturn)
    rewards: Optional[np.ndarray] = None  # [T] the outcome after the step, acting seat's view (central critic)
    next_dones: Optional[np.ndarray] = None  # [T] a game ended after the step
    value_dists: Optional[np.ndarray] = None  # [T, 3] the public value head's prediction at collection
    bootstrap: Optional[np.ndarray] = None  # [3] the main seat's value after the segment (actor's)
    critic_returns: Optional[np.ndarray] = None  # [T, 3] set by the learner when a central critic trains


def _cut(seat_state, row: int):
    """One environment's seat state with its memory and chunk summaries cut to the visible tail."""
    buffer, count, last, chunks, ccount, cturn = (np.asarray(x[row]) for x in seat_state)
    slots, chunk_slots = buffer.shape[0], chunks.shape[0]
    n, nc = min(int(count), slots), min(int(ccount), chunk_slots)
    return (buffer[slots - n:].copy(), count, last, chunks[chunk_slots - nc:].copy(), ccount, cturn)


def _uncut(seat, slots: int, chunk_slots: int, d: int):
    tail, count, last, ctail, ccount, cturn = seat
    buffer = np.zeros((slots, d), np.float32)
    chunks = np.zeros((chunk_slots, d), np.float32)
    if len(tail):
        buffer[slots - len(tail):] = tail
    if len(ctail):
        chunks[chunk_slots - len(ctail):] = ctail
    return buffer, count, last, chunks, ccount, cturn


def pack_update(steps: Sequence[Dict[str, np.ndarray]], small: Dict[str, np.ndarray], withdrawn: np.ndarray,
                rstate, pool: Optional[concurrent.futures.Executor] = None) -> Tuple[Layout, List[Segment]]:
    """One update's segments. ``steps``: per step, every packed array [B, ...] (observations, behaviour logits, belief
    labels); ``small``: dones, mains, actions [T, B], returns [T, B, 3], advantages [T, B]; ``withdrawn`` [T, B] (kept
    as views); ``rstate``: the pair of seat states [B, ...] at the segment start."""
    layout = Layout.of(steps[0])
    stacked = {k: np.stack([s[k] for s in steps]) for k in layout.keys}  # [T, B, ...]
    batch = small["dones"].shape[1]

    def one(env: int) -> Segment:
        blob = zlib.compress(b"".join(np.ascontiguousarray(stacked[k][:, env]).tobytes() for k in layout.keys), LEVEL)
        extra = {name: small[name][:, env].copy() for name in ("rewards", "next_dones", "value_dists")
                 if name in small}
        if "bootstrap" in small:
            extra["bootstrap"] = small["bootstrap"][env].copy()
        return Segment(blob, small["dones"][:, env].copy(), small["mains"][:, env].copy(),
                       small["actions"][:, env].copy(), small["returns"][:, env].copy(),
                       small["advantages"][:, env].copy(), withdrawn[:, env],
                       tuple(_cut(seat, env) for seat in rstate), **extra)
    segments = list(pool.map(one, range(batch))) if pool is not None else [one(e) for e in range(batch)]
    return layout, segments


def unpack(segment: Segment, layout: Layout, steps: int) -> Dict[str, np.ndarray]:
    raw = zlib.decompress(segment.blob)
    out, offset = {}, 0
    for key, shape, dtype in zip(layout.keys, layout.shapes, layout.dtypes):
        size = int(np.prod((steps,) + shape)) * dtype.itemsize
        out[key] = np.frombuffer(raw, dtype, count=size // dtype.itemsize, offset=offset).reshape((steps,) + shape)
        offset += size
    if offset != len(raw):
        raise ValueError("a segment's blob does not match its layout")
    return out


def _assemble_packed(segments, layout, steps, devices, pool, excluded=()):
    """Same complete blob validation and time-major placement, copying only the requested packed fields."""
    if len(segments) % devices:
        raise ValueError(f"{len(segments)} segments do not split over {devices} devices")
    per = len(segments) // devices
    packed = {k: np.empty((devices, steps * per) + shape, dtype)
              for k, shape, dtype in zip(layout.keys, layout.shapes, layout.dtypes) if k not in excluded}
    views = {k: v.reshape((devices, steps, per) + v.shape[2:]) for k, v in packed.items()}

    def place(index: int):
        device, j = divmod(index, per)
        arrays = unpack(segments[index], layout, steps)
        for k in packed:
            views[k][device, :, j] = arrays[k]
    if pool is not None:
        list(pool.map(place, range(len(segments))))
    else:
        for index in range(len(segments)):
            place(index)
    return packed


def assemble_critic(segments: Sequence[Segment], layout: Layout, steps: int, devices: int,
                    pool: Optional[concurrent.futures.Executor] = None):
    """The critic pre-forward's complete observation only: no unused small arrays or recurrent-state expansion.

    ``make_critic.forward`` already excludes logits_/labels_. Every other field, including all legal-menu and
    priv: fields, is kept unchanged. ``unpack`` still decompresses and validates the entire blob, including those
    two omitted fields. The normal training assembler is unchanged in what it returns.
    """
    return _assemble_packed(segments, layout, steps, devices, pool, ("logits_", "labels_"))


def assemble(segments: Sequence[Segment], layout: Layout, steps: int, devices: int, slots: int, chunk_slots: int,
             d: int, pool: Optional[concurrent.futures.Executor] = None):
    """A minibatch's segments split evenly over ``devices``: per device, (packed arrays [T * S, ...] time-major,
    small arrays [T * S] time-major, recurrent state pair [S, ...]); returned stacked over devices [D, ...].

    Each segment is decompressed straight into its rows of the preallocated arrays (row ``t * S + j`` of its device
    for segment ``j``), in the pool's threads: one copy per byte, parallel over segments."""
    packed = _assemble_packed(segments, layout, steps, devices, pool)
    per = len(segments) // devices

    def time_major(arrays):  # [S, T, ...] -> [T * S, ...]
        a = np.stack(arrays)
        return np.swapaxes(a, 0, 1).reshape((steps * a.shape[0],) + a.shape[2:])
    small = {}
    names = ("dones", "mains", "actions", "returns", "advantages", "withdrawn") + (
        ("critic_returns",) if segments[0].critic_returns is not None else ())
    for name in names:
        small[name] = np.stack([time_major([getattr(s, name) for s in segments[i * per:(i + 1) * per]])
                                for i in range(devices)])
    seats = []
    for seat in range(2):
        fields = [_uncut(s.rstate[seat], slots, chunk_slots, d) for s in segments]
        seats.append(tuple(np.stack([np.stack([np.asarray(f[j]) for f in fields[i * per:(i + 1) * per]])
                                     for i in range(devices)]) for j in range(6)))
    return packed, small, tuple(seats)


def nbytes(segments: Sequence[Segment]) -> int:
    total = 0
    for s in segments:
        total += len(s.blob) + sum(a.nbytes for a in (s.dones, s.mains, s.actions, s.returns, s.advantages))
        total += sum(np.asarray(x).nbytes for seat in s.rstate for x in seat)
    return total
