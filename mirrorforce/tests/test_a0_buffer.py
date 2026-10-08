"""A0's host buffer (mirrorforce/agent/train/a0_buffer.py): segments packed per environment and assembled per learner
device in time-major order, recurrent states cut to their visible tails and restored with zeros before them, and withdrawn
flags that stay live views (a withdrawal reported after packing still masks the decision)."""
import concurrent.futures

import numpy as np

from mirrorforce.agent.train import a0_buffer


def test_pack_and_assemble_round_trip():
    rng = np.random.default_rng(0)
    T, B, slots, cs, d = 4, 3, 5, 2, 6
    steps = [{"x_": rng.integers(0, 255, (B, 7, 2), dtype=np.uint8),
              "logits_": rng.normal(size=(B, 9)).astype(np.float32)} for _ in range(T)]
    small = {"dones": rng.random((T, B)) < 0.2, "mains": rng.random((T, B)) < 0.5,
             "actions": rng.integers(0, 9, (T, B)).astype(np.int32),
             "returns": rng.normal(size=(T, B, 3)).astype(np.float32),
             "advantages": rng.normal(size=(T, B)).astype(np.float32)}
    W = np.zeros((T, B), bool)

    def seat():
        return (rng.normal(size=(B, slots, d)).astype(np.float32), np.array([0, 3, 9], np.int32),
                np.array([-1, 2, 5], np.int32), rng.normal(size=(B, cs, d)).astype(np.float32),
                np.array([0, 1, 0], np.int32), np.array([-1, 2, -1], np.int32))
    rstate = (seat(), seat())
    pool = concurrent.futures.ThreadPoolExecutor(2)
    layout, segs = a0_buffer.pack_update(steps, small, W, rstate, pool)
    W[2, 1] = True  # a later withdrawal reaches the packed segment
    packed, sm, rs = a0_buffer.assemble([segs[2], segs[0], segs[1]], layout, T, 1, slots, cs, d, pool)
    order = [2, 0, 1]
    x = np.stack([s["x_"] for s in steps])  # [T, B, ...]
    exp = np.swapaxes(x[:, order], 0, 1)  # [S, T]
    exp = np.swapaxes(exp, 0, 1).reshape((T * 3,) + x.shape[2:])
    assert np.array_equal(packed["x_"][0], exp)
    assert sm["withdrawn"][0].reshape(T, 3)[2, 2] and sm["withdrawn"][0].sum() == 1
    b, count, last, ch, cc, ct = rs[0]
    assert np.array_equal(b[0, 0], rstate[0][0][2])  # env 2: count 9 >= 5 slots, all visible
    assert not b[0, 1].any()  # env 0: count 0
    assert np.array_equal(b[0, 2][slots - 3:], rstate[0][0][1][slots - 3:]) and not b[0, 2][:slots - 3].any()
    assert list(count[0]) == [9, 0, 3] and list(cc[0]) == [0, 0, 1]
    assert np.array_equal(ch[0, 2][cs - 1:], rstate[0][3][1][cs - 1:]) and not ch[0, 2][:cs - 1].any()
    assert a0_buffer.nbytes(segs) > 0


def test_assemble_places_every_segment_at_its_time_major_rows():
    """Two devices, segments in a shuffled order, with and without the pool: the same arrays as stacking each
    segment's unpacked arrays time-major per device."""
    rng = np.random.default_rng(1)
    T, B, slots, cs, d = 5, 8, 4, 2, 3
    steps = [{"x_": rng.integers(0, 255, (B, 6, 3), dtype=np.uint8), "y_": rng.normal(size=(B, 7)).astype(np.float32),
              "priv:x_": rng.integers(0, 255, (B, 6, 3), dtype=np.uint8)} for _ in range(T)]
    small = {"dones": rng.random((T, B)) < 0.2, "mains": rng.random((T, B)) < 0.5,
             "actions": rng.integers(0, 9, (T, B)).astype(np.int32),
             "returns": rng.normal(size=(T, B, 3)).astype(np.float32),
             "advantages": rng.normal(size=(T, B)).astype(np.float32)}
    seat = lambda: (rng.normal(size=(B, slots, d)).astype(np.float32), np.full(B, 9, np.int32), np.zeros(B, np.int32),
                    rng.normal(size=(B, cs, d)).astype(np.float32), np.ones(B, np.int32), np.zeros(B, np.int32))
    layout, segs = a0_buffer.pack_update(steps, small, np.zeros((T, B), bool), (seat(), seat()))
    order = list(rng.permutation(B))
    chosen = [segs[i] for i in order]
    for pool in (None, concurrent.futures.ThreadPoolExecutor(3)):
        packed, _, _ = a0_buffer.assemble(chosen, layout, T, 2, slots, cs, d, pool)
        for key in layout.keys:
            x = np.stack([s[key] for s in steps])  # [T, B, ...]
            for device in range(2):
                part = x[:, order[device * 4:(device + 1) * 4]]  # [T, S, ...]
                assert np.array_equal(packed[key][device], part.reshape((T * 4,) + x.shape[2:])), key
