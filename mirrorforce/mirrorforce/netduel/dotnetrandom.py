"""System.Random, reimplemented, so WindBot's throw can be predicted.

Knuth's subtractive generator, the same one .NET Framework and Mono ship.  With
WINDBOT_SEED set, the bot's rock-paper-scissors throw is the first draw off this
stream, so knowing the seed is knowing the throw.
"""
MBIG = 2147483647
MSEED = 161803398


class NetRandom:
    def __init__(self, seed: int):
        subtract = MBIG if seed == -2147483648 else abs(seed)
        mj = MSEED - subtract
        arr = [0] * 56
        arr[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            arr[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += MBIG
            mj = arr[ii]
        for _ in range(1, 5):
            for i in range(1, 56):
                arr[i] -= arr[1 + (i + 30) % 55]
                if arr[i] < 0:
                    arr[i] += MBIG
        self._a = arr
        self._inext = 0
        self._inextp = 21

    def _sample_int(self) -> int:
        i, j = self._inext + 1, self._inextp + 1
        if i >= 56:
            i = 1
        if j >= 56:
            j = 1
        v = self._a[i] - self._a[j]
        if v == MBIG:
            v -= 1
        if v < 0:
            v += MBIG
        self._a[i] = v
        self._inext, self._inextp = i, j
        return v

    def next(self, lo: int, hi: int) -> int:
        return int(self._sample_int() * (1.0 / MBIG) * (hi - lo)) + lo


#: What beats each throw, read off the server's own comparison
#: (``single_duel.cpp`` ``HandResult``): the player holding 2 beats 1, 3 beats
#: 2, and 1 beats 3.  Getting this backwards would lose every throw instead of
#: winning it, which is why it is taken from the rule rather than from the
#: usual naming of the three hands.
BEATS = {1: 2, 2: 3, 3: 1}


def windbot_throw(seed: int) -> int:
    """WindBot's rock-paper-scissors throw for a given ``WINDBOT_SEED``.

    ``Executor.OnRockPaperScissors`` is ``Program.Rand.Next(1, 4)``, and nothing
    else draws from that generator before the duel starts, so the throw is the
    first value off the stream.
    """
    return NetRandom(seed).next(1, 4)


def hand_that_wins(seed: int) -> int:
    """The hand to play against a WindBot seeded with ``seed``.

    Winning the throw is what makes the seat ours to choose: the server sends
    ``STOC_SELECT_TP`` to the winner and nobody else.  Without it the seat comes
    out of a coin we do not hold -- measured at 29% in our favour -- and the
    first-seat column of every table pays for it in width.
    """
    return BEATS[windbot_throw(seed)]
