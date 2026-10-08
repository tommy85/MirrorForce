"""Contract tests for the search-branch forks (the design notes).

BPTS branches a live policy into hypothetical particle rollouts.  The fork
contract has three clauses, each with a catastrophic wrong implementation:

- *shared*: weights, config, tables, and the KV dict.  Deep-copying the KV
  dict would work but cost a full cache copy per branch; sharing is sound
  only because the cache is functional (``BranchPurityTest`` in
  tests_wm.py pins that on the real model).
- *deep-copied*: the serializer stream and rule trackers.  Sharing any of
  them would leak simulated tokens into the real game's stream.
- *severed*: episode records and the lossless capture.  Sharing those
  would train on -- or archive -- lines that were never played.  That is
  the wrong implementation these tests exist to reject.

No torch and no engine: the contract is about which objects are shared,
copied, or severed, so sentinels stand in for the heavy members.
"""

from __future__ import annotations

from collections import Counter

from mirrorforce.netduel.wmplay import WholeGameStream, WorldModelPolicy


class Sentinel:
    """Identity-compared stand-in for a heavy shared member."""

    def __init__(self, name):
        self.name = name

    def __deepcopy__(self, memo):
        raise AssertionError(
            f"{self.name} must be shared, not deep-copied: copying it in "
            "a per-branch fork would be quadratic in practice")


def make_stream():
    stream = WholeGameStream(
        Sentinel("model"), Sentinel("cfg"), Sentinel("ctx"),
        Sentinel("emb_source"), device="cpu", use_cache=True, seat=1,
    )
    stream.builder = {"buf": [1, 2, 3], "dp_pos": [0]}
    stream.persist = {"chain": ["a"]}
    stream.block = {"pending": [5]}
    stream.kv = {"layers": (Sentinel("kv-k"), Sentinel("kv-v")),
                 "length": 12}
    stream.encoded = 7
    return stream


def test_stream_fork_shares_the_cache_and_copies_the_stream():
    stream = make_stream()
    twin = stream.fork()

    # shared by identity: the functional cache and the heavy members
    assert twin.kv is stream.kv
    assert twin.model is stream.model
    assert twin.cfg is stream.cfg
    assert twin.ctx is stream.ctx
    assert twin.emb_source is stream.emb_source
    assert twin.seat == 1 and twin.encoded == 7

    # deep-copied: equal now, independent afterwards
    assert twin.builder == stream.builder and twin.builder is not stream.builder
    assert twin.persist == stream.persist and twin.persist is not stream.persist
    assert twin.block == stream.block and twin.block is not stream.block
    twin.builder["buf"].append(4)
    twin.persist["chain"].append("b")
    assert stream.builder["buf"] == [1, 2, 3]
    assert stream.persist["chain"] == ["a"]

    # the branch replacing its cache must not move the parent's
    twin.kv = {"layers": (), "length": 13}
    assert stream.kv["length"] == 12


def make_policy():
    policy = object.__new__(WorldModelPolicy)
    policy.model = Sentinel("model")
    policy.cfg = Sentinel("cfg")
    policy.ctx = Sentinel("ctx")
    policy.texts = Sentinel("texts")
    policy.emb_source = Sentinel("emb_source")
    policy.card_pool = Sentinel("card_pool")
    policy.device = "cpu"
    policy.fallback = "first"
    policy.temperature = 1.0
    policy.banlists = {1: Sentinel("banlist-table")}
    policy.verify_every = 64
    policy.our_player = 1
    policy.sample = True
    policy.rng = object()
    policy.keep_item = True
    policy.stream = make_stream()
    policy._ordinal = 9
    policy._public_messages = [(1, b"m")]
    policy._since_action = [(2, b"n")]
    policy._pending = {"open": True}
    policy._rule_tracker = {"opt": {3: 1}}
    policy.stats = Counter(decisions=4)
    policy.values = [0.1, 0.2]
    policy.banlist_id = 1
    policy.banlist = ((100, 1),)
    policy.last_action_probs = object()
    policy.decisions = ["real-episode-record"]
    policy.item = object()
    policy.capture = [(7, b"real-capsule-bytes")]
    policy._client = object()
    return policy


def test_policy_fork_severs_experience_and_capture():
    policy = make_policy()
    twin = policy.fork_for_search()

    # shared by identity
    for name in ("model", "cfg", "ctx", "texts", "emb_source", "card_pool",
                 "banlists"):
        assert getattr(twin, name) is getattr(policy, name), name
    assert twin.stream.kv is policy.stream.kv

    # deep-copied serializer-facing state, independent afterwards
    assert twin._rule_tracker == policy._rule_tracker
    assert twin._rule_tracker is not policy._rule_tracker
    twin._public_messages.append((9, b"simulated"))
    twin._since_action.append((9, b"simulated"))
    assert policy._public_messages == [(1, b"m")]
    assert policy._since_action == [(2, b"n")]

    # severed: a hypothetical line is not experience.  The wrong
    # implementation -- sharing these lists -- would train on and archive
    # lines that were never played.
    assert twin.keep_item is False
    assert twin.decisions == [] and twin.decisions is not policy.decisions
    assert twin.capture == [] and twin.capture is not policy.capture
    assert twin.values == [] and twin.values is not policy.values
    assert twin.item is None
    assert twin._client is None
    twin.decisions.append("simulated-decision")
    twin.capture.append((9, b"simulated"))
    assert policy.decisions == ["real-episode-record"]
    assert policy.capture == [(7, b"real-capsule-bytes")]

    # rollouts act greedily and never sample through the parent's rng
    assert twin.sample is False
    assert twin.rng is not policy.rng

    # forking twice yields independent branches
    other = policy.fork_for_search()
    other.capture.append((9, b"other"))
    assert twin.capture == [(9, b"simulated")]


def test_fork_does_not_disturb_the_parent_midgame_fields():
    policy = make_policy()
    before = {
        "ordinal": policy._ordinal,
        "decisions": policy.decisions,
        "capture": policy.capture,
        "item": policy.item,
        "keep_item": policy.keep_item,
        "sample": policy.sample,
        "client": policy._client,
    }
    policy.fork_for_search()
    assert policy._ordinal == before["ordinal"]
    assert policy.decisions is before["decisions"]
    assert policy.capture is before["capture"]
    assert policy.item is before["item"]
    assert policy.keep_item is before["keep_item"]
    assert policy.sample is before["sample"]
    assert policy._client is before["client"]
