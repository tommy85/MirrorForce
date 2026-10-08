"""Loaders for replay-helper protocol tests; imported by the spawned helper, so no torch here.

Each loader returns an object with ``replay_batch`` that answers like ``test_collector.Native``:
every recorded choice comes back as its one-byte response, winner 0, one turn.
"""
import os
import time

#: Set by a test in its own process; a spawned helper sees only this import-time value.
MARK = "import"


def _result(choices):
    return {"ok": True, "responses": [bytes([c]) for c in choices], "winner": 0, "turns": 1,
            "fallbacks": 0, "truncations": 0, "mark": MARK, "pid": os.getpid()}


class Healthy:
    @staticmethod
    def replay_batch(deals, choices, num_threads=1, **kwargs):
        assert len(deals) == len(choices) == num_threads and kwargs == {"record_actions": False, "menu_digest": False}
        return [_result(row) for row in choices]

    @staticmethod
    def semantics_info():
        return {"artifact_fingerprint": "f" * 64, "card_rows": 3}


class Crashing(Healthy):
    @staticmethod
    def replay_batch(deals, choices, num_threads=1, **kwargs):
        os._exit(3)


class Hanging(Healthy):
    @staticmethod
    def replay_batch(deals, choices, num_threads=1, **kwargs):
        time.sleep(60)


class Failing(Healthy):
    @staticmethod
    def replay_batch(deals, choices, num_threads=1, **kwargs):
        raise RuntimeError("native replay refused this deal")


class Dropping(Healthy):
    @staticmethod
    def replay_batch(deals, choices, num_threads=1, **kwargs):
        return [_result(row) for row in choices][:-1]


class Diverging(Healthy):
    @staticmethod
    def replay_batch(deals, choices, num_threads=1, **kwargs):
        rows = [_result(row) for row in choices]
        rows[-1]["responses"] = rows[-1]["responses"][:-1]
        return rows


class RefusingFirst(Healthy):
    """The core refuses the first game of every request, as the old core's duel_set race did once."""
    @staticmethod
    def replay_batch(deals, choices, num_threads=1, **kwargs):
        rows = [_result(row) for row in choices]
        rows[0] = {**rows[0], "ok": False, "responses": [], "winner": -1, "turns": 0,
                   "error": "query_local_entity_map refused this duel's state"}
        return rows


def runtime(loader="healthy"):
    """A helper runtime whose files are this module and whose loader is one of the fakes below."""
    here = os.path.abspath(__file__)
    return {"core": {"path": here, "sha256": "c" * 64}, "extension": {"path": here, "sha256": "e" * 64},
            "semantics": {"path": here, "sha256": "5" * 64}, "cards_db": "/cards.cdb", "scripts": "/script",
            "loader": "replay_helper_fakes:" + loader}


def joint_runtime(config, local):
    """``mf_runtime_train_joint.replay_runtime`` for joint tests on ``test_collector.Native``."""
    return runtime()


def healthy(runtime):
    return Healthy


def crashing(runtime):
    return Crashing


def hanging(runtime):
    return Hanging


def failing(runtime):
    return Failing


def dropping(runtime):
    return Dropping


def diverging(runtime):
    return Diverging


def refusing_first(runtime):
    return RefusingFirst


def refusing_joint_runtime(config, local):
    """``replay_runtime`` whose helper refuses the first game of every request."""
    return runtime("refusing_first")


def refusing(runtime):
    raise RuntimeError("replay helper core digest differs")
