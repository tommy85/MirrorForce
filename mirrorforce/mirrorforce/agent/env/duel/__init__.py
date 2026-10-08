"""Duel environment classes over the built extension ``duel_native``.

The extension is a build artifact, not part of the source tree: ``MF_DUEL_NATIVE``
names the module file produced by ``cxx/duelpool/build.py`` and, when set,
``MF_DUEL_NATIVE_SHA256`` pins its digest. The build's ``BUILD.json`` must sit
next to the module, and every output it lists (the module, the core library
``libmfcore.so``, Lua) must match its digest there; ``native_build`` keeps that
record (core commit and tree included) for the checkpoint identity. Loading
refuses a missing path, a missing build record or a digest mismatch instead of
searching other locations.
"""
import hashlib
import importlib.util
import os
from pathlib import Path

from mirrorforce.agent.env.build_record import load_build_record

from mirrorforce.agent.env.python.api import py_env


def load_native():
  global native_build
  path = os.environ.get("MF_DUEL_NATIVE")
  if not path:
    raise ImportError("MF_DUEL_NATIVE must name the built duel_native module file")
  path = Path(path)
  expected = os.environ.get("MF_DUEL_NATIVE_SHA256")
  if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
    raise ImportError("duel_native digest differs from MF_DUEL_NATIVE_SHA256")
  native_build = load_build_record(path)
  spec = importlib.util.spec_from_file_location("duel_native", path)
  if spec is None or spec.loader is None:
    raise ImportError("cannot load duel_native from " + str(path))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


native_build = None
native = load_native()
_DuelEnvPool = native._DuelEnvPool
_DuelEnvSpec = native._DuelEnvSpec
init_module = native.init_module

(
  DuelEnvSpec,
  DuelDMEnvPool,
  DuelGymEnvPool,
  DuelGymnasiumEnvPool,
) = py_env(_DuelEnvSpec, _DuelEnvPool)


__all__ = [
  "DuelEnvSpec",
  "DuelDMEnvPool",
  "DuelGymEnvPool",
  "DuelGymnasiumEnvPool",
  "init_module",
  "native",
  "native_build",
]
