"""Read the static card-feature artifacts built from the card scripts.

Three artifacts, three granularities, all keyed by passcode and all built from
the same pinned ``ygopro-scripts`` snapshot.  Effects are numbered identically
in all three, so effect ``i`` is the same effect everywhere.

======================  ==========================  ====================
artifact                built by                    loaded by
======================  ==========================  ====================
``atomic_ops``          ``build_atomic_ops.py``     ``load_atomic_ops``
``op_sequences``        ``build_op_sequences.py``   ``load_op_sequences``
``effect_blocks``       ``build_effect_blocks.py``  ``load_effect_blocks``
``desc_map``            ``build_desc_map.py``       ``load_desc_map``
``card_struct``         ``build_card_struct.py``    ``load_card_struct``
======================  ==========================  ====================

The first is a *bag* of the primitives a card invokes: order-free and
argument-free.  The second keeps what the bag drops -- call order, the callback
each call sits in, and every argument the compiler could read.  The third is
the normalised Lua of each effect, the input to the frozen code-embedding
channel.  The fourth turns the description id the engine reports for a chain
link or a menu entry into the effect index the other three are keyed by, which
is what lets a per-effect vector be bound to the effect the engine actually
named instead of to a card-level aggregate.

This module is the read side of all three: pure standard library, loads once
and caches, and holds no reference to the compilers that produced the files.

Below, the atomic-operation profile.

What a profile carries, per passcode:

``ops`` / ``card_ops``
    Bag of ``Duel.<Name>`` and ``Card.<Name>`` calls the script makes, counted
    over the whole file with comments and string literals masked out.

``category_flags`` / ``category_mask``
    The ``CATEGORY_*`` semantics the script declares -- the duel engine's own
    labelling of what an effect does.  The mask is the bitwise-or of the named
    flags using the values from ``constant.lua``.

``type_flags`` / ``event_flags`` / ``effect_code_flags`` / ``property_flags`` / ``zones``
    Declared effect types, trigger events, continuously-applied effect codes,
    effect flags, and activation zones.

``effects``
    Per-effect breakdown: activation kind, which callbacks exist, count limit,
    where the effect was registered.

Intended use is as a third feature channel beside the card-text embedding and
the typed-reference layer: ``encode_ops`` turns the bag into a fixed-width
vector over a vocabulary that is stable across rebuilds, so a card the model
has never seen still lands in a space it has learned.

The lookup key is the passcode, but nothing in a profile's *values* names a
card -- the builder asserts that with ``ruleir.schema.validate_identity_free``
before writing.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

__all__ = [
    "ARTIFACT_SCHEMA",
    "AtomicOpsIndex",
    "CardFeature",
    "DEFAULT_ARTIFACT",
    "ARROW_ORDER",
    "CARD_STRUCT_SCHEMA",
    "CardStructFeature",
    "CardStructIndex",
    "DEFAULT_CARD_STRUCT",
    "DEFAULT_DESC_MAP",
    "DEFAULT_EFFECT_BLOCKS",
    "DEFAULT_OP_SEQUENCES",
    "DESCRIPTION_LIMIT",
    "DESC_MAP_SCHEMA",
    "DescMapIndex",
    "EFFECT_BLOCKS_SCHEMA",
    "EffectBlock",
    "EffectBlockIndex",
    "EffectFeature",
    "EffectSequence",
    "OP_SEQUENCES_SCHEMA",
    "OpCall",
    "OpSequenceIndex",
    "PAD_ID",
    "ROLES",
    "SequenceFeature",
    "UNKNOWN_ID",
    "VOCAB_OFFSET",
    "load_atomic_ops",
    "load_card_struct",
    "load_desc_map",
    "load_effect_blocks",
    "load_op_sequences",
]


ARTIFACT_SCHEMA = "mirrorforce_atomic_ops/v3"
OP_SEQUENCES_SCHEMA = "mirrorforce_op_sequences/v4"
EFFECT_BLOCKS_SCHEMA = "mirrorforce_effect_blocks/v4"
DESC_MAP_SCHEMA = "mirrorforce_desc_map/v3"
CARD_STRUCT_SCHEMA = "mirrorforce_card_struct/v3"

_CARDFEAT_DIR = Path(__file__).resolve().parents[1] / "data" / "cardfeat"
DEFAULT_ARTIFACT = _CARDFEAT_DIR / "atomic_ops.json.gz"
DEFAULT_OP_SEQUENCES = _CARDFEAT_DIR / "op_sequences.json.gz"
DEFAULT_EFFECT_BLOCKS = _CARDFEAT_DIR / "effect_blocks.json.gz"
DEFAULT_DESC_MAP = _CARDFEAT_DIR / "desc_map.json.gz"
DEFAULT_CARD_STRUCT = _CARDFEAT_DIR / "card_struct.json.gz"

#: A description below this is a system string, not ``code * 16 + n``.  Same
#: constant as ``mirrorforce/netduel/actions.py``; it is repeated rather than
#: imported so this module stays pure standard library, and the artifact
#: records the value it was built with.
DESCRIPTION_LIMIT = 10_000

#: Callback roles, in the order the duel engine consults them.
ROLES = ("condition", "cost", "target", "operation")

#: Link arrow columns, bottom row then middle then top, each left to right.
#: Fixed forever: it is a feature layout, and reordering it would silently
#: repurpose whatever weights were learned for it.
ARROW_ORDER = ("BL", "B", "BR", "L", "R", "TL", "T", "TR")

#: Reserved encoder ids.  ``0`` pads a truncated slot, ``1`` stands for a name
#: the vocabulary does not contain, and real vocabulary entries start at ``2``.
#: Keeping both reserved means a padded slot and an unseen operation are
#: distinguishable, which they must be -- one carries no information, the other
#: carries "something happens here that you have not been taught".
PAD_ID = 0
UNKNOWN_ID = 1
VOCAB_OFFSET = 2


def _schema_error(name: str, found: object, expected: str) -> str:
    """Say why a stale artifact is refused, not just that it is.

    A v1 artifact is not merely older: v2 renumbers effects, so an index stored
    against v1 points at a different effect.  Loading it anyway would be worse
    than failing, and a caller who is told only "unsupported schema" is likely
    to reach for a compatibility shim rather than rebuild.
    """
    message = f"unsupported {name} artifact schema {found!r}, expected {expected!r}"
    if isinstance(found, str) and found.endswith("/v1"):
        message += (
            "; v2 renumbers effects -- rebuild with the mirrorforce/v1/ruleir/build_*.py scripts, "
            "and re-derive anything keyed on effect_index"
        )
    return message


@dataclass(frozen=True)
class EffectFeature:
    """One effect the card's script registers."""

    registration: str
    activation_kind: str
    #: 3 for a Counter Trap, 2 for a Trap / Quick-Play Spell / quick effect,
    #: 1 otherwise.  The direct legality test for a chain response.
    spell_speed: int = 1
    type_flags: tuple[str, ...] = ()
    category_flags: tuple[str, ...] = ()
    property_flags: tuple[str, ...] = ()
    event_flags: tuple[str, ...] = ()
    effect_code_flags: tuple[str, ...] = ()
    zones: tuple[str, ...] = ()
    hint_timing_flags: tuple[str, ...] = ()
    reset_flags: tuple[str, ...] = ()
    callbacks: tuple[str, ...] = ()
    count_limit: int | None = None
    clone_family: int | None = None
    symbolic_fields: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: dict) -> "EffectFeature":
        return cls(
            registration=payload["registration"],
            activation_kind=payload["activation_kind"],
            spell_speed=int(payload.get("spell_speed", 1)),
            **{
                name: tuple(payload.get(name, ()))
                for name in (
                    "type_flags", "category_flags", "property_flags", "event_flags",
                    "effect_code_flags", "zones", "hint_timing_flags", "reset_flags",
                    "callbacks", "symbolic_fields",
                )
            },
            count_limit=payload.get("count_limit"),
            clone_family=payload.get("clone_family"),
        )


@dataclass(frozen=True)
class CardFeature:
    """The atomic-operation profile of one card."""

    passcode: int
    ops: dict[str, int] = field(default_factory=dict)
    card_ops: dict[str, int] = field(default_factory=dict)
    category_mask: int = 0
    category_flags: tuple[str, ...] = ()
    type_flags: tuple[str, ...] = ()
    event_flags: tuple[str, ...] = ()
    property_flags: tuple[str, ...] = ()
    effect_code_flags: tuple[str, ...] = ()
    zones: tuple[str, ...] = ()
    effects: tuple[EffectFeature, ...] = ()
    registration_counts: dict[str, int] = field(default_factory=dict)
    confidence: str = "unknown"

    @classmethod
    def from_payload(cls, passcode: int, payload: dict) -> "CardFeature":
        return cls(
            passcode=passcode,
            ops=dict(payload.get("ops", {})),
            card_ops=dict(payload.get("card_ops", {})),
            category_mask=int(payload.get("category_mask", 0)),
            effects=tuple(
                EffectFeature.from_payload(entry) for entry in payload.get("effects", ())
            ),
            registration_counts=dict(payload.get("registration_counts", {})),
            confidence=payload.get("confidence", "unknown"),
            **{
                name: tuple(payload.get(name, ()))
                for name in (
                    "category_flags", "type_flags", "event_flags",
                    "property_flags", "effect_code_flags", "zones",
                )
            },
        )

    def uses(self, op: str) -> bool:
        """Whether the script calls ``Duel.<op>`` anywhere."""
        return op in self.ops

    def op_count(self, op: str) -> int:
        return self.ops.get(op, 0)

    def has_category(self, flag: str) -> bool:
        """Whether any effect declares ``flag``, e.g. ``\"CATEGORY_DRAW\"``."""
        return flag in self.category_flags

    def has_event(self, flag: str) -> bool:
        return flag in self.event_flags

    @property
    def is_exact(self) -> bool:
        """True when every setter argument read was a constant literal."""
        return self.confidence == "exact"


class AtomicOpsIndex:
    """Passcode-keyed view over one built artifact."""

    def __init__(self, payload: dict) -> None:
        if payload.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError(_schema_error("atomic-ops", payload.get("schema"), ARTIFACT_SCHEMA))
        self._raw: dict[int, dict] = {
            int(code): entry for code, entry in payload["cards"].items()
        }
        self._cache: dict[int, CardFeature] = {}
        self.schema: str = payload["schema"]
        self.script_revision: str = payload.get("script_revision", "unknown")
        self.card_count: int = int(payload.get("card_count", len(self._raw)))
        self.duel_op_totals: dict[str, int] = dict(payload.get("duel_op_totals", {}))
        self.category_flag_totals: dict[str, int] = dict(payload.get("category_flag_totals", {}))

    def __len__(self) -> int:
        return len(self._raw)

    def __contains__(self, passcode: object) -> bool:
        return int(passcode) in self._raw if isinstance(passcode, int) else False

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._raw))

    def get(self, passcode: int) -> CardFeature | None:
        """The profile for ``passcode``, or ``None`` when the card has no script.

        Cards with no script are normal monsters and alternate artworks; a
        caller should treat a miss as "no scripted behaviour", not an error.
        """
        passcode = int(passcode)
        if passcode not in self._raw:
            return None
        feature = self._cache.get(passcode)
        if feature is None:
            feature = CardFeature.from_payload(passcode, self._raw[passcode])
            self._cache[passcode] = feature
        return feature

    def __getitem__(self, passcode: int) -> CardFeature:
        feature = self.get(passcode)
        if feature is None:
            raise KeyError(passcode)
        return feature

    def op_vocabulary(self, min_count: int = 1) -> tuple[str, ...]:
        """Operation names appearing at least ``min_count`` times, sorted by name.

        Sorted by name rather than frequency on purpose: the index a name gets
        must not move when the corpus is repinned and counts shift.  Raising
        ``min_count`` trims the long tail of one-off operations.
        """
        return tuple(sorted(
            name for name, count in self.duel_op_totals.items() if count >= min_count
        ))

    def category_vocabulary(self) -> tuple[str, ...]:
        return tuple(sorted(self.category_flag_totals))

    def encode_ops(
        self,
        passcode: int,
        vocabulary: tuple[str, ...],
        binary: bool = False,
    ) -> list[int]:
        """Fixed-width bag-of-operations vector aligned to ``vocabulary``.

        A passcode with no profile encodes as all zeros, which is the correct
        representation of a card whose script does nothing.
        """
        feature = self.get(passcode)
        if feature is None:
            return [0] * len(vocabulary)
        if binary:
            return [1 if name in feature.ops else 0 for name in vocabulary]
        return [feature.ops.get(name, 0) for name in vocabulary]

    def cards_using(self, op: str) -> tuple[int, ...]:
        """Every passcode whose script calls ``Duel.<op>``."""
        return tuple(
            passcode for passcode, entry in sorted(self._raw.items())
            if op in entry.get("ops", {})
        )

    def cards_with_category(self, flag: str) -> tuple[int, ...]:
        return tuple(
            passcode for passcode, entry in sorted(self._raw.items())
            if flag in entry.get("category_flags", ())
        )


def _load(path: Path) -> AtomicOpsIndex:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return AtomicOpsIndex(json.load(handle))


@lru_cache(maxsize=4)
def _load_cached(path: Path) -> AtomicOpsIndex:
    return _load(path)


def load_atomic_ops(path: Path | str | None = None, cached: bool = True) -> AtomicOpsIndex:
    """Load the artifact.  Repeated calls for the same path share one index."""
    resolved = Path(path) if path is not None else DEFAULT_ARTIFACT
    return _load_cached(resolved) if cached else _load(resolved)


# ---------------------------------------------------------------------------
# Ordered, parameterised operation sequences (v0.5).
#
# The bag above answers "which primitives does this card use".  It cannot
# answer "in what order", "in which callback" or "with what arguments", and in
# this game all three decide what a card does: ``Draw(1)`` is not ``Draw(3)``,
# a ``SendtoGrave`` in a cost is a payment while the same call in an operation
# is the effect, and banish-then-summon is not summon-then-banish.  The
# sequence artifact keeps them; this is its read side.
#
# Everything below is additive.  The bag API above is untouched.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpCall:
    """One call site: what was called, with what, and how deeply inlined.

    ``op`` is a bare name (``"Draw"``) for a duel primitive -- the same
    vocabulary ``AtomicOpsIndex.op_vocabulary`` publishes -- ``"aux:Name"`` for
    a helper-library call and ``"m:Name"`` for a method call on a card, group
    or effect object.

    An argument is one of: ``"V"`` for an expression the compiler read but did
    not evaluate, ``"#<n>"`` for an integer literal, ``"NIL"`` / ``"TRUE"`` /
    ``"FALSE"``, ``"FN"`` for a reference to one of the card's own functions,
    ``"@<op>"`` for a nested call named by its callee, ``"&<op>"`` for a library
    function passed by reference, or a constant name -- possibly a ``+``
    composition, as written, e.g. ``"REASON_COST+REASON_DISCARD"``.

    Three tokens stand in for values that would carry card identity: ``"ID"``
    for the card's own passcode, ``"#CODE"`` for another card's, and ``"#HEX"``
    for any hexadecimal literal, which in these scripts is an archetype set code
    or a bit pattern rather than a quantity.
    """

    op: str
    args: tuple[str, ...] = ()
    #: 0 for a call written in the callback itself, higher when the call came
    #: from a local helper the callback delegates to.
    depth: int = 0
    #: Resolved cross-card references, in the order their ``#REF`` /
    #: ``#SETCODE`` tokens appear in ``args``: ``("card", passcode)`` or
    #: ``("setcode", code)``.  The token stays anonymous; bind the value as a
    #: pointer to that card's vector or to the archetype column, never as a
    #: vocabulary entry.
    refs: tuple[tuple[str, int], ...] = ()

    @classmethod
    def from_payload(cls, payload: list) -> "OpCall":
        return cls(
            op=payload[0],
            args=tuple(payload[1]) if len(payload) > 1 else (),
            depth=int(payload[2]) if len(payload) > 2 else 0,
            refs=tuple((kind, int(value)) for kind, value in payload[3])
            if len(payload) > 3 else (),
        )

    @property
    def is_duel_primitive(self) -> bool:
        return not self.op.startswith(("m:", "aux:"))


@dataclass(frozen=True)
class EffectSequence:
    """The four role sequences of one registered effect."""

    registration: str
    activation_kind: str
    #: See ``EffectFeature.spell_speed``.
    spell_speed: int = 1
    condition: tuple[OpCall, ...] = ()
    cost: tuple[OpCall, ...] = ()
    target: tuple[OpCall, ...] = ()
    operation: tuple[OpCall, ...] = ()
    declared_roles: tuple[str, ...] = ()
    #: Roles the effect declares but whose callback body could not be found --
    #: an ``aux.*`` helper, or a reference the static scan could not resolve.
    unresolved_roles: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: dict) -> "EffectSequence":
        return cls(
            registration=payload["registration"],
            activation_kind=payload["activation_kind"],
            spell_speed=int(payload.get("spell_speed", 1)),
            declared_roles=tuple(payload.get("declared_roles", ())),
            unresolved_roles=tuple(payload.get("unresolved_roles", ())),
            **{
                role: tuple(OpCall.from_payload(call) for call in payload.get(role, ()))
                for role in ROLES
            },
        )

    def role(self, name: str) -> tuple[OpCall, ...]:
        """The sequence for one role.  An undeclared role is an empty tuple."""
        if name not in ROLES:
            raise KeyError(f"unknown role {name!r}; expected one of {ROLES}")
        return getattr(self, name)

    def calls(self) -> tuple[OpCall, ...]:
        """Every call of this effect, roles concatenated in engine order."""
        return tuple(call for role in ROLES for call in self.role(role))


@dataclass(frozen=True)
class SequenceFeature:
    """The role-grouped operation sequences of one card."""

    passcode: int
    effects: tuple[EffectSequence, ...] = ()
    #: Calls in functions no callback/helper closure reaches -- chiefly
    #: ``initial_effect`` and genuinely unused helpers.
    body: tuple[OpCall, ...] = ()
    function_count: int = 0
    attributed_function_count: int = 0
    literal_arg_count: int = 0
    total_arg_count: int = 0
    #: Every other card this script names, and every archetype it queries.
    referenced_passcodes: tuple[int, ...] = ()
    referenced_setcodes: tuple[int, ...] = ()

    @classmethod
    def from_payload(cls, passcode: int, payload: dict) -> "SequenceFeature":
        args = payload.get("args", (0, 0))
        return cls(
            passcode=passcode,
            referenced_passcodes=tuple(payload.get("referenced_passcodes", ())),
            referenced_setcodes=tuple(payload.get("referenced_setcodes", ())),
            effects=tuple(
                EffectSequence.from_payload(entry) for entry in payload.get("effects", ())
            ),
            body=tuple(OpCall.from_payload(call) for call in payload.get("body", ())),
            function_count=int(payload.get("functions", 0)),
            attributed_function_count=int(payload.get("attributed_functions", 0)),
            literal_arg_count=int(args[0]),
            total_arg_count=int(args[1]),
        )

    def effect(self, index: int) -> EffectSequence | None:
        """Effect ``index``, numbered as in the bag artifact, or ``None``."""
        if 0 <= index < len(self.effects):
            return self.effects[index]
        return None

    def role(self, effect_index: int, role: str) -> tuple[OpCall, ...]:
        effect = self.effect(effect_index)
        return () if effect is None else effect.role(role)

    @property
    def literal_arg_rate(self) -> float:
        """Share of arguments the compiler read as a value, not an expression."""
        if not self.total_arg_count:
            return 0.0
        return self.literal_arg_count / self.total_arg_count


def _encode_names(
    names: tuple[str, ...] | list[str],
    lookup: dict[str, int],
    width: int,
) -> list[int]:
    """Ids for ``names``, truncated or padded to exactly ``width`` slots."""
    encoded = [lookup.get(name, UNKNOWN_ID) for name in names[:width]]
    return encoded + [PAD_ID] * (width - len(encoded))


class OpSequenceIndex:
    """Passcode-keyed view over one built sequence artifact."""

    def __init__(self, payload: dict) -> None:
        if payload.get("schema") != OP_SEQUENCES_SCHEMA:
            raise ValueError(
                _schema_error("op-sequence", payload.get("schema"), OP_SEQUENCES_SCHEMA)
            )
        self._raw: dict[int, dict] = {
            int(code): entry for code, entry in payload["cards"].items()
        }
        self._cache: dict[int, SequenceFeature] = {}
        self._vocab_cache: dict[tuple[str, ...], dict[str, int]] = {}
        self.schema: str = payload["schema"]
        self.script_revision: str = payload.get("script_revision", "unknown")
        self.card_count: int = int(payload.get("card_count", len(self._raw)))
        self.effect_count: int = int(payload.get("effect_count", 0))
        self.op_totals: dict[str, int] = dict(payload.get("op_totals", {}))
        self.arg_totals: dict[str, int] = dict(payload.get("arg_totals", {}))
        self.role_call_totals: dict[str, int] = dict(payload.get("role_call_totals", {}))

    def __len__(self) -> int:
        return len(self._raw)

    def __contains__(self, passcode: object) -> bool:
        try:
            return int(passcode) in self._raw  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._raw))

    def get(self, passcode: int) -> SequenceFeature | None:
        """Sequences for ``passcode``, or ``None`` when the card has no script."""
        passcode = int(passcode)
        if passcode not in self._raw:
            return None
        feature = self._cache.get(passcode)
        if feature is None:
            feature = SequenceFeature.from_payload(passcode, self._raw[passcode])
            self._cache[passcode] = feature
        return feature

    def __getitem__(self, passcode: int) -> SequenceFeature:
        feature = self.get(passcode)
        if feature is None:
            raise KeyError(passcode)
        return feature

    def effect(self, passcode: int, effect_index: int) -> EffectSequence | None:
        feature = self.get(passcode)
        return None if feature is None else feature.effect(effect_index)

    def role(self, passcode: int, effect_index: int, role: str) -> tuple[OpCall, ...]:
        """The ordered calls of one role of one effect of one card.

        A bad role name raises whether or not the card exists: a typo must not
        look like an effect that happens to do nothing.
        """
        if role not in ROLES:
            raise KeyError(f"unknown role {role!r}; expected one of {ROLES}")
        feature = self.get(passcode)
        return () if feature is None else feature.role(effect_index, role)

    def _lookup(self, vocabulary: tuple[str, ...]) -> dict[str, int]:
        """Name -> id for one vocabulary, built once and kept.

        ``encode_effect`` calls ``encode_role`` four times and a caller encodes
        thousands of effects; rebuilding a 1,700-entry dict every time dominated
        the cost of encoding.
        """
        cached = self._vocab_cache.get(vocabulary)
        if cached is None:
            cached = {name: index + VOCAB_OFFSET for index, name in enumerate(vocabulary)}
            self._vocab_cache[vocabulary] = cached
        return cached

    def op_vocabulary(
        self,
        min_count: int = 1,
        families: tuple[str, ...] = ("duel", "aux", "method"),
    ) -> tuple[str, ...]:
        """Operation names appearing at least ``min_count`` times, sorted by name.

        Sorted by name and not by frequency for the same reason the bag's
        vocabulary is: repinning the corpus shifts counts, and an index that
        moves silently repurposes every weight that was learned for it.

        ``families`` selects among the duel primitives (``Duel.<Name>``), the
        helper library (``aux:<Name>``) and object methods (``m:<Name>``).
        Methods are over half of all call sites and are mostly predicates; a
        consumer that wants only engine primitives passes ``("duel",)``.
        """
        wanted = set(families)
        selected = []
        for name, count in self.op_totals.items():
            if count < min_count:
                continue
            family = "aux" if name.startswith("aux:") else (
                "method" if name.startswith("m:") else "duel"
            )
            if family in wanted:
                selected.append(name)
        return tuple(sorted(selected))

    def arg_vocabulary(self, min_count: int = 1) -> tuple[str, ...]:
        """Argument tokens appearing at least ``min_count`` times, sorted by name."""
        return tuple(sorted(
            name for name, count in self.arg_totals.items() if count >= min_count
        ))

    def encode_role(
        self,
        passcode: int,
        effect_index: int,
        role: str,
        op_vocabulary: tuple[str, ...],
        arg_vocabulary: tuple[str, ...] = (),
        max_calls: int = 12,
        max_args: int = 4,
    ) -> list[int]:
        """One role's sequence as a fixed-width integer array.

        Layout is ``max_calls`` slots of ``1 + max_args`` integers: the
        operation id, then its argument ids.  Truncation is from the end --
        the first calls of a callback are the ones that decide what it does.
        A card with no script, an effect that does not exist, or a role that is
        not declared all encode as all-``PAD_ID``, which is the correct
        representation of "nothing happens here".
        """
        op_lookup = self._lookup(op_vocabulary)
        arg_lookup = self._lookup(arg_vocabulary)
        width = 1 + max_args
        encoded: list[int] = []
        for call in self.role(passcode, effect_index, role)[:max_calls]:
            encoded.append(op_lookup.get(call.op, UNKNOWN_ID))
            encoded.extend(_encode_names(call.args, arg_lookup, max_args))
        return encoded + [PAD_ID] * (max_calls * width - len(encoded))

    def encode_effect(
        self,
        passcode: int,
        effect_index: int,
        op_vocabulary: tuple[str, ...],
        arg_vocabulary: tuple[str, ...] = (),
        max_calls: int = 12,
        max_args: int = 4,
        roles: tuple[str, ...] = ROLES,
    ) -> list[int]:
        """All four roles of one effect, concatenated in engine order.

        Width is ``len(roles) * max_calls * (1 + max_args)``.  Roles stay in
        separate spans rather than being merged into one sequence: which
        callback a call sits in is the distinction the bag artifact could not
        make, and flattening would throw it away again.
        """
        return [
            value
            for role in roles
            for value in self.encode_role(
                passcode, effect_index, role, op_vocabulary,
                arg_vocabulary, max_calls, max_args,
            )
        ]

    def cards_calling(self, op: str, role: str | None = None) -> tuple[int, ...]:
        """Passcodes whose script calls ``op``, optionally only in ``role``."""
        if role is not None and role not in ROLES:
            raise KeyError(f"unknown role {role!r}; expected one of {ROLES}")
        roles = ROLES if role is None else (role,)
        found = []
        for passcode, entry in sorted(self._raw.items()):
            sequences: list = []
            if role is None:
                sequences.append(entry.get("body", ()))
            for effect in entry.get("effects", ()):
                sequences.extend(effect.get(name, ()) for name in roles)
            if any(call[0] == op for sequence in sequences for call in sequence):
                found.append(passcode)
        return tuple(found)


def _load_sequences(path: Path) -> OpSequenceIndex:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return OpSequenceIndex(json.load(handle))


@lru_cache(maxsize=4)
def _load_sequences_cached(path: Path) -> OpSequenceIndex:
    return _load_sequences(path)


def load_op_sequences(path: Path | str | None = None, cached: bool = True) -> OpSequenceIndex:
    """Load the sequence artifact.  Repeated calls for a path share one index."""
    resolved = Path(path) if path is not None else DEFAULT_OP_SEQUENCES
    return _load_sequences_cached(resolved) if cached else _load_sequences(resolved)


# ---------------------------------------------------------------------------
# Normalised per-effect Lua blocks (v0.5).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectBlock:
    """One effect's normalised Lua source and how it is labelled."""

    passcode: int
    index: int
    #: ``activation`` for an effect the player activates, ``applied`` for one
    #: that applies once registered, ``procedure`` for a summon procedure.
    kind: str
    #: ``card`` for ``c:RegisterEffect``, ``duel`` for ``Duel.RegisterEffect``,
    #: ``unregistered`` for an effect the scan never saw installed.
    registration: str
    activation_kind: str
    code: str
    callbacks: tuple[str, ...] = ()
    unresolved_callbacks: tuple[str, ...] = ()
    #: 1-based effect number in the card text, or ``None`` when unaligned.
    text_index: int | None = None
    #: Which alignment rule produced ``text_index``: ``exact``, ``activation``,
    #: ``scoped``, or ``unaligned``.
    alignment: str = "unaligned"

    @classmethod
    def from_payload(cls, passcode: int, index: int, alignment: str, payload: dict) -> "EffectBlock":
        return cls(
            passcode=passcode,
            index=index,
            kind=payload["kind"],
            registration=payload["registration"],
            activation_kind=payload["activation_kind"],
            code=payload["code"],
            callbacks=tuple(payload.get("callbacks", ())),
            unresolved_callbacks=tuple(payload.get("unresolved_callbacks", ())),
            text_index=payload.get("text_index"),
            alignment=alignment,
        )


class EffectBlockIndex:
    """Passcode-keyed view over the normalised per-effect Lua blocks."""

    def __init__(self, payload: dict) -> None:
        if payload.get("schema") != EFFECT_BLOCKS_SCHEMA:
            raise ValueError(
                _schema_error("effect-block", payload.get("schema"), EFFECT_BLOCKS_SCHEMA)
            )
        self._raw: dict[int, dict] = {
            int(code): entry for code, entry in payload["cards"].items()
        }
        self._cache: dict[int, tuple[EffectBlock, ...]] = {}
        self.schema: str = payload["schema"]
        self.script_revision: str = payload.get("script_revision", "unknown")
        self.card_count: int = int(payload.get("card_count", len(self._raw)))
        self.block_count: int = int(payload.get("block_count", 0))
        self.alignment_tier_totals: dict[str, int] = dict(
            payload.get("alignment_tier_totals", {})
        )

    def __len__(self) -> int:
        return len(self._raw)

    def __contains__(self, passcode: object) -> bool:
        try:
            return int(passcode) in self._raw  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._raw))

    def alignment(self, passcode: int) -> str:
        """Which alignment rule fired for this card, or ``unaligned``."""
        entry = self._raw.get(int(passcode))
        return "unaligned" if entry is None else entry.get("alignment", "unaligned")

    def text_effect_count(self, passcode: int) -> int:
        """How many numbered effects the card text declares."""
        entry = self._raw.get(int(passcode))
        return 0 if entry is None else int(entry.get("text_effects", 0))

    def blocks(self, passcode: int) -> tuple[EffectBlock, ...]:
        """Every block of a card, in the same effect order as the other artifacts."""
        passcode = int(passcode)
        cached = self._cache.get(passcode)
        if cached is not None:
            return cached
        entry = self._raw.get(passcode)
        if entry is None:
            return ()
        alignment = entry.get("alignment", "unaligned")
        built = tuple(
            EffectBlock.from_payload(passcode, index, alignment, payload)
            for index, payload in enumerate(entry.get("blocks", ()))
        )
        self._cache[passcode] = built
        return built

    def block(self, passcode: int, effect_index: int) -> EffectBlock | None:
        """One block, or ``None`` when the card or the effect does not exist."""
        blocks = self.blocks(passcode)
        if 0 <= effect_index < len(blocks):
            return blocks[effect_index]
        return None

    def block_for_text_effect(self, passcode: int, text_index: int) -> EffectBlock | None:
        """The block aligned to text effect ``text_index`` (1-based), if any."""
        for block in self.blocks(passcode):
            if block.text_index == text_index:
                return block
        return None

    def iter_blocks(self) -> Iterator[EffectBlock]:
        """Every block of every card, passcode order then effect order."""
        for passcode in self:
            yield from self.blocks(passcode)


def _load_blocks(path: Path) -> EffectBlockIndex:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return EffectBlockIndex(json.load(handle))


@lru_cache(maxsize=4)
def _load_blocks_cached(path: Path) -> EffectBlockIndex:
    return _load_blocks(path)


def load_effect_blocks(path: Path | str | None = None, cached: bool = True) -> EffectBlockIndex:
    """Load the effect-block artifact.  Repeated calls for a path share one index."""
    resolved = Path(path) if path is not None else DEFAULT_EFFECT_BLOCKS
    return _load_blocks_cached(resolved) if cached else _load_blocks(resolved)


# ---------------------------------------------------------------------------
# Engine description id -> effect index.
#
# The engine names an effect by a description id, not by position:
# ``MSG_CHAINING`` carries ``(ch_code, ch_desc)`` and a menu entry carries the
# same thing.  Everything that wants a *per-effect* card vector has to turn that
# into an effect index, and the text-derived effect axis cannot do it exactly --
# it is an over-approximation of the script's effect table.  The script itself
# has the answer: ``e:SetDescription(aux.Stringid(code, n))`` binds description
# ``code * 16 + n`` to the effect being configured, and the registration walker
# reads that setter anyway.
#
# Measured against 310,652 decision points of the PV-14 corpus: of the 144,331
# engine-offered activate-effect entries, 98.88% resolve to exactly one effect
# index.  Numbers and the residual's anatomy are in mirrorforce/v1/pv/cardfeat-v05.md.
# ---------------------------------------------------------------------------


class DescMapIndex:
    """Passcode-keyed view over the description-to-effect-index map."""

    def __init__(self, payload: dict) -> None:
        if payload.get("schema") != DESC_MAP_SCHEMA:
            raise ValueError(
                _schema_error("desc-map", payload.get("schema"), DESC_MAP_SCHEMA)
            )
        self._raw: dict[int, dict] = {
            int(code): entry for code, entry in payload["cards"].items()
        }
        self.schema: str = payload["schema"]
        self.script_revision: str = payload.get("script_revision", "unknown")
        self.description_limit: int = int(
            payload.get("description_limit", DESCRIPTION_LIMIT)
        )
        self.stringid_radix: int = int(payload.get("stringid_radix", 16))
        self.card_count: int = int(payload.get("card_count", 0))
        self.effect_count: int = int(payload.get("effect_count", 0))
        self.described_effect_count: int = int(payload.get("described_effect_count", 0))
        self.unique_desc_count: int = int(payload.get("unique_desc_count", 0))
        self.ambiguous_desc_count: int = int(payload.get("ambiguous_desc_count", 0))

    def __len__(self) -> int:
        return len(self._raw)

    def __contains__(self, passcode: object) -> bool:
        try:
            return int(passcode) in self._raw  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._raw))

    def effect_index_for_desc(self, passcode: int, desc: int) -> int | None:
        """The effect index this description names, or ``None``.

        ``None`` covers three different situations on purpose, because a caller
        that cannot bind an effect should fall back the same way in all three:
        the card has no script, the description is one no effect of this card
        sets, or several of its effects set it and the description alone does
        not choose between them.  ``candidates_for_desc`` separates them when
        that matters.

        ``desc == 0`` is always ``None``.  The engine sends 0 for an effect that
        set no description, and the action layer already reads that as "activate
        this card" rather than "this numbered effect"; returning an index would
        invent a binding the script never made.
        """
        entry = self._raw.get(int(passcode))
        if entry is None or not desc:
            return None
        return entry.get("desc", {}).get(str(int(desc)))

    def candidates_for_desc(self, passcode: int, desc: int) -> tuple[int, ...]:
        """Every effect index that sets this description.

        One element is an exact binding, several is a genuine ambiguity -- 93.3%
        of those are an effect and its ``Clone()``, which share a description
        because the script gave them one, so the *engine* cannot tell them apart
        by description either.  Empty means unmapped.
        """
        entry = self._raw.get(int(passcode))
        if entry is None or not desc:
            return ()
        key = str(int(desc))
        unique = entry.get("desc", {}).get(key)
        if unique is not None:
            return (unique,)
        return tuple(entry.get("ambiguous", {}).get(key, ()))

    def is_system_description(self, desc: int) -> bool:
        """Whether ``desc`` is a system string rather than ``code * 16 + n``."""
        return 0 < int(desc) < self.description_limit

    def described_effects(self, passcode: int) -> int:
        entry = self._raw.get(int(passcode))
        return 0 if entry is None else int(entry.get("described_effects", 0))

    def descriptions(self, passcode: int) -> dict[int, int]:
        """Every unambiguous ``desc -> effect_index`` binding of one card."""
        entry = self._raw.get(int(passcode))
        if entry is None:
            return {}
        return {int(desc): index for desc, index in entry.get("desc", {}).items()}


def _load_desc_map(path: Path) -> DescMapIndex:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return DescMapIndex(json.load(handle))


@lru_cache(maxsize=4)
def _load_desc_map_cached(path: Path) -> DescMapIndex:
    return _load_desc_map(path)


def load_desc_map(path: Path | str | None = None, cached: bool = True) -> DescMapIndex:
    """Load the description map.  Repeated calls for a path share one index."""
    resolved = Path(path) if path is not None else DEFAULT_DESC_MAP
    return _load_desc_map_cached(resolved) if cached else _load_desc_map(resolved)


# ---------------------------------------------------------------------------
# Printed card structure, with the type-dependent fields decoded.
#
# For a Link monster the ``def`` column is an 8-direction arrow bitmap, not a
# defence value, and ``level`` is a link rating, not a Level.  The PV-1 format
# string printed both as plain numbers, so the arrows -- which decide which
# zones an Extra Deck monster can be summoned to point at, and therefore which
# plays are legal -- never reached the text channel at all (audit item G-1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardStructFeature:
    """One card's printed structure."""

    passcode: int
    card_type: int
    spell_speed: int = 1
    attack: int | None = None
    #: Defence, or ``None`` when the card is a Link monster and has none.
    defense: int | None = None
    level: int | None = None
    link_rating: int | None = None
    #: Arrow direction names in ``ARROW_ORDER``; empty for a non-Link card.
    arrows: tuple[str, ...] = ()
    setcodes: tuple[int, ...] = ()
    alias: int = 0

    @classmethod
    def from_payload(cls, passcode: int, payload: dict) -> "CardStructFeature":
        return cls(
            passcode=passcode,
            card_type=int(payload["type"]),
            spell_speed=int(payload.get("spell_speed", 1)),
            attack=payload.get("atk"),
            defense=payload.get("def"),
            level=payload.get("level"),
            link_rating=payload.get("link_rating"),
            arrows=tuple(payload.get("arrows", ())),
            setcodes=tuple(payload.get("setcodes", ())),
            alias=int(payload.get("alias", 0)),
        )

    @property
    def is_link(self) -> bool:
        return self.link_rating is not None

    def arrow_bits(self) -> list[int]:
        """The eight arrow columns, in ``ARROW_ORDER``.  All zero for non-Link."""
        return [1 if name in self.arrows else 0 for name in ARROW_ORDER]

    def points_to(self, direction: str) -> bool:
        if direction not in ARROW_ORDER:
            raise KeyError(f"unknown arrow {direction!r}; expected one of {ARROW_ORDER}")
        return direction in self.arrows


class CardStructIndex:
    """Passcode-keyed view over the printed-structure artifact."""

    def __init__(self, payload: dict) -> None:
        if payload.get("schema") != CARD_STRUCT_SCHEMA:
            raise ValueError(
                _schema_error("card-struct", payload.get("schema"), CARD_STRUCT_SCHEMA)
            )
        self._raw: dict[int, dict] = {
            int(code): entry for code, entry in payload["cards"].items()
        }
        self._cache: dict[int, CardStructFeature] = {}
        self.schema: str = payload["schema"]
        self.cards_db_sha256: str = payload.get("cards_db_sha256", "unknown")
        self.arrow_order: tuple[str, ...] = tuple(payload.get("arrow_order", ARROW_ORDER))
        self.card_count: int = int(payload.get("card_count", len(self._raw)))
        self.link_monster_count: int = int(payload.get("link_monster_count", 0))

    def __len__(self) -> int:
        return len(self._raw)

    def __contains__(self, passcode: object) -> bool:
        try:
            return int(passcode) in self._raw  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._raw))

    def get(self, passcode: int) -> CardStructFeature | None:
        passcode = int(passcode)
        if passcode not in self._raw:
            return None
        feature = self._cache.get(passcode)
        if feature is None:
            feature = CardStructFeature.from_payload(passcode, self._raw[passcode])
            self._cache[passcode] = feature
        return feature

    def __getitem__(self, passcode: int) -> CardStructFeature:
        feature = self.get(passcode)
        if feature is None:
            raise KeyError(passcode)
        return feature

    def arrow_bits(self, passcode: int) -> list[int]:
        """Eight arrow columns for any card; all zero when it has no arrows."""
        feature = self.get(passcode)
        return [0] * len(ARROW_ORDER) if feature is None else feature.arrow_bits()

    def text_fragment(self, passcode: int) -> str:
        """The phrase to append to this card's embedding input string.

        Empty for anything that is not a Link monster.  Spelled out in words
        rather than glyphs because a language model already knows what
        "bottom-left" means.
        """
        entry = self._raw.get(int(passcode))
        return "" if entry is None else entry.get("text_fragment", "")

    def link_monsters(self) -> tuple[int, ...]:
        return tuple(
            passcode for passcode, entry in sorted(self._raw.items()) if "arrows" in entry
        )


def _load_card_struct(path: Path) -> CardStructIndex:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return CardStructIndex(json.load(handle))


@lru_cache(maxsize=4)
def _load_card_struct_cached(path: Path) -> CardStructIndex:
    return _load_card_struct(path)


def load_card_struct(path: Path | str | None = None, cached: bool = True) -> CardStructIndex:
    """Load the printed-structure artifact.  Repeated calls share one index."""
    resolved = Path(path) if path is not None else DEFAULT_CARD_STRUCT
    return _load_card_struct_cached(resolved) if cached else _load_card_struct(resolved)
