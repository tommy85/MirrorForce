"""Explicit public deck-list declarations shared by the service and clients.

Only deck contents are declared, never an order or hidden engine state. The
content digest uses the same order-free JSON as internal Elo's recipe identity.
A closed-decklist service refuses a declaration rather than quietly enabling it.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA = "mirrorforce_public_opponent_recipe/v1"
INFORMATION_SET_SEARCH = True


def declare(main, extra) -> dict:
    """Canonical content and identity of an explicitly public opponent recipe."""
    main, extra = list(main), list(extra)
    if not 1 <= len(main) <= 60 or len(extra) > 15 or any(
            type(code) is not int or not 0 < code < 2 ** 32 for code in main + extra):
        raise ValueError("public opponent recipe needs valid card codes, 1-60 main and 0-15 extra cards")
    cards = {"main": sorted(main), "extra": sorted(extra)}
    raw = json.dumps(cards, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return {"schema": SCHEMA, **cards, "sha256": hashlib.sha256(raw).hexdigest()}


def checked(value) -> dict:
    """Validate a declaration without normalizing away a corrupted identity."""
    if not isinstance(value, dict) or set(value) != {"schema", "main", "extra", "sha256"}:
        raise ValueError("an explicit public opponent recipe declaration is required")
    expected = declare(value["main"], value["extra"])
    if value != expected:
        raise ValueError("public opponent recipe content or digest differs")
    return expected


def mode(identity) -> str | None:
    """Validate the service's explicitly selected mirror/known/closed contract."""
    required = identity.get("client_config", {}).get("public_opponent_recipe", False)
    value = identity.get("public_opponent_recipe")
    selected = identity.get("opponent_recipe_mode")
    file_sha = identity.get("opponent_deck_sha256")
    if type(required) is not bool:
        raise ValueError("public_opponent_recipe must be an explicit boolean")
    if not required:
        if any(v is not None for v in (selected, value, file_sha)):
            raise ValueError("closed-decklist service cannot declare the opponent's recipe")
        return None
    if selected == "mirror":
        if value is not None or file_sha is not None:
            raise ValueError("mirror mode takes each client's own recipe, not an opponent deck file")
    elif selected == "known":
        checked(value)
        if not isinstance(file_sha, str) or len(file_sha) != 64 or any(c not in "0123456789abcdef" for c in file_sha):
            raise ValueError("known mode requires the declared opponent deck file SHA-256")
    else:
        raise ValueError("checkpoint requires an explicit opponent recipe mode: mirror or known")
    return selected


def for_game(identity, main, extra) -> dict | None:
    """Declare one game's opponent from the selected rule, never from host state."""
    selected = mode(identity)
    if selected == "mirror":
        return declare(main, extra)
    if selected == "known":
        return checked(identity["public_opponent_recipe"])
    return None
