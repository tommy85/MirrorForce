"""The boundary between the public observation and the privileged stream (search plan section 2, acceptance 4).

The policy and the search read only public keys (``obs:`` and ``info:``). True hidden state -- the god-view card rows
and hidden layouts of ``SearchDuel.privileged()``, belief labels, critic inputs -- travels under ``priv:`` and
``label:`` keys and may reach losses and offline audits only. ``assert_public`` is the startup check every policy or
search path runs on the keys it will feed: a privileged key is refused, never dropped.
"""
PRIVILEGED_PREFIXES = ("priv:", "label:", "priv.", "label.")


def privileged_keys(keys):
    """The keys of ``keys`` that belong to the privileged stream (any nesting level of a dotted or colon name)."""
    out = []
    for key in keys:
        parts = str(key).replace(".", ":").split(":")
        if any(part in ("priv", "label") for part in parts[:-1]) or str(key).startswith(PRIVILEGED_PREFIXES):
            out.append(key)
    return out


def assert_public(keys, where):
    """Refuse a policy or search input that carries privileged keys."""
    found = privileged_keys(keys)
    if found:
        raise ValueError(f"{where} would read privileged keys {sorted(map(str, found))}; only public observations "
                         "may reach the policy or the search")
    return list(keys)
