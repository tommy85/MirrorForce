"""Disclosure of a blank follower's hypothetical engine, not its wire journal.

Reserved placeholder codes identify implementation objects, never cards. Treat
them as unknown at ingress so an old placeholder anchor cannot override a later
real public reveal. Packet matching, engine messages and the received public
board are unchanged; real identities keep the ordinary audience/movement law.
"""
from __future__ import annotations

import struct

from ..netduel.disclosure import DisclosureLedger


class AnonymousDisclosureLedger(DisclosureLedger):
    __slots__ = ("anonymous_codes",)

    def __init__(self, anonymous_codes):
        super().__init__()
        self.anonymous_codes = frozenset(anonymous_codes)
        if not self.anonymous_codes or any(type(code) is not int or not 0 < code < 2**31
                                           for code in self.anonymous_codes):
            raise ValueError("anonymous native identities need explicit reserved positive codes")

    def _identity(self, code):
        code = int(code) & 0x7fffffff
        return 0 if code in self.anonymous_codes else code

    def disclose(self, controller, location, code, *, sequence=None, audience=0b11):
        return super().disclose(controller, location, self._identity(code),
                                sequence=sequence, audience=audience)

    def disclose_faceup_extra(self, controller, code, *, copies=1, audience=0b11):
        return super().disclose_faceup_extra(controller, self._identity(code), copies=copies, audience=audience)

    def _identified_code_for(self, viewer, *, wire_code, **kwargs):
        return super()._identified_code_for(viewer, wire_code=self._identity(wire_code), **kwargs)

    def _anonymous_packet(self, body, offsets):
        masked = None
        for offset in offsets:
            if offset + 4 > len(body):
                continue  # The base ledger retains its existing malformed-input law.
            raw = struct.unpack_from('<I', body, offset)[0]
            if (raw & 0x7fffffff) in self.anonymous_codes:
                if masked is None:
                    masked = bytearray(body)
                struct.pack_into('<I', masked, offset, 0)
        return body if masked is None else bytes(masked)

    def observe_draw(self, body, *, viewer=None, hand_start=None):
        offsets = range(2, 2 + 4 * body[1], 4) if len(body) >= 2 else ()
        return super().observe_draw(self._anonymous_packet(body, offsets), viewer=viewer, hand_start=hand_start)

    def observe_chaining(self, body):
        return super().observe_chaining(self._anonymous_packet(body, (0,)))
