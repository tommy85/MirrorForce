"""Bind hypothetical root choices back to the original published selector.

No sockets, engine admission or policy state mutation. Native branch prompts
are projected before parsing; only exact wire equality or the existing follower's
command/chain/unselect record permutations are accepted. Display descriptions never identify
an action. The surrounding RootController still owns epoch and send checks.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from types import MappingProxyType

from ..netduel.actions import SelectContext, parse_select, MultiSelector, ActionAct
from ..netduel import constants as C
from ..netduel.host_view import deliver
from .client_root import _digest
from .client_shadow import prompt_permutation, unselect_permutation, card_permutation

INFORMATION_SET_SEARCH = True
SCHEMA = 'client-root-menu-binding/v1'


class MenuBindingError(ValueError):
    pass


@dataclass(frozen=True)
class ServerChoice:
    binding_sha256: str
    server_menu_sha256: str
    prefix: tuple[int, ...]
    index: int
    response: bytes | None

    @property
    def key(self):
        return self.server_menu_sha256, self.prefix, self.index


class RootMenuBinding:
    """Frozen original public parsing context; caller keeps its card pool read-only.

    A binding is not a RootEnvelope capability. Never send from this object;
    apply its server index through the real selector, and commit only via the
    current controller. Re-generated origin events must not enter real memory.
    """

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise MenuBindingError('root menu certificate is immutable')
        object.__setattr__(self, name, value)

    def __init__(self, server_prompt: bytes, native_branch_prompt: bytes,
                 context: SelectContext, *, prefix=()):
        if not isinstance(context, SelectContext) or context.our_player not in (0, 1) \
                or not isinstance(server_prompt, bytes) or not server_prompt \
                or not isinstance(native_branch_prompt, bytes) or not native_branch_prompt \
                or any(type(i) is not int or i < 0 for i in prefix):
            raise MenuBindingError('root menu requires immutable prompts and an original public context')
        projection = deliver(native_branch_prompt[0], native_branch_prompt[1:])
        local = projection.payloads.get(context.our_player)
        if local is None or projection.counts.get(context.our_player, 1) != 1:
            raise MenuBindingError('branch prompt is not one published prompt for this observer')
        self.server_prompt, self.local_prompt, self.prefix = server_prompt, bytes(local), tuple(prefix)
        self._native_prompt = native_branch_prompt
        self._context = self._clone_context(context)
        self._context_digest = _digest(self._context, self._context.card_pool)
        self._exact = self.server_prompt == self.local_prompt
        self._response_inverse = None
        self._card_order = None
        if not self._exact:
            if self.server_prompt[0] == C.MSG_SELECT_CARD:
                # The pending native vector keeps its original index order.
                # Match complete published records, including hand coordinates.
                order = card_permutation(self.server_prompt, self.local_prompt)
                if order is None or sorted(order) != list(range(len(order))):
                    raise MenuBindingError('card root is not a complete record-index bijection')
                self._card_order = tuple(order)  # server index -> native vector index
            elif self.prefix:
                raise MenuBindingError('reordered multi-choice prefixes are not supported')
            if self._card_order is not None:
                mapping = None
            elif self.server_prompt[0] == C.MSG_SELECT_UNSELECT_CARD:
                # The follower already certifies both selectable and chosen
                # record lists, including their unchanged envelope. This
                # prompt answers one index (or finish), not a multi-step path.
                order = unselect_permutation(self.server_prompt, self.local_prompt)
                mapping = None if order is None else {
                    bytes([1, server]): bytes([1, local])
                    for server, local in enumerate(order[:self.server_prompt[6]])}
                if mapping is not None and any(self.server_prompt[i] for i in (2, 3)):
                    mapping[b'\xff\xff\xff\xff'] = b'\xff\xff\xff\xff'
            else:
                mapping = prompt_permutation(self.server_prompt, self.local_prompt)
            if self._card_order is None:
                if mapping is None or len(set(mapping.values())) != len(mapping):
                    raise MenuBindingError('branch prompt is not an unambiguous complete-record permutation')
                self._response_inverse = MappingProxyType({local: server for server, local in mapping.items()})
        server = self._parse(self.server_prompt)
        local = self._parse(self.local_prompt)
        self._automatic = server.auto_response
        if (server.auto_response is None) != (local.auto_response is None):
            raise MenuBindingError('automatic/choice status changed across root menus')
        if server.auto_response is not None:
            if self.prefix or self._map_wire(local.auto_response) != server.auto_response:
                raise MenuBindingError('automatic response changed or has a selector prefix')
            self._mapping = ()
        elif self._exact:
            # Parsing two independent copies must reproduce the same sub-step.
            a, b = server.selector.options(), local.selector.options()
            if a != b:
                raise MenuBindingError('identical published prompts produced different original selectors')
            self._mapping = tuple(range(len(a)))
        elif self._card_order is not None:
            self._mapping = self._card_mapping(server.selector, local.selector)
        else:
            source = self._single_responses(self.server_prompt, len(server.selector.options()))
            target = self._single_responses(self.local_prompt, len(local.selector.options()))
            if len(set(source)) != len(source) or len(set(target)) != len(target):
                raise MenuBindingError('non-exact menu has duplicate encoded responses')
            mapped = [self._map_wire(response) for response in target]
            if len(mapped) != len(source) or set(mapped) != set(source):
                raise MenuBindingError('encoded selector actions are not a bijection')
            self._mapping = tuple(source.index(response) for response in mapped)
        self.server_menu_sha256 = _digest((SCHEMA, self.server_prompt, self._context_digest, self.prefix))
        binding = (self.server_menu_sha256, self.local_prompt, self._mapping)
        if self._card_order is not None:
            binding += ('complete-selected-record-index-bijection/v1', self._card_order)
        self.binding_sha256 = _digest(binding)
        self._sealed = True

    @staticmethod
    def _clone_context(context):
        memo = {} if context.card_pool is None else {id(context.card_pool): context.card_pool}
        return copy.deepcopy(context, memo)

    def _check(self):
        if _digest(self._context, self._context.card_pool) != self._context_digest:
            raise MenuBindingError('saved original parsing context was mutated')

    def _parse(self, prompt):
        self._check()
        result = parse_select(prompt[0], prompt[1:], self._clone_context(self._context))
        if result.player != self._context.our_player or not result.complete_menu:
            raise MenuBindingError('wrong observer or truncated original phase menu')
        if self.prefix and result.selector is None:
            raise MenuBindingError('automatic prompt cannot have a selector prefix')
        path = self._prefix_map(self.prefix, to_local=True) \
            if self._card_order is not None and prompt == self.local_prompt else self.prefix
        for local_index in path:
            if result.selector.choose(local_index) is not None:
                raise MenuBindingError('prefix already completed its original response')
        return result

    def _prefix_map(self, path, *, to_local):
        if self._exact:
            return tuple(path)
        if self._card_order is None:
            if path:
                raise MenuBindingError('permuted command prefix is unsupported')
            return ()
        server = parse_select(self.server_prompt[0], self.server_prompt[1:], self._clone_context(self._context)).selector
        local = parse_select(self.local_prompt[0], self.local_prompt[1:], self._clone_context(self._context)).selector
        output = []
        for index in path:
            order = self._card_mapping(server, local)
            if type(index) is not int or index < 0 or (index not in order if to_local else index >= len(order)):
                raise MenuBindingError('card prefix is outside its complete selector bijection')
            native = order.index(index) if to_local else index
            original = index if to_local else order[index]
            a, b = server.choose(original), local.choose(native)
            if a is not None or b is not None:
                raise MenuBindingError('prefix already completed its original response')
            output.append(native if to_local else original)
        return tuple(output)

    @property
    def local_prefix(self):
        self._check()
        return self._prefix_map(self.prefix, to_local=True)

    def with_local_prefix(self, path):
        self._check()
        return RootMenuBinding(self.server_prompt, self._native_prompt, self._context,
                               prefix=self._prefix_map(tuple(path), to_local=False))

    def _card_mapping(self, server, local):
        if not isinstance(server, MultiSelector) or not isinstance(local, MultiSelector):
            raise MenuBindingError('card permutation requires complete multi-select states')
        inverse = {native: original for original, native in enumerate(self._card_order)}
        def keys(selector, translated):
            result = []
            for action in selector.options():
                if action.finish:
                    key = ('finish',)
                elif action.act == ActionAct.CANCEL:
                    key = ('cancel',)
                else:
                    index = selector.ms.spec2idx.get(action.spec)
                    if index is None or translated and index not in inverse:
                        raise MenuBindingError('card selector lost its original record index')
                    key = ('card', inverse[index] if translated else index)
                result.append(key)
            if len(set(result)) != len(result):
                raise MenuBindingError('card selector repeats an original record index')
            return result
        a, b = keys(server, False), keys(local, True)
        if set(a) != set(b):
            raise MenuBindingError('card permutation changed legal continuation choices')
        return tuple(a.index(key) for key in b)

    def _card_wire(self, response, *, to_local):
        response = bytes(response)
        if response == b'\xff\xff\xff\xff':
            if not self._parse(self.server_prompt).selector._can_cancel():
                raise MenuBindingError('card cancellation is not legal at this prefix')
            return response
        count = response[0] if response else -1
        indices = list(response[1:])
        if len(response) != count + 1 or len(set(indices)) != count \
                or not self.server_prompt[3] <= count <= self.server_prompt[4] \
                or any(i >= len(self._card_order) for i in indices):
            raise MenuBindingError('card response is not a complete legal distinct-index selection')
        order = self._card_order if to_local else tuple(self._card_order.index(i) for i in range(len(self._card_order)))
        mapped = [order[i] for i in indices]
        selected = set(self._parse(self.server_prompt).selector.ms.r_idxs)
        if not selected <= set(indices if to_local else mapped):
            raise MenuBindingError('card response abandoned its already selected public prefix')
        return bytes([count, *mapped])

    def _map_wire(self, response):
        response = bytes(response)
        if self._exact:
            return response
        if self._card_order is not None:
            return self._card_wire(response, to_local=False)
        try:
            return self._response_inverse[response]
        except KeyError as exc:
            raise MenuBindingError('response absent from the certified inverse mapping') from exc

    def _single_responses(self, prompt, count):
        encoded = [self._parse(prompt).selector.choose(i) for i in range(count)]
        if any(value is None for value in encoded):
            raise MenuBindingError('non-exact multi-step selectors need a separate path certificate')
        return tuple(bytes(value) for value in encoded)

    def local_response(self, server_response: bytes) -> bytes:
        """Translate an original wire response to this certified branch prompt.

        A client already encodes its complete response, rather than choosing
        a selector row. Exact prompts preserve those bytes (including full
        multi-selections); reordered prompts require the same bijection used
        by choice()/complete_path(), traversed in the opposite direction.
        """
        self._check()
        if type(server_response) is not bytes or not server_response:
            raise MenuBindingError('root response must be nonempty immutable wire bytes')
        if self._exact:
            return server_response
        if self._card_order is not None:
            return self._card_wire(server_response, to_local=True)
        for local, server in self._response_inverse.items():
            if server == server_response:
                return local
        raise MenuBindingError('original response absent from the certified root permutation')

    @property
    def automatic_response(self):
        self._check()
        return self._automatic

    @property
    def choices(self):
        self._check()
        return len(self._mapping)

    def choice(self, branch_index: int) -> ServerChoice:
        self._check()
        if type(branch_index) is not int or not 0 <= branch_index < self.choices:
            raise MenuBindingError('branch row outside this bound selector')
        index = self._mapping[branch_index]
        local = self._parse(self.local_prompt).selector.choose(branch_index)
        server = self._parse(self.server_prompt).selector.choose(index)
        if (local is None) != (server is None) or local is not None and self._map_wire(local) != server:
            raise MenuBindingError('choice changed its certified original response')
        return ServerChoice(self.binding_sha256, self.server_menu_sha256, self.prefix, index,
                            None if server is None else bytes(server))

    def complete_path(self, branch_indices) -> tuple[tuple[int, ...], bytes]:
        """Verify a proposed full response, but never skip real client callbacks."""
        indices = tuple(branch_indices)
        if not indices or any(type(i) is not int or i < 0 for i in indices):
            raise MenuBindingError('complete path requires nonnegative selector choices')
        if not self._exact and self._card_order is None:
            if len(indices) != 1:
                raise MenuBindingError('permuted root accepts one encoded choice only')
            result = self.choice(indices[0])
            return (result.index,), result.response
        server, local = self._parse(self.server_prompt).selector, self._parse(self.local_prompt).selector
        if server is None or local is None:
            raise MenuBindingError('automatic prompt has no searchable path')
        response = None
        mapped_path = []
        for index in indices:
            if response is not None:
                raise MenuBindingError('path contains choices after response completion')
            mapped = self._card_mapping(server, local)[index] if self._card_order is not None else index
            a, b = server.choose(mapped), local.choose(index)
            if (a is None) != (b is None) or b is not None and self._map_wire(b) != a:
                raise MenuBindingError('exact root sub-selection responses differ')
            mapped_path.append(mapped)
            response = a
        if response is None:
            raise MenuBindingError('path did not complete the original response')
        return self.prefix + tuple(mapped_path), bytes(response)

    @property
    def original_prompt_sha256(self):
        return hashlib.sha256(self.server_prompt).hexdigest()
