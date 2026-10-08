"""Deeply immutable records, shared by reference wherever state is copied."""


class ImmutableRecord:
    """A frozen dataclass whose fields hold only deeply immutable values.

    Copies share such a record instead of copying it, and a follower root
    digest reads its content digest, computed once per object: saved history
    then costs nothing per capture however long the duel gets. A change is
    always a new record with its own digest.
    """

    __slots__ = ()

    def __deepcopy__(self, memo):
        return self
