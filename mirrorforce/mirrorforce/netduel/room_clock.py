"""Explicit search room clocks; the legacy600-second contract remains the default."""

def registered_room_seconds(seconds):
    from .untimed_search import ROOM_SECONDS as UNTIMED_ROOM_SECONDS
    if type(seconds) is not int or seconds not in (450, 600, UNTIMED_ROOM_SECONDS):
        raise ValueError('search requires an explicitly supported450/600-second room clock or the untimed evaluation room')
    return seconds


def verified_room_seconds(client):
    declared = registered_room_seconds(getattr(getattr(client, 'limits', None), 'room_seconds', 600))
    if not getattr(client, 'room_verified', False) or client.host_info.time_limit != declared:
        raise ValueError('search public room clock differs from its explicit client contract')
    return declared


def received_prompt_start(client, msg, body, now):
    origin = getattr(client, '_clock_prompt_origin', None)
    if origin is None:
        return now
    if type(origin) is not tuple or len(origin) != 3 or origin[:2] != (msg, bytes(body)) \
            or type(origin[2]) is not int or not 0 < origin[2] / 1e9 <= now:
        raise ValueError('deferred clock cannot refresh or relabel the original prompt arrival')
    return origin[2] / 1e9
