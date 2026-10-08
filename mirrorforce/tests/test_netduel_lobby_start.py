"""A client that may host a room starts the duel once the other duelist is ready, and only then."""

import socket
import struct
import threading

import pytest

from mirrorforce.netduel import protocol as P
from mirrorforce.netduel.client import NetDuelClient

HOST_INFO = struct.pack('<IBBBBB3xiBBH', 0x7dfcee6a, 0, 0, 5, 0, 0, 8000, 5, 1, 600)


def lobby(start_when_ready, script):
    """Run a client against a scripted host; return every packet the client sent after it joined."""
    server = socket.socket()
    server.bind(('127.0.0.1', 0))
    server.listen(1)
    sent = []

    def host():
        connection, _ = server.accept()
        stream = P.PacketStream(connection)
        for _ in range(2):  # PLAYER_INFO, JOIN_GAME
            stream.recv(timeout=5)
        for step in script:
            if step == 'read':
                sent.append(stream.recv(timeout=5)[0])
            elif step == 'quiet':  # anything the client sends now lands in ``sent`` and fails the test
                try:
                    sent.append(stream.recv(timeout=.3)[0])
                except socket.timeout:
                    pass
            else:
                stream.send(*step)
        connection.close()

    thread = threading.Thread(target=host)
    thread.start()
    client = NetDuelClient('127.0.0.1', server.getsockname()[1], 'MirrorForce', [1] * 40, [], policy=None,
                           timeout=5, password='MF1', start_when_ready=start_when_ready)
    result = client.run()
    thread.join(5)
    server.close()
    return sent, result


def ready(position):
    return (P.STOC.HS_PLAYER_CHANGE, bytes([(position << 4) | P.PLAYERCHANGE.READY]))


def not_ready(position):
    return (P.STOC.HS_PLAYER_CHANGE, bytes([(position << 4) | P.PLAYERCHANGE.NOTREADY]))


def test_the_host_starts_when_the_other_duelist_is_ready_and_again_after_a_refused_start():
    script = [(P.STOC.JOIN_GAME, HOST_INFO), (P.STOC.TYPE_CHANGE, bytes([0x10])), 'read', 'read',
              (P.STOC.HS_PLAYER_ENTER, P.encode_name('Guest') + bytes([1])), 'quiet',
              ready(1), 'read', not_ready(1), ready(1), 'read']
    sent, result = lobby(True, script)
    assert sent == [P.CTOS.UPDATE_DECK, P.CTOS.HS_READY, P.CTOS.HS_START, P.CTOS.HS_START]
    assert result.error  # the scripted host closes instead of dealing


@pytest.mark.parametrize('start_when_ready, type_change', [(False, 0x10), (True, 0x01)])
def test_a_guest_or_a_client_that_did_not_opt_in_never_starts(start_when_ready, type_change):
    script = [(P.STOC.JOIN_GAME, HOST_INFO), (P.STOC.TYPE_CHANGE, bytes([type_change])), 'read', 'read',
              ready(1 - (type_change & 0xF)), 'quiet']
    sent, _ = lobby(start_when_ready, script)
    assert sent == [P.CTOS.UPDATE_DECK, P.CTOS.HS_READY]
