"""Admitted total BO2, dynamic token mapping, paired terminal and real wire continuity."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import socket
import struct
import threading
from types import SimpleNamespace

import pytest

from mirrorforce.netduel import constants as C, protocol as P
from mirrorforce.netduel.client import NetDuelClient
from mirrorforce.netduel.league import Credentials, Enrollment, LeagueConnection
from mirrorforce.netduel.agent_policy import RemotePolicy
from tools import mf_runtime_league_client as T
from test_netduel_league import Policy


def credentials():
    return Credentials("127.0.0.1",6009,True,tuple(Enrollment("player"+str(i),
        ("RoomToken"+str(i)+"A","RoomToken"+str(i)+"B")) for i in (1,2,3)))


def plan():
    return {"schema":T.SCHEMA,"mode":"test","approved_total_bo2":30,"finals_allowed":False,
        "rooms":[{"id":"m-01","session_id":"s-one","bo2":10,"server_bo2_limit":10,
                  "members":[{"player":"player1","credential_index":0,"side":"a"},
                             {"player":"player2","credential_index":0,"side":"b"}]},
                 {"id":"m-02","session_id":"s-two","bo2":10,"server_bo2_limit":10,
                  "members":[{"player":"player1","credential_index":1,"side":"a"},
                             {"player":"player3","credential_index":0,"side":"b"}]},
                 {"id":"m-03","session_id":"s-three","bo2":10,"server_bo2_limit":10,
                  "members":[{"player":"player2","credential_index":1,"side":"a"},
                             {"player":"player3","credential_index":1,"side":"b"}]}],
        "service":{"socket":"/registered.sock","identity":{"path":"/registered.json","sha256":"a"*64},
                   "actor_sha256":"b"*64},"deck":{"path":"/deck","sha256":"c"*64},
        "cards_db":{"path":"/cards","sha256":"d"*64},"seed":11,"max_seconds":7200,"timeout_seconds":600}


@pytest.mark.parametrize("fault",["quota","final","partial","duplicate_token","duplicate_identity","unknown"])
def test_changed_quota_final_or_credential_mapping_refuses_before_any_join(fault):
    value=plan()
    if fault=="quota":value["approved_total_bo2"]=10
    elif fault=="final":value["finals_allowed"]=True
    elif fault=="partial":value["rooms"][0]["bo2"]=3
    elif fault=="duplicate_token":value["rooms"][1]["members"][0]["credential_index"]=0
    elif fault=="duplicate_identity":value["rooms"][0]["members"][1]["player"]="player1"
    else:value["network_secret"]="forbidden"
    with pytest.raises(ValueError):T.validate_plan(value,credentials())


def views(value):
    rows=[]
    for room in value["rooms"]:
        for game in range(1,room["bo2"]*2+1):
            for member in room["members"]:
                seat=(0 if member["side"]=="a" else 1) if game%2 else (1 if member["side"]=="a" else 0)
                rows.append({"room":room["id"],"game":game,"player":member["player"],"natural_terminal":True,
                    "result":{"error":"","our_player":seat,"winner":0,"win_reason":1,"lp":[8000,0]}})
    return rows


def test_three_parallel_pairs_count_sixty_duels_thirty_bo2_not_120_viewer_records():
    value=plan()
    assert len(T.validate_plan(value,credentials()))==6
    rows=views(value); result=T.pair_results(rows,value["rooms"])
    assert len(rows)==120 and result["completed_duels"]==60 and result["completed_bo2"]==30
    assert T.pair_results(rows[:-1],value["rooms"])["completed_bo2"]==29


@pytest.mark.parametrize("fault",["swap","winner","lp","administrative","duplicate"])
def test_actual_seat_swap_and_two_viewer_terminal_must_agree(fault):
    value=plan();rows=views(value)
    if fault=="swap":
        rows[0]["result"]["our_player"]=1;rows[1]["result"]["our_player"]=0
    elif fault=="winner":rows[1]["result"]["winner"]=1
    elif fault=="lp":rows[1]["result"]["lp"][0]=7000
    elif fault=="administrative":rows[0]["natural_terminal"]=False
    else:rows.append(copy.deepcopy(rows[0]))
    if fault=="duplicate":
        assert T.pair_results(rows,value["rooms"])["completed_bo2"]==29
    else:
        with pytest.raises(ValueError):T.pair_results(rows,value["rooms"])


def test_persistent_socket_plays_two_games_and_fixed_deck_side_handshake_without_rejoin():
    server=socket.socket();server.bind(("127.0.0.1",0));server.listen(1)
    received=[]; errors=[]
    def host():
        try:
            sock,_=server.accept();stream=P.PacketStream(sock)
            received.extend(stream.recv(3) for _ in range(2))
            info=struct.pack("<IBBBBB3xiBBH",0,0,1,4,1,0,8000,5,1,600)
            stream.send(P.STOC.JOIN_GAME,info)
            stream.send(P.STOC.TYPE_CHANGE,b"\x00")
            received.extend(stream.recv(3) for _ in range(2))
            for seat in (0,1):
                start=bytes([C.MSG_START,seat,4])+struct.pack("<iiHHHH",8000,8000,40,15,40,15)
                stream.send(P.STOC.GAME_MSG,start)
                stream.send(P.STOC.GAME_MSG,bytes([C.MSG_NEW_TURN,seat]))
                stream.send(P.STOC.GAME_MSG,bytes([C.MSG_LPUPDATE,1-seat])+struct.pack("<I",0))
                stream.send(P.STOC.GAME_MSG,bytes([C.MSG_WIN,seat,1]))
                if seat==0:
                    stream.send(P.STOC.CHANGE_SIDE)
                    received.append(stream.recv(3))
            sock.close()
        except Exception as exc:errors.append(exc)
        finally:server.close()
    thread=threading.Thread(target=host);thread.start()
    created=[];rows=[]
    def factory(index):
        client=NetDuelClient("127.0.0.1",server.getsockname()[1],"player1",[1]*40,[100],Policy(index),
            password="RoomTokenForTest",allow_match_mode=True,timeout=3,capture=[])
        created.append(client);return client
    connection=LeagueConnection(factory,room="m-01",player="player1",games=2,on_game=rows.append,timeout=3)
    assert len(connection.run())==2
    thread.join(5)
    assert not thread.is_alive() and not errors
    assert [op for op,_ in received]==[P.CTOS.PLAYER_INFO,P.CTOS.JOIN_GAME,P.CTOS.UPDATE_DECK,
                                      P.CTOS.HS_READY,P.CTOS.UPDATE_DECK]
    assert [row["result"]["our_player"] for row in rows]==[0,1]
    assert len(created)==2 and created[0].policy is not created[1].policy


def test_view_capture_is_not_aliased_into_remote_policy_observer_twice():
    identity={"protocol":"mirrorforce_policy_service/v1","client_config":{"public_opponent_recipe":False}}
    policy=RemotePolicy("unix:/unused",identity)
    policy.sock=SimpleNamespace()
    policy._call=lambda request:{"session":"s-test"}
    client=NetDuelClient("unused",6009,"player1",[1]*40,[100],policy,capture=[])
    policy.validate_client(client)
    payload=bytes([C.MSG_START,0,4])+struct.pack("<iiHHHH",8000,8000,40,15,40,15)
    client._handle(P.STOC.GAME_MSG,payload)
    assert len(client.capture)==len(policy.capture)==1 and client.capture is not policy.capture


def resume_plan(tmp_path):
    value=plan(); value["schema"]=T.RESUME_SCHEMA; value["mode"]="competition"
    value["rooms"]=[copy.deepcopy(room) for room in value["rooms"]]
    for room in value["rooms"]:
        room["members"]=room["members"][:1]
        room["resume"]={"profile":"official-special-preserving/v1","start_game":1,
                        "remaining_games":20,"consumed":[]}
    first=value["rooms"][1]
    evidence=tmp_path/"official-game1.json"
    evidence.write_text(json.dumps({"schema":"mirrorforce_league_official_consumed_evidence/v1",
        "room":first["id"],"session_id":first["session_id"],"absolute_game":1,
        "registered_side":first["members"][0]["side"],"official_result":{"game_number":1,"winner":"b","reason":0},
        "classification":"official_special_or_administrative_not_natural"}))
    first["resume"]={"profile":"official-special-preserving/v1","start_game":2,"remaining_games":19,
        "consumed":[{"game":1,"session_id":first["session_id"],"side":first["members"][0]["side"],
                     "winner":"b","win_reason":0,"natural_terminal":False,
                     "evidence":{"path":str(evidence),"sha256":hashlib.sha256(evidence.read_bytes()).hexdigest()}}]}
    return value


def competition_credentials():
    return Credentials("127.0.0.1",6009,False,(Enrollment("player1",("RoomToken1A","RoomToken1B","RoomToken1C")),))


def test_resume_contract_binds_consumed_proof_identity_count_and_three_room_starts(tmp_path):
    value=resume_plan(tmp_path)
    # Make the ONE identity/token mapping explicit in all three rooms.
    for index,room in enumerate(value["rooms"]):
        room["members"]=[{"player":"player1","credential_index":index,"side":room["members"][0]["side"]}]
    assert len(T.validate_plan(value,competition_credentials()))==3
    assert [room["resume"]["start_game"] for room in value["rooms"]]==[1,2,1]
    for fault in ("hash","identity","count"):
        broken=copy.deepcopy(value)
        consumed=broken["rooms"][1]["resume"]["consumed"][0]
        if fault=="hash": consumed["evidence"]["sha256"]="0"*64
        elif fault=="identity": consumed["session_id"]="different-session"
        else: broken["rooms"][1]["resume"]["start_game"]=3
        with pytest.raises(ValueError): T.validate_plan(broken,competition_credentials())


@pytest.mark.parametrize("fault",["room","session","game","side","winner","reason","unrelated","array","bool_game","classification"])
def test_consumed_evidence_semantics_are_bound_not_only_its_hash(tmp_path,fault):
    value=resume_plan(tmp_path)
    for index,room in enumerate(value["rooms"]):
        room["members"]=[{"player":"player1","credential_index":index,"side":room["members"][0]["side"]}]
    consumed=value["rooms"][1]["resume"]["consumed"][0]
    path=tmp_path/(fault+".json")
    evidence=json.loads(Path(consumed["evidence"]["path"]).read_text())
    if fault=="unrelated": evidence={"official":True}
    elif fault=="array": evidence=[]
    elif fault=="bool_game": evidence["absolute_game"]=True
    elif fault=="classification": evidence["classification"]="official_natural"
    elif fault in ("room","session","side"): evidence[{"room":"room","session":"session_id","side":"registered_side"}[fault]]="wrong"
    elif fault=="game": evidence["absolute_game"]=9
    else: evidence["official_result"][fault]=9 if fault=="reason" else "a"
    path.write_text(json.dumps(evidence)); consumed["evidence"]={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError): T.validate_plan(value,competition_credentials())


@pytest.mark.parametrize("reason,natural",[(3,True),(1,False),(4,False)])
def test_consumed_contract_reason_and_natural_classification_cannot_disagree(tmp_path,reason,natural):
    value=resume_plan(tmp_path)
    for index,room in enumerate(value["rooms"]):
        room["members"]=[{"player":"player1","credential_index":index,"side":room["members"][0]["side"]}]
    row=value["rooms"][1]["resume"]["consumed"][0]; row["win_reason"]=reason; row["natural_terminal"]=natural
    with pytest.raises(ValueError): T.validate_plan(value,competition_credentials())


def test_resume_game2_uses_swapped_seat_and_reason0_remains_non_natural():
    rows=[]
    connection=LeagueConnection(lambda index: NetDuelClient("unused",6009,"player1",[1]*40,[100],Policy(index),
        password="RoomTokenForTest",allow_match_mode=True,timeout=3,capture=[]), room="m-04",player="player1",
        games=19,start_game=2,on_game=rows.append,timeout=3,admit_special=True)
    connection.stream=SimpleNamespace()
    connection.client.stream=connection.stream
    connection.client._handle(P.STOC.GAME_MSG,bytes([C.MSG_START,0,4])+struct.pack("<iiHHHH",8000,8000,40,15,40,15))
    connection.in_game=True; connection.started_at=0.0
    connection.client._handle(P.STOC.GAME_MSG,bytes([C.MSG_WIN,0,0]))
    connection._finish()
    assert rows[0]["game"]==2 and rows[0]["result"]["our_player"]==0
    assert rows[0]["natural_terminal"] is False and rows[0]["terminal_kind"]=="special_or_administrative"


def test_subset_resume_is_bound_to_pinned_full_plan_and_only_approved_timeup(tmp_path):
    full=resume_plan(tmp_path)
    for index,room in enumerate(full["rooms"]):
        room["members"]=[{"player":"player1","credential_index":index,"side":room["members"][0]["side"]}]
    raw=json.dumps(full,sort_keys=True).encode(); parent=tmp_path/"parent.json"; parent.write_bytes(raw)
    value=copy.deepcopy(full); value["schema"]=T.SUBSET_SCHEMA; value["rooms"]=[value["rooms"][2]]
    evidence=tmp_path/"m03-timeup.json"; evidence.write_text(json.dumps({
        "schema":"mirrorforce_league_official_consumed_evidence/v1","room":"m-03","session_id":"s-three",
        "absolute_game":1,"registered_side":"a","official_result":{"game_number":1,"winner":"a","reason":3},
        "classification":"official_timeup_not_natural"}))
    value["rooms"][0]["resume"]={"profile":"official-special-preserving/v2","start_game":2,"remaining_games":19,
        "consumed":[{"game":1,"session_id":"s-three","side":"a","winner":"a","win_reason":3,
                     "natural_terminal":False,"evidence":{"path":str(evidence),
                         "sha256":hashlib.sha256(evidence.read_bytes()).hexdigest()}}]}
    value["resume_subset"]={"parent_plan":{"path":str(parent),"sha256":hashlib.sha256(raw).hexdigest()},
                            "selected_room_ids":["m-03"]}
    assert len(T.validate_plan(value,competition_credentials()))==1
    broken=copy.deepcopy(value); broken["resume_subset"]["selected_room_ids"]=["m-02"]
    with pytest.raises(ValueError): T.validate_plan(broken,competition_credentials())


def test_lobby_wait_is_explicit_subset_only_and_does_not_change_in_game_timeout(tmp_path,monkeypatch):
    full=resume_plan(tmp_path)
    for index,room in enumerate(full["rooms"]):
        room["members"]=[{"player":"player1","credential_index":index,"side":room["members"][0]["side"]}]
    raw=json.dumps(full,sort_keys=True).encode(); parent=tmp_path/"parent-lobby.json"; parent.write_bytes(raw)
    value=copy.deepcopy(full); value["schema"]=T.LOBBY_SUBSET_SCHEMA; value["rooms"]=[value["rooms"][0]]
    value["resume_subset"]={"parent_plan":{"path":str(parent),"sha256":hashlib.sha256(raw).hexdigest()},
                            "selected_room_ids":["m-01"]}
    value["lobby_wait_until_deadline"]=True
    assert len(T.validate_plan(value,competition_credentials()))==1
    broken=copy.deepcopy(value); broken["lobby_wait_until_deadline"]=False
    with pytest.raises(ValueError): T.validate_plan(broken,competition_credentials())

    now=100.0; monkeypatch.setattr("mirrorforce.netduel.league.time.monotonic",lambda:now)
    def connection(**changes):
        return LeagueConnection(lambda index: NetDuelClient("unused",6009,"player1",[1]*40,[100],Policy(index),
            password="RoomTokenForTest",allow_match_mode=True,timeout=3,capture=[]),room="m",player="p",games=2,
            on_game=lambda row:None,timeout=600,deadline=1000,lobby_wait_until_deadline=True,**changes)
    waiting=connection(); assert waiting.receive_timeout()==900  # may remain in an empty lobby beyond 600
    waiting.in_game=True; assert waiting.receive_timeout()==600  # started duel keeps the old receive timeout
    other=connection(); other.deadline=90; assert other.receive_timeout()==-10  # whole-run deadline still terminates it
    assert waiting.deadline==1000 and other.deadline==90  # independent rooms do not share mutable wait state


@pytest.mark.parametrize("reason,admitted",[(0,True),(3,True),(4,False),(16,False)])
def test_resume_special_profile_only_admits_surrender_and_timeup(reason,admitted):
    connection=LeagueConnection(lambda index: NetDuelClient("unused",6009,"player1",[1]*40,[100],Policy(index),
        password="RoomTokenForTest",allow_match_mode=True,timeout=3,capture=[]), room="m-06",player="player1",
        games=19,start_game=2,on_game=lambda row:None,timeout=3,admit_special=True)
    record={"natural_terminal":False,"result":{"win_reason":reason}}
    assert connection._terminal_admitted(record) is admitted


@pytest.mark.parametrize("natural,allow",[(True,False),(False,True)])
def test_accept_only_allows_zero_prompt_clean_reports_for_official_special(monkeypatch,natural,allow):
    seen=[]
    monkeypatch.setattr(T,"check_report",lambda report,identity,result,**kw:seen.append(kw))
    run=object.__new__(T.LeagueRun); run.identity={}; run.rows=[]; run.references=[]; run.lock=threading.Lock()
    run.plan={"rooms":[{"id":"m","bo2":1,"members":[{"player":"player1","side":"a"}]}]}
    run.publish=lambda category,value:{"path":"/evidence","sha256":"a"*64}
    row={"room":"m","player":"player1","game":1,"natural_terminal":natural,"policy_report":{},
         "terminal_evidence":"received_MSG_WIN","result":{"our_player":0,"win_reason":1 if natural else 3}}
    run.accept(row)
    assert seen==[{"allow_forced_only_terminal":allow}]
