"""Card extras and effect-unit evidence from a small SQLite/Lua corpus with the pinned builder."""
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from mirrorforce.common import card_exact
from mirrorforce.cardrules import rules, static_extras as X


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[2] / "friend-ygo-agent-20260920/source/sky-project/scripts/card/build_structured_semantics.py"
    if not path.is_file():
        pytest.skip("local reviewed reference builder was not supplied")
    return path


HEADER = "local s,id=GetID()\nfunction s.initial_effect(c)\n"
SCRIPTS = {
    # One literal description per unit, the second unit a clone that overrides it.
    100: HEADER + """local e1=Effect.CreateEffect(c)
e1:SetDescription(aux.Stringid(id,0))
e1:SetType(EFFECT_TYPE_IGNITION)
c:RegisterEffect(e1)
local e2=e1:Clone()
e2:SetDescription(aux.Stringid(id,1))
c:RegisterEffect(e2)
end
""",
    # A clone that inherits: ordinal 0 is carried by both units.
    200: HEADER + """local e1=Effect.CreateEffect(c)
e1:SetDescription(aux.Stringid(id,0))
c:RegisterEffect(e1)
local e2=e1:Clone()
e2:SetCode(EVENT_SPSUMMON_SUCCESS)
c:RegisterEffect(e2)
end
""",
    # Ordinal 1 is also an option label elsewhere; ordinal 0 stays bound.
    300: HEADER + """local e1=Effect.CreateEffect(c)
e1:SetDescription(aux.Stringid(id,0))
c:RegisterEffect(e1)
local e2=Effect.CreateEffect(c)
e2:SetDescription(aux.Stringid(id,1))
e2:SetOperation(s.op)
c:RegisterEffect(e2)
end
function s.op(e,tp)
local opt=Duel.SelectOption(tp,aux.Stringid(id,1),aux.Stringid(id,2))
end
""",
    # A description computed at run time makes the whole card unknown.
    400: HEADER + """for i=0,1 do
local e1=Effect.CreateEffect(c)
e1:SetDescription(aux.Stringid(id,i))
c:RegisterEffect(e1)
end
end
""",
    # A description set on a runtime-created effect in another function is unknown too.
    500: HEADER + """local e1=Effect.CreateEffect(c)
e1:SetDescription(aux.Stringid(id,0))
e1:SetOperation(s.op)
c:RegisterEffect(e1)
end
function s.op(e,tp)
local e2=Effect.CreateEffect(e:GetHandler())
e2:SetDescription(aux.Stringid(id,0))
Duel.RegisterEffect(e2,tp)
end
""",
    # A script may name itself by passcode and by a local alias.
    600: """local s,id=GetID()
local m=600
function s.initial_effect(c)
local e1=Effect.CreateEffect(c)
e1:SetDescription(aux.Stringid(600,3))
c:RegisterEffect(e1)
local e2=Effect.CreateEffect(c)
e2:SetDescription(aux.Stringid(m,4))
c:RegisterEffect(e2)
end
""",
}


def corpus(tmp_path):
    cdb, scripts = tmp_path / "cards.cdb", tmp_path / "scripts"
    scripts.mkdir()
    rows = [(code, 0, 0, 1, 1000, 1000, 4, 1, 1) for code in SCRIPTS]
    rows.append((700, 600, 0, 1, 1000, 1000, 4, 1, 1))           # an alias that runs another card's script
    rows.append((800, 0, 0x1234002a, 0x4000001, 2000, 0x1 | 0x20 | 0x100, 2, 1, 1))  # a link monster with setcodes
    with sqlite3.connect(cdb) as connection:
        connection.execute("CREATE TABLE datas(id INTEGER PRIMARY KEY,alias INTEGER,setcode INTEGER,type INTEGER,"
                           "atk INTEGER,def INTEGER,level INTEGER,race INTEGER,attribute INTEGER)")
        connection.executemany("INSERT INTO datas VALUES(?,?,?,?,?,?,?,?,?)", rows)
    for code, text in SCRIPTS.items():
        (scripts / f"c{code}.lua").write_text(text)
    return cdb, scripts


def tables(tmp_path, builder):
    cdb, scripts = corpus(tmp_path)
    path, report = rules.build(cdb, scripts, builder, tmp_path / "rules")
    table = path.parent / report["artifact"]["path"]
    return cdb, scripts, table, report["artifact"]["sha256"]


def test_evidence_binds_only_literal_fully_accounted_descriptions(tmp_path, builder):
    cdb, scripts, table, checksum = tables(tmp_path, builder)
    path, report = X.build_evidence(table, checksum, cdb, scripts, builder, tmp_path / "out")
    codes = rules.load(table, checksum)[0]
    masks = X.load_evidence(path, report["artifact"]["sha256"], rules_sha256=checksum, codes=codes, units=16)
    row = {int(code): index for index, code in enumerate(codes)}
    assert masks[row[100], :3].tolist() == [0b01, 0b10, 0]
    assert masks[row[200], 0] == 0b11                       # an effect and its clone share the description
    assert masks[row[300], :2].tolist() == [0b01, 0]         # ordinal 1 is reused as an option label
    assert not masks[row[400]].any() and not masks[row[500]].any()
    assert masks[row[600], 3] == 0b01 and masks[row[600], 4] == 0b10
    assert not masks[row[700]].any() and not masks[row[800]].any()
    assert report["registration_order_used"] is False and report["reason_counts"]["bound"] == 6
    with pytest.raises(ValueError, match="rule rows or its law"):
        X.load_evidence(path, report["artifact"]["sha256"], rules_sha256="0" * 64, codes=codes, units=16)


def test_evidence_refuses_scripts_the_rule_table_did_not_pin(tmp_path, builder):
    cdb, scripts, table, checksum = tables(tmp_path, builder)
    (scripts / "c100.lua").write_text(SCRIPTS[100].replace("id,1", "id,2"))
    with pytest.raises(ValueError, match="pins"):
        X.build_evidence(table, checksum, cdb, scripts, builder, tmp_path / "out")


def test_card_extras_carry_arrows_and_setcode_bags_row_aligned(tmp_path, builder):
    cdb, _, table, checksum = tables(tmp_path, builder)
    path, report = X.build_extras(table, checksum, cdb, tmp_path / "out")
    codes = rules.load(table, checksum)[0]
    extras, series = X.load_extras(path, report["artifact"]["sha256"], rules_sha256=checksum, codes=codes)
    exact = card_exact.build_table(cdb)
    link = int(np.flatnonzero(codes == 800)[0])
    arrows = card_exact.SEGMENT_OFFSETS["link_arrows"]
    assert extras[link, :8].tolist() == exact.row(800)[arrows:arrows + 8].tolist() == [1, 0, 0, 0, 1, 0, 0, 1]
    assert np.isclose(np.linalg.norm(extras[link, 8:]), 1.0) and not extras[0].any()
    assert series[link].tolist() == [0x002a, 0x1234, 0, 0] and not series[0].any()
    with pytest.raises(ValueError, match="rule rows"):
        X.load_extras(path, report["artifact"]["sha256"], rules_sha256=checksum, codes=codes[:-1])
