"""Build real features from a small SQLite/Lua corpus, with the pinned builder."""
import io
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from mirrorforce.common.sidecar_io import digest
from mirrorforce.cardrules import rules


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[2] / "friend-ygo-agent-20260920/source/sky-project/scripts/card/build_structured_semantics.py"
    if not path.is_file():
        pytest.skip("local reviewed reference builder was not supplied")
    return path


def fixture(tmp_path):
    cdb, scripts = tmp_path / "cards.cdb", tmp_path / "scripts"
    scripts.mkdir()
    with sqlite3.connect(cdb) as connection:
        connection.execute("CREATE TABLE datas(id INTEGER PRIMARY KEY,alias INTEGER,type INTEGER,atk INTEGER,def INTEGER,level INTEGER,race INTEGER,attribute INTEGER)")
        connection.executemany("INSERT INTO datas VALUES(?,?,?,?,?,?,?,?)",
            [(20, 0, 1, 2000, 1000, 4, 1, 1), (10, 0, 2, 0, 0, 0, 0, 0), (30, 10, 2, 0, 0, 0, 0, 0)])
    (scripts / "c10.lua").write_text("""local s,id=GetID()
function s.initial_effect(c)
local e1=Effect.CreateEffect(c)
e1:SetOperation(s.operation)
c:RegisterEffect(e1)
end
function s.operation(e,tp)
Duel.Draw(tp,1,REASON_EFFECT)
end
""")
    return cdb, scripts


def test_real_builder_keeps_all_codes_and_aliases_without_learned_vectors(tmp_path, builder):
    cdb, scripts = fixture(tmp_path)
    path, report = rules.build(cdb, scripts, builder, tmp_path / "output")
    assert path.is_file() and report["learned_weights"] is False
    codes, exact, effects, mask, metadata = rules.load(path.parent / report["artifact"]["path"], report["artifact"]["sha256"])
    assert codes.tolist() == [0, 10, 20, 30]
    assert exact.shape == (4, 78) and effects.shape == (4, 16, 64)
    assert np.array_equal(effects[1], effects[3]) and mask[1].sum() == 1
    assert not mask[2].any() and metadata["reference_metadata"]["alias_lua_fallback_count"] == 1
    assert metadata["printed_effect_unit_alignment"] == "unknown-pool-all-units"
    assert rules.build(cdb, scripts, builder, tmp_path / "output")[1] == report


def test_unknown_builder_is_rejected_before_its_code_runs(tmp_path):
    bad = tmp_path / "builder.py"
    bad.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="reviewed reference"):
        rules.reference_builder(bad)


def test_script_changes_produce_distinct_bound_artifacts(tmp_path, builder):
    cdb, scripts = fixture(tmp_path)
    _, first = rules.build(cdb, scripts, builder, tmp_path / "output")
    path = scripts / "c10.lua"
    path.write_text(path.read_text().replace("Draw(tp,1", "Draw(tp,2"))
    _, second = rules.build(cdb, scripts, builder, tmp_path / "output")
    assert first["source"]["scripts_sha256"] != second["source"]["scripts_sha256"]
    assert first["artifact"]["sha256"] != second["artifact"]["sha256"]


def test_forged_unknown_row_is_rejected_even_with_a_matching_file_checksum(tmp_path, builder):
    cdb, scripts = fixture(tmp_path)
    path, report = rules.build(cdb, scripts, builder, tmp_path / "output")
    with np.load(path.parent / report["artifact"]["path"], allow_pickle=False) as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    fields["cdb_exact"][0, 0] = 1
    stream = io.BytesIO()
    np.savez_compressed(stream, **fields)
    raw = stream.getvalue()
    changed = tmp_path / "changed.npz"
    changed.write_bytes(raw)
    with pytest.raises(ValueError, match="unknown-card"):
        rules.load(changed, digest(raw))


def test_sorted_codebook_never_silently_reuses_reference_row_numbers(tmp_path):
    cdb, _ = fixture(tmp_path)
    assert rules.card_codes(cdb) == [10, 20, 30]


def test_normalized_lua_hash_is_not_a_unique_card_identity(builder):
    original = rules.reference_builder(builder)
    one = "function s.operation(e,tp)\nDuel.CreateToken(tp,11111111)\nend"
    two = one.replace("11111111", "22222222")
    assert one != two
    a, b = original.canonicalize_lua(one, 100), original.canonicalize_lua(two, 100)
    assert a == b
    assert np.array_equal(original.hash_embedding(original.semantic_tokens(a), 64),
        original.hash_embedding(original.semantic_tokens(b), 64))
