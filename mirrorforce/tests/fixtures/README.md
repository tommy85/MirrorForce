# Test fixtures

Data that must be committed together with the tests. Regression tests must never pin an index into generated data
that is not committed: once the generated pool changes, the index points to another game, the assertion sees nothing,
and the failure looks like broken functionality while it is only the fixture that drifted.

| File | Origin | Why it is committed |
|---|---|---|
| `synth-00260.ydk` | Deck 260 of a synthetic deck pool generated from seed `20260823` | It is the only deck of that pool containing Tierra, Source of Destruction (91588074), which `test_tierra_override` needs |
