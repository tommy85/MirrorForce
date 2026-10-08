/* The canonical encodings the contract compares, and their version markers.
 *
 * Every layer of the port is verified by hashing a canonical JSON encoding of
 * its output and comparing the digest with the Python twin's.  That makes the
 * encoders themselves a cross-language contract, so they live in one place
 * rather than one copy per layer, and each carries an explicit version:
 * without it, a change to an encoder shows up as a mysterious mismatch in
 * whichever layer happens to be compared first, and a Python side built
 * against the old encoding would look like a porting bug.
 *
 * The Python twins are in ``mirrorforce/mirrorforce/cxxenv/contract.py``; the
 * version constants there and here must be equal, and a test asserts it.
 */
#ifndef MFENV_ENCODING_H_
#define MFENV_ENCODING_H_

#include <cstdint>
#include <string>
#include <vector>

namespace mfenv {

/* Bumped whenever a canonical encoding's field order or content changes.
 * Each encoding writes its own, so one layer can move without invalidating
 * the others' recorded digests. */
/* v2 added ``revealed_unpositioned``: the identities a viewer can name but
 * cannot place (disclosure ledger round 4). */
constexpr int kSnapshotEncodingVersion = 2;
constexpr int kStateEncodingVersion = 1;
constexpr int kMenuEncodingVersion = 1;
constexpr int kMenuRowsEncodingVersion = 1;

/* Python's ``json.dumps(..., ensure_ascii=True)`` string escaping: control
 * characters, the quote and the backslash escape, everything outside printable
 * ASCII becomes ``\uXXXX``. */
void AppendJsonString(const std::string& value, std::string* out);

/* A float32 as its exact bit pattern.  The compared columns are bit-encoded
 * features, so a one-ulp difference *is* a flipped feature bit and a printed
 * decimal would hide it. */
void AppendFloatBits(float value, std::string* out);

/* ``[a,b,c]`` with no spaces, matching ``separators=(",", ":")``. */
void AppendInt64List(const std::vector<int64_t>& values, std::string* out);
void AppendUint32List(const std::vector<uint32_t>& values, std::string* out);
void AppendBoolList(const std::vector<uint8_t>& values, std::string* out);
void AppendFloatBitsList(const std::vector<float>& values, std::string* out);

}  // namespace mfenv

#endif  /* MFENV_ENCODING_H_ */
