/* A small SHA-256, so the per-prompt menu digest needs no external library.
 *
 * The digest is the comparison unit of the phase (2) contract: 612,254
 * decisions with ~3.6 menu rows each is far too much to ship row-by-row across
 * the process boundary, so each side hashes its own canonical encoding and only
 * the diverging prompts are re-run with full rows (design §4.2).
 */
#ifndef MFENV_SHA256_H_
#define MFENV_SHA256_H_

#include <cstdint>
#include <string>

namespace mfenv {

/* Lowercase hex digest of ``data``. */
std::string Sha256Hex(const std::string& data);

}  // namespace mfenv

#endif  /* MFENV_SHA256_H_ */
