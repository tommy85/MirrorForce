/* The engine error every mfenv component throws (bindings translate it); kept apart from instance.h so components
 * that never touch a duel instance (the public tracker) can throw it without pulling in the instance. */
#ifndef MFENV_DUEL_ERROR_H_
#define MFENV_DUEL_ERROR_H_

#include <stdexcept>
#include <string>

namespace mfenv {

class DuelError : public std::runtime_error {
 public:
  explicit DuelError(const std::string& what) : std::runtime_error(what) {}
};

}  // namespace mfenv

#endif  /* MFENV_DUEL_ERROR_H_ */
