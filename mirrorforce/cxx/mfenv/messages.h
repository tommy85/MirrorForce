/* Split one ``get_message`` buffer into individual messages.
 *
 * Line-by-line port of mirrorforce/mirrorforce/puzzle/messages.py (itself a
 * port of ``SingleMode::SinglePlayAnalyze``).  The message count is
 * load-bearing beyond phase (1): ``trace_index = message_cursor + index within the interval``
 * (design §3.3), so a length rule that differs by one byte changes every
 * history token fingerprint downstream.
 */
#ifndef MFENV_MESSAGES_H_
#define MFENV_MESSAGES_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfenv {

/* messages.py:29 -- a message id with no known length rule. */
class UnknownMessage : public std::runtime_error {
 public:
  explicit UnknownMessage(const std::string& what) : std::runtime_error(what) {}
};

struct Message {
  int msg = 0;
  std::vector<uint8_t> payload;
};

/* messages.py:212-228 */
std::vector<Message> SplitMessages(const uint8_t* buf, size_t len);

std::string MessageName(int msg);

}  // namespace mfenv

#endif  /* MFENV_MESSAGES_H_ */
