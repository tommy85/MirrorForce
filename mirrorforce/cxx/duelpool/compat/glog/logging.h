// Copyright 2026 MirrorForce. Minimal stand-in for the glog macros the
// imported envpool headers use (LOG, DLOG, CHECK_*, DCHECK_*).
//
// The system libglog/libgflags static archives are not position independent,
// so they cannot be linked into the extension, and the training hosts have no
// shared glog. Semantics follow glog: LOG(FATAL) and a failed CHECK abort after
// printing; DLOG/DCHECK compile away under NDEBUG.
#ifndef DUELPOOL_COMPAT_GLOG_LOGGING_H_
#define DUELPOOL_COMPAT_GLOG_LOGGING_H_

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace duelpool_log {

class Line {
 public:
  Line(const char* severity, const char* file, int line)
      : fatal_(std::string(severity) == "FATAL") {
    stream_ << severity[0] << ' ' << file << ':' << line << "] ";
  }
  ~Line() {
    std::cerr << stream_.str() << std::endl;
    if (fatal_) {
      std::abort();
    }
  }
  template <typename T>
  Line& operator<<(const T& value) {
    stream_ << value;
    return *this;
  }

 private:
  bool fatal_;
  std::ostringstream stream_;
};

class Null {
 public:
  template <typename T>
  Null& operator<<(const T&) {
    return *this;
  }
};

}  // namespace duelpool_log

#define LOG(severity) ::duelpool_log::Line(#severity, __FILE__, __LINE__)
#define DUELPOOL_CHECK_OP(a, b, op)                                        \
  if ((a)op(b)) {                                                        \
  } else /* NOLINT */                                                    \
    ::duelpool_log::Line("FATAL", __FILE__, __LINE__) << "Check failed: " #a " " #op " " #b " "
#define CHECK_EQ(a, b) DUELPOOL_CHECK_OP(a, b, ==)
#define CHECK_NE(a, b) DUELPOOL_CHECK_OP(a, b, !=)
#define CHECK_LT(a, b) DUELPOOL_CHECK_OP(a, b, <)
#define CHECK_LE(a, b) DUELPOOL_CHECK_OP(a, b, <=)
#define CHECK_GT(a, b) DUELPOOL_CHECK_OP(a, b, >)
#define CHECK_GE(a, b) DUELPOOL_CHECK_OP(a, b, >=)

#ifdef NDEBUG
#define DUELPOOL_DISCARD \
  if (true) {          \
  } else /* NOLINT */  \
    ::duelpool_log::Null()
#define DLOG(severity) DUELPOOL_DISCARD
#define DCHECK_EQ(a, b) DUELPOOL_DISCARD
#define DCHECK_NE(a, b) DUELPOOL_DISCARD
#define DCHECK_LT(a, b) DUELPOOL_DISCARD
#define DCHECK_LE(a, b) DUELPOOL_DISCARD
#define DCHECK_GT(a, b) DUELPOOL_DISCARD
#define DCHECK_GE(a, b) DUELPOOL_DISCARD
#else
#define DLOG(severity) LOG(severity)
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#define DCHECK_LE(a, b) CHECK_LE(a, b)
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#define DCHECK_GE(a, b) CHECK_GE(a, b)
#endif

#endif  // DUELPOOL_COMPAT_GLOG_LOGGING_H_
