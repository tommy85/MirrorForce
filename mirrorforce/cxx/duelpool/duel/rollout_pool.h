// The continuation pool of search (search plan sections 4-5): K continuations of a root, each optionally starting
// with a given root action, stepped in batch under the caller's policy for both players to a depth in the root player's
// own decisions -- the shape of update-equivalence search (Ataraxos, Sokota et al. 2026: for each candidate root action,
// roll out to depth d with the policy for both sides, average the network's leaf values, take one KL-regularized
// mirror-descent step at the root; mirrorforce/agent/search/update.py). It runs on the true state now; on ReplaceHidden
// particle worlds once a particle carries the replaced player's private past (S3, search plan section 3.1).
//
// A root is a collection game at a decision point, handed over as its record (DuelEnvImpl::root_record: outer seed,
// both decks as dealt, the menu indices so far, the core stream digest, the player to move, the env generator as its
// play began). A core snapshot cannot be loaded into another duel (an mfsnap arena restores only to its own
// addresses), so a slot clones a root by replay: a SearchDuel from the record, play starting from the record's env
// generator, replays the indices, must reach the record's stream digest and player to move (otherwise the pool fails:
// a determinism defect), and takes one S0 snapshot. A root may occupy up to ``width`` slots at once, each its own clone.
//
// A continuation restores the snapshot, calls reshuffle_future with its seed (from pool seed, root id, k: the unrevealed
// deck orders, the core's future seed and the env generator are redrawn from the pool's stream, so no continuation
// foresees the real game's draws or reuses its random state), takes its root action if it has one (a menu index of the
// root's shown menu), then is stepped by the caller: every decision of both players is observed, each flagged whether
// it is the root player's own and its index among the root player's decisions since the root action. The decision with
// own index ``depth`` - 1 is the leaf (``cut``: evaluate its value, do not act); depth 0 runs to the game's end. The
// collection env is only read for its record.
//
// Slots prepare, observe and step in parallel on the pool's threads; a batch holds every live slot, across roots.
// Timings are kept per phase for the cost report (tools/mf_runtime_rollout_bench.py).
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "duel/search_api.h"
#include "envpool/BS_thread_pool.h"

namespace duelenv {

class RolloutPool {
 public:
  using State = SearchDuel::State;

  struct Root {
    int64_t id = 0;
    ScriptedDeal deal;
    std::vector<int> actions;  // the record's menu indices up to the root
    uint64_t stream_hash = 0;
    int to_play = 0;
    std::string play_gen;           // the env generator as the game's play began
    std::vector<int> root_actions;  // one per continuation: the root menu index it starts with, -1 none
    int depth = 0;                  // the root player's own decisions to the leaf; 0: to the game's end
    int width = 1;                  // slots this root may occupy at once
  };

  // A live slot at a decision point (Observe).
  struct Live {
    int slot = 0;
    int64_t root = 0;
    int continuation = 0;
    int root_action = -1;
    int to_play = 0;
    bool own = false;    // the root player's decision
    int own_index = -1;  // its index among the root player's decisions since the root action (own only)
    bool fresh = false;  // the first decision of a continuation: the caller resets this slot's memory
    bool cut = false;    // the leaf (own index depth - 1): evaluate, do not act
  };

  // One finished continuation.
  struct Result {
    int64_t root = 0;
    int continuation = 0;
    uint64_t seed = 0;  // its reshuffle_future seed: the continuation reproduces exactly
    int root_action = -1;
    int root_player = 0;
    int depth = 0;
    int winner = -1;  // 0 or 1; 2 a draw; -1 cut at the leaf
    int win_reason = -1;
    int decisions = 0;      // decisions stepped after the root action (both players)
    int own_decisions = 0;  // the root player's decisions observed (the leaf included)
    bool truncated = false;
    std::string error;      // the env's error that ended it (a card script error and the like): no outcome
  };

  // One clone of a root: its replay length and costs, or why its continuations could not start.
  struct RootLog {
    int64_t root = 0;
    int replay_steps = 0;
    double replay = 0, snapshot = 0;
    std::string refused;
  };

  struct Timings {
    double replay = 0, snapshot = 0, restore = 0, reshuffle = 0, root_action = 0, observe = 0, step = 0;
    int64_t clones = 0, continuations = 0, steps = 0, replay_steps = 0, errors = 0;
  };

  RolloutPool(DuelEnvSpec spec, int slots, int threads, uint64_t seed)
      : spec_(std::move(spec)), slots_(static_cast<size_t>(slots)), seed_(seed), pool_(static_cast<size_t>(threads)) {
    if (slots < 1 || threads < 1) throw std::runtime_error("a rollout pool needs at least one slot and one thread");
  }

  int64_t Submit(Root root) {
    if (root.root_actions.empty() || root.depth < 0 || root.width < 1)
      throw std::runtime_error("a root needs continuations, depth >= 0 and width >= 1");
    root.id = next_id_++;
    RootState state;
    for (size_t k = 0; k < root.root_actions.size(); ++k) state.pending.push_back(static_cast<int>(k));
    state.root = std::move(root);
    const int64_t id = state.root.id;
    roots_.emplace(id, std::move(state));
    order_.push_back(id);
    return id;
  }

  // Continuations not yet started, and live slots.
  size_t queued() const {
    size_t n = 0;
    for (const auto &[id, r] : roots_) n += r.pending.size();
    return n;
  }
  size_t live() const {
    size_t n = 0;
    for (const Slot &s : slots_) n += s.duel != nullptr;
    return n;
  }

  // Fills idle slots (a clone of a root with continuations to start, up to its width), then returns every live slot
  // and its observation (the player to move's).
  std::vector<Live> Observe(std::vector<State> *states) {
    std::vector<size_t> starting;
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i].duel || slots_[i].root >= 0) continue;
      RootState *r = NextRoot();
      if (!r) break;
      slots_[i].root = r->root.id;
      slots_[i].continuation = r->pending.front();
      r->pending.pop_front();
      ++r->active;
      starting.push_back(i);
    }
    Parallel(starting, [this](size_t i) { Clone(slots_[i]); });
    for (size_t i : starting)
      if (!slots_[i].refusal.empty()) Release(slots_[i], true);
    Settle();
    std::vector<size_t> active;
    for (size_t i = 0; i < slots_.size(); ++i)
      if (slots_[i].duel) active.push_back(i);
    std::vector<Live> out(active.size());
    states->clear();
    states->resize(active.size());
    const auto t0 = Clock::now();
    std::vector<size_t> index(active.size());
    for (size_t j = 0; j < active.size(); ++j) index[j] = j;
    Parallel(index, [&](size_t j) {
      Slot &s = slots_[active[j]];
      const Root &root = roots_.at(s.root).root;
      (*states)[j] = s.duel->Observe();
      const bool own = s.duel->player() == root.to_play;
      out[j] = Live{static_cast<int>(active[j]), s.root, s.continuation, root.root_actions[s.continuation],
                    s.duel->player(), own, own ? s.own : -1, s.fresh, own && root.depth > 0 && s.own == root.depth - 1};
      s.fresh = false;
    });
    timings_.observe += Seconds(t0);
    return out;
  }

  // Steps the named live slots (the actions index their shown menus; a cut slot's action is ignored). A continuation
  // that ends -- the game over, or its leaf -- is recorded; the slot starts its root's next continuation, or is
  // released when none is left to start.
  void Step(const std::vector<int> &slots, const std::vector<int> &actions) {
    if (slots.size() != actions.size()) throw std::runtime_error("one action per stepped slot");
    std::vector<size_t> index(slots.size());
    for (size_t j = 0; j < slots.size(); ++j) {
      if (slots[j] < 0 || static_cast<size_t>(slots[j]) >= slots_.size() || !slots_[slots[j]].duel)
        throw std::runtime_error("step of an idle or unknown slot");
      index[j] = j;
    }
    const auto t0 = Clock::now();
    Parallel(index, [&](size_t j) {
      Slot &s = slots_[slots[j]];
      const Root &root = roots_.at(s.root).root;
      const bool own = s.duel->player() == root.to_play;
      const bool cut = own && root.depth > 0 && s.own == root.depth - 1;
      if (cut) {
        ++s.own;
      } else {
        try {
          s.duel->Step(actions[j]);  // an index of the shown menu (the observation written by Observe)
        } catch (const std::runtime_error &error) {
          if (std::string(error.what()).rfind("menu row ", 0) == 0) throw;  // not an index of the shown menu
          // the env's error (a card script error and the like) ends the continuation without an outcome, counted;
          // the next one restores the root's snapshot, which replaces the whole duel state
          s.ended = Finish(s, false, error.what());
          return;
        }
        ++s.steps;
        s.own += own;
      }
      if (cut || s.duel->finished()) s.ended = Finish(s, cut);
    });
    timings_.step += Seconds(t0);
    timings_.steps += static_cast<int64_t>(slots.size());
    Settle();
  }

  std::vector<Result> TakeResults() { return std::exchange(results_, {}); }
  std::vector<RootLog> TakeRootLog() { return std::exchange(logs_, {}); }
  Timings timings() const { return timings_; }

 private:
  using Clock = std::chrono::steady_clock;

  struct RootState {
    Root root;
    std::deque<int> pending;  // continuations not yet started
    int active = 0;           // slots holding a clone of it
  };

  struct Slot {
    std::unique_ptr<SearchDuel> duel;
    std::shared_ptr<SearchDuel::Snapshot> snapshot;
    int64_t root = -1;
    int continuation = 0;
    int steps = 0;
    int own = 0;
    bool fresh = false;
    std::string refusal;
    std::unique_ptr<Result> ended;  // set when its continuation ended (Settle starts the next one)
  };

  static double Seconds(Clock::time_point since) {
    return std::chrono::duration<double>(Clock::now() - since).count();
  }

  // A continuation's reshuffle_future seed from (pool seed, root id, k): SplitMix64 steps, the pool's own stream.
  uint64_t ContinuationSeed(int64_t root, int k) const {
    auto mix = [](uint64_t z) {
      z += 0x9E3779B97F4A7C15ULL;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      return z ^ (z >> 31);
    };
    return mix(mix(mix(seed_) ^ static_cast<uint64_t>(root)) ^ static_cast<uint64_t>(k));
  }

  // The first root (submission order) with a continuation to start and room for another clone.
  RootState *NextRoot() {
    for (int64_t id : order_) {
      RootState &r = roots_.at(id);
      if (!r.pending.empty() && r.active < r.root.width) return &r;
    }
    return nullptr;
  }

  // Clone the slot's root by replay (it must reach the record's stream digest and player to move), snapshot it, and
  // start the slot's continuation.
  void Clone(Slot &s) {
    const Root &root = roots_.at(s.root).root;
    auto t0 = Clock::now();
    s.duel = std::make_unique<SearchDuel>(spec_, root.deal, false);
    s.duel->Start();
    s.duel->SetGenerator(root.play_gen);  // the source's env generator from where its play began
    for (int action : root.actions) {
      if (s.duel->finished()) throw std::runtime_error("rollout root: the replay ended before the root");
      s.duel->Step(action);  // writes each observation first, as the env does: the guards' shown menus replay too
    }
    const double replay = Seconds(t0);
    // a replay that misses the record is a determinism defect, never a refusal
    if (s.duel->finished() || s.duel->stream_hash() != root.stream_hash || s.duel->player() != root.to_play)
      throw std::runtime_error("rollout root " + std::to_string(root.id) + ": the replayed game differs from the "
                               "record (stream digest or player to move)");
    t0 = Clock::now();
    s.snapshot = s.duel->Take();
    const double snapshot = Seconds(t0);
    RootLog log{root.id, static_cast<int>(root.actions.size()), replay, snapshot, ""};
    try {
      Begin(s, false);
    } catch (const std::runtime_error &error) {
      // the root's menu names a deck place of the opponent, or the like: no continuation can redraw the future
      log.refused = std::string("reshuffle: ") + error.what();
      s.refusal = log.refused;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    timings_.replay += replay;
    timings_.snapshot += snapshot;
    timings_.clones += 1;
    timings_.replay_steps += static_cast<int64_t>(root.actions.size());
    logs_.push_back(log);
  }

  // Start the slot's continuation from its root: restore (unless the clone is fresh), reshuffle_future, the root
  // action. A game that ends on the root action is a finished continuation right away.
  void Begin(Slot &s, bool restore) {
    const Root &root = roots_.at(s.root).root;
    double restore_seconds = 0;
    auto t0 = Clock::now();
    if (restore) {
      s.duel->Restore(*s.snapshot);
      restore_seconds = Seconds(t0);
    }
    t0 = Clock::now();
    s.duel->ReshuffleFuture(ContinuationSeed(root.id, s.continuation));
    const double reshuffle = Seconds(t0);
    t0 = Clock::now();
    s.steps = 0;
    s.own = 0;
    s.fresh = true;
    const int action = root.root_actions[s.continuation];
    if (action >= 0) {
      try {
        s.duel->Step(action);
        if (s.duel->finished()) s.ended = Finish(s, false);
      } catch (const std::runtime_error &error) {
        if (std::string(error.what()).rfind("menu row ", 0) == 0) throw;  // not a menu index of the root: a caller bug
        s.ended = Finish(s, false, error.what());
      }
    }
    const double root_action = Seconds(t0);
    std::lock_guard<std::mutex> lock(mutex_);
    timings_.restore += restore_seconds;
    timings_.reshuffle += reshuffle;
    timings_.root_action += root_action;
    timings_.continuations += 1;
  }

  std::unique_ptr<Result> Finish(const Slot &s, bool cut, const std::string &error = "") {
    const Root &root = roots_.at(s.root).root;
    const bool outcome = !cut && error.empty();
    if (!error.empty()) {
      std::lock_guard<std::mutex> lock(mutex_);
      ++timings_.errors;
    }
    return std::make_unique<Result>(Result{s.root, s.continuation, ContinuationSeed(s.root, s.continuation),
                                           root.root_actions[s.continuation], root.to_play, root.depth,
                                           outcome ? s.duel->winner() : -1, outcome ? s.duel->win_reason() : -1,
                                           s.steps, s.own, cut, error});
  }

  // Record ended continuations and start each slot's next one (in parallel), until no live slot is ended.
  void Settle() {
    while (true) {
      std::vector<size_t> next;
      for (size_t i = 0; i < slots_.size(); ++i) {
        Slot &s = slots_[i];
        if (!s.duel || !s.ended) continue;
        results_.push_back(*s.ended);
        s.ended.reset();
        RootState &r = roots_.at(s.root);
        if (r.pending.empty()) {
          Release(s, false);
          continue;
        }
        s.continuation = r.pending.front();
        r.pending.pop_front();
        next.push_back(i);
      }
      if (next.empty()) return;
      Parallel(next, [this](size_t i) { Begin(slots_[i], true); });
    }
  }

  // The slot gives back its clone; a root with nothing left to start or run is done. A refused clone drops its
  // root's remaining continuations (each would be refused the same way).
  void Release(Slot &s, bool refused) {
    RootState &r = roots_.at(s.root);
    if (refused) r.pending.clear();
    --r.active;
    s.snapshot.reset();
    s.duel.reset();
    s.refusal.clear();
    s.ended.reset();
    if (r.pending.empty() && r.active == 0) {
      const int64_t id = s.root;
      roots_.erase(id);
      for (auto it = order_.begin(); it != order_.end(); ++it)
        if (*it == id) {
          order_.erase(it);
          break;
        }
    }
    s.root = -1;
  }

  // Runs ``work(i)`` for every index on the pool's threads; the first exception is rethrown after all finish.
  template <class F>
  void Parallel(const std::vector<size_t> &indices, F work) {
    std::vector<std::future<void>> futures;
    futures.reserve(indices.size());
    for (size_t i : indices) futures.push_back(pool_.submit_task([&work, i] { work(i); }));
    std::exception_ptr error;
    for (auto &f : futures) {
      try {
        f.get();
      } catch (...) {
        if (!error) error = std::current_exception();
      }
    }
    if (error) std::rethrow_exception(error);
  }

  DuelEnvSpec spec_;
  std::vector<Slot> slots_;
  std::map<int64_t, RootState> roots_;
  std::deque<int64_t> order_;
  std::vector<Result> results_;
  std::vector<RootLog> logs_;
  uint64_t seed_ = 0;
  int64_t next_id_ = 0;
  Timings timings_;
  std::mutex mutex_;
  BS::thread_pool pool_;
};

}  // namespace duelenv
