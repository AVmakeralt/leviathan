#ifndef TT_H
#define TT_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "movegen.h"

namespace tt {

enum class Bound : std::uint8_t { NONE = 0, EXACT = 1, LOWER = 2, UPPER = 3 };
enum class NodeType : std::uint8_t { PV = 0, TACTICAL = 1, QUIET = 2 };

struct Entry {
  std::uint64_t key = 0;
  int depth = -1;
  int score = 0;
  movegen::Move bestMove;
  Bound bound = Bound::NONE;
  std::uint8_t age = 0;
  std::uint16_t priority = 0;
};

struct Probe {
  bool hit = false;
  Entry entry{};
};

struct Table {
  std::vector<Entry> pvEntries;
  std::vector<Entry> tacticalEntries;
  std::vector<Entry> quietEntries;
  std::uint8_t generation = 0;

  void initialize(std::size_t mb) {
    std::size_t bytes = mb * 1024ULL * 1024ULL;
    std::size_t totalCount = bytes / sizeof(Entry);
    if (totalCount < 3) totalCount = 3;
    std::size_t pvCount = totalCount / 3;
    std::size_t tacticalCount = totalCount / 3;
    std::size_t quietCount = totalCount - pvCount - tacticalCount;
    pvEntries.assign(pvCount, Entry{});
    tacticalEntries.assign(tacticalCount, Entry{});
    quietEntries.assign(quietCount, Entry{});
    generation = 0;
  }

  void newSearch() { ++generation; }

  static std::size_t indexFor(std::uint64_t key, std::size_t size) {
    return size == 0 ? 0 : static_cast<std::size_t>(key % size);
  }

  Probe probe(std::uint64_t key) const {
    for (const auto* vec : {&pvEntries, &tacticalEntries, &quietEntries}) {
      if (vec->empty()) continue;
      const Entry& e = (*vec)[indexFor(key, vec->size())];
      if (e.key == key && e.bound != Bound::NONE) return {true, e};
    }
    return {};
  }

  void store(std::uint64_t key, int depth, int score, Bound bound, const movegen::Move& bestMove, NodeType type,
             std::uint16_t priority) {
    std::vector<Entry>* bucket = bucketFor(type);
    if (!bucket || bucket->empty()) return;

    Entry& e = (*bucket)[indexFor(key, bucket->size())];
    const bool replace = (e.bound == Bound::NONE) || (depth > e.depth) ||
                         (depth == e.depth && priority >= e.priority) || (e.age != generation);
    if (!replace) return;

    e.key = key;
    e.depth = depth;
    e.score = score;
    e.bound = bound;
    e.bestMove = bestMove;
    e.age = generation;
    e.priority = priority;
  }

  void clear() {
    pvEntries.assign(pvEntries.size(), Entry{});
    tacticalEntries.assign(tacticalEntries.size(), Entry{});
    quietEntries.assign(quietEntries.size(), Entry{});
  }

 private:
  std::vector<Entry>* bucketFor(NodeType type) {
    switch (type) {
      case NodeType::PV: return &pvEntries;
      case NodeType::TACTICAL: return &tacticalEntries;
      case NodeType::QUIET: return &quietEntries;
      default: return &quietEntries;
    }
  }
};

}  // namespace tt

#endif
