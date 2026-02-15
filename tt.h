#ifndef TT_H
#define TT_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "board.h"
#include "movegen.h"

namespace tt {

enum Bound : int { EXACT = 0, LOWER = 1, UPPER = 2 };

struct Entry {
  std::uint64_t key = 0;
  int depth = -1;
  int score = 0;
  Bound bound = EXACT;
  movegen::Move bestMove{};
};

class Table {
 public:
  void initialize(std::size_t mb);
  void clear();
  bool probe(std::uint64_t key, Entry& out) const;
  void store(const Entry& e);

 private:
  std::vector<Entry> entries_;
};

void initializeZobrist();
std::uint64_t hash(const board::Board& b);

}  // namespace tt

#endif
