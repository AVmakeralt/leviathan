#ifndef SEARCH_H
#define SEARCH_H

#include <cstdint>
#include <string>
#include <vector>

#include "board.h"
#include "eval.h"
#include "movegen.h"
#include "tt.h"

namespace search {

struct Limits {
  int depth = 5;
  int movetimeMs = 0;
  bool infinite = false;
};

struct Result {
  movegen::Move bestMove;
  movegen::Move ponder;
  int depth = 0;
  std::uint64_t nodes = 0;
  int scoreCp = 0;
  std::vector<movegen::Move> pv;
  std::string evalBreakdown;
  std::vector<int> candidateDepths;
};

class Searcher {
 public:
  Searcher(const eval::Params& params, tt::Table* table, bool* stop);
  Result think(board::Board& b, const Limits& l);

 private:
  int alphaBeta(board::Board& b, int depth, int alpha, int beta, int ply, bool allowNull);
  int quiescence(board::Board& b, int alpha, int beta, int ply);
  std::vector<movegen::Move> order(board::Board& b, const std::vector<movegen::Move>& moves, const movegen::Move& ttMove);

  const eval::Params& params_;
  tt::Table* tt_;
  bool* stop_;
  std::uint64_t nodes_ = 0;
  movegen::Move rootBest_{};
};

}  // namespace search

#endif
