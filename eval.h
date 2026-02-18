#ifndef EVAL_H
#define EVAL_H

#include <array>
#include <string>

#include "board.h"

namespace eval {

struct Params {
  std::array<int, 6> piece{100, 320, 330, 500, 900, 0};
  int bishopPairBonus = 30;
  int tempoBonus = 12;
};

void initialize(Params& params);
int evaluate(const board::Board& b, const Params& params);
std::string breakdown(const board::Board& b, const Params& params);

}  // namespace eval

#endif
