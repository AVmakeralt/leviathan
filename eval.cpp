#include "eval.h"

#include <array>
#include <cctype>
#include <sstream>

namespace eval {

void initialize(Params& params) {
  if (params.piece[0] <= 0) params.piece[0] = 100;
  if (params.piece[1] <= 0) params.piece[1] = 320;
  if (params.piece[2] <= 0) params.piece[2] = 330;
  if (params.piece[3] <= 0) params.piece[3] = 500;
  if (params.piece[4] <= 0) params.piece[4] = 900;
}

static int knightPst(int sq) {
  static const std::array<int, 64> knight = {
      -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0,   5,   5,   0,   -20, -40,
      -30, 5,   10,  15,  15,  10,  5,   -30, -30, 0,   15,  20,  20,  15,  0,   -30,
      -30, 5,   15,  20,  20,  15,  5,   -30, -30, 0,   10,  15,  15,  10,  0,   -30,
      -40, -20, 0,   0,   0,   0,   -20, -40, -50, -40, -30, -30, -30, -30, -40, -50};
  return knight[sq];
}

static int pst(char piece, int sq) {
  char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
  if (p == 'n') return knightPst(sq);
  return 0;
}

int evaluate(const board::Board& b, const Params& params) {
  int score = 0;
  int whiteBishops = 0;
  int blackBishops = 0;

  for (int sq = 0; sq < 64; ++sq) {
    const char c = b.squares[sq];
    if (c == '.') continue;

    const bool white = std::isupper(static_cast<unsigned char>(c));
    const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const int idx = p == 'p' ? 0 : p == 'n' ? 1 : p == 'b' ? 2 : p == 'r' ? 3 : p == 'q' ? 4 : 5;
    const int material = params.piece[idx];
    const int psq = pst(c, white ? sq : (56 ^ sq));
    const int term = material + psq;

    if (white) {
      score += term;
    } else {
      score -= term;
    }

    if (c == 'B') ++whiteBishops;
    if (c == 'b') ++blackBishops;
  }

  if (whiteBishops >= 2) score += params.bishopPairBonus;
  if (blackBishops >= 2) score -= params.bishopPairBonus;
  score += b.whiteToMove ? params.tempoBonus : -params.tempoBonus;

  return b.whiteToMove ? score : -score;
}

std::string breakdown(const board::Board& b, const Params& params) {
  std::ostringstream out;
  out << "eval=" << evaluate(b, params) << " stm=" << (b.whiteToMove ? 'w' : 'b')
      << " bp=" << params.bishopPairBonus << " tempo=" << params.tempoBonus;
  return out.str();
}

}  // namespace eval
