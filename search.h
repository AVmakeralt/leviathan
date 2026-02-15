#ifndef SEARCH_H
#define SEARCH_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include "board.h"
#include "engine_components.h"
#include "movegen.h"
#include "tt.h"

namespace search {

struct Limits {
  int depth = 3;
  int movetimeMs = 0;
  bool infinite = false;
};

struct Result {
  movegen::Move bestMove;
  movegen::Move ponder;
  int depth = 0;
  long long nodes = 0;
  int scoreCp = 0;
  std::vector<movegen::Move> pv;
  std::vector<int> candidateDepths;
  std::string evalBreakdown;
};

class Searcher {
 public:
  Searcher(engine_components::search_arch::Features features,
           engine_components::search_helpers::KillerTable* killer,
           engine_components::search_helpers::HistoryHeuristic* history,
           engine_components::search_helpers::CounterMoveTable* counter,
           engine_components::search_helpers::PVTable* pvTable,
           engine_components::search_helpers::SEE* see,
           engine_components::search_helpers::TacticalSolver* tacticalSolver,
           engine_components::search_helpers::DynamicPVWeights* dynamicPv,
           engine_components::eval_model::Handcrafted* handcrafted,
           engine_components::eval_model::EndgameHeuristics* endgame,
           tt::Table* table,
           bool* stopFlag)
      : features_(features),
        killer_(killer),
        history_(history),
        counter_(counter),
        pvTable_(pvTable),
        see_(see),
        tacticalSolver_(tacticalSolver),
        dynamicPv_(dynamicPv),
        handcrafted_(handcrafted),
        endgame_(endgame),
        table_(table),
        stopFlag_(stopFlag) {}

  void setOpeningKey(const std::string& key) { openingKey_ = key; }

  Result think(const board::Board& board, const Limits& limits) {
    rootBoard_ = board;
    limits_ = limits;
    start_ = std::chrono::steady_clock::now();
    nodes_ = 0;
    if (table_) table_->newSearch();

    Result out;
    auto rootMoves = movegen::generatePseudoLegal(rootBoard_);
    if (rootMoves.empty()) return out;

    const auto rootCtx = analyzePhase(rootBoard_);
    out.depth = std::max(1, limits.depth);
    const int multiPv = (features_.useMultiPV && features_.multiPV > 1)
                            ? std::min<int>(features_.multiPV, static_cast<int>(rootMoves.size()))
                            : 1;

    assignCandidateDepths(out, multiPv, out.depth, rootCtx);
    int prevEval = 0;

    for (int depth = 1; depth <= out.depth; ++depth) {
      if (shouldStop()) break;

      int delta = adaptiveAspirationWindow(rootCtx, depth, prevEval);
      int alpha = (features_.useAspiration && depth > 1) ? prevEval - delta : -kMate;
      int beta = (features_.useAspiration && depth > 1) ? prevEval + delta : kMate;

      auto ordered = rootMoves;
      orderMoves(ordered, 0, rootCtx, movegen::Move{});

      std::vector<std::pair<int, movegen::Move>> scored;
      scored.reserve(ordered.size());

      for (std::size_t i = 0; i < ordered.size(); ++i) {
        if (shouldStop()) break;

        board::Board child = rootBoard_;
        if (!child.applyMove(ordered[i].from, ordered[i].to, ordered[i].promotion)) continue;

        auto childCtx = analyzePhase(child);
        int d = depth;
        if (i < out.candidateDepths.size()) d = out.candidateDepths[i];
        d += extensionForMove(rootBoard_, ordered[i], childCtx, 0);

        int score = -pvs(child, d - 1, -beta, -alpha, 1, childCtx);
        if (features_.usePVS && score > alpha && score < beta) {
          score = -pvs(child, d - 1, -beta, -alpha, 1, childCtx);
        }

        scored.push_back({score, ordered[i]});
        alpha = std::max(alpha, score);
      }

      std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      if (!scored.empty()) {
        out.scoreCp = scored.front().first;
        prevEval = out.scoreCp;
        out.bestMove = scored.front().second;
        out.pv.clear();
        for (int i = 0; i < multiPv && i < static_cast<int>(scored.size()); ++i) out.pv.push_back(scored[i].second);
        if (dynamicPv_) dynamicPv_->reward(openingKey_ + ":" + out.bestMove.toUCI(), 2 + depth);
        updateHeuristics(depth, out.bestMove, true);
      }
    }

    out.nodes = nodes_;
    out.ponder = out.pv.size() > 1 ? out.pv[1] : out.bestMove;
    out.evalBreakdown = handcrafted_ ? handcrafted_->breakdown() : "";
    return out;
  }

 private:
  static constexpr int kMate = 30000;

  engine_components::search_arch::Features features_;
  engine_components::search_helpers::KillerTable* killer_;
  engine_components::search_helpers::HistoryHeuristic* history_;
  engine_components::search_helpers::CounterMoveTable* counter_;
  engine_components::search_helpers::PVTable* pvTable_;
  engine_components::search_helpers::SEE* see_;
  engine_components::search_helpers::TacticalSolver* tacticalSolver_;
  engine_components::search_helpers::DynamicPVWeights* dynamicPv_;
  engine_components::eval_model::Handcrafted* handcrafted_;
  engine_components::eval_model::EndgameHeuristics* endgame_;
  tt::Table* table_;
  bool* stopFlag_;

  board::Board rootBoard_;
  Limits limits_;
  long long nodes_ = 0;
  std::string openingKey_;
  std::chrono::steady_clock::time_point start_{};

  bool shouldStop() const {
    if (stopFlag_ && *stopFlag_) return true;
    if (limits_.infinite || limits_.movetimeMs <= 0) return false;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_);
    return elapsed.count() >= limits_.movetimeMs;
  }

  int adaptiveAspirationWindow(const engine_components::search_arch::PhaseContext& ctx, int depth, int prevEval) const {
    int delta = 30 + depth * 3;
    delta += ctx.endgame ? 12 : 0;
    delta += std::min(40, std::abs(ctx.materialImbalance) / 40);
    delta += std::min(24, std::abs(prevEval) / 25);
    return delta;
  }

  static std::uint64_t hashBoard(const board::Board& b) {
    std::uint64_t h = 1469598103934665603ULL;
    for (char c : b.squares) {
      h ^= static_cast<unsigned char>(c);
      h *= 1099511628211ULL;
    }
    h ^= b.whiteToMove ? 1ULL : 0ULL;
    return h * 1099511628211ULL;
  }

  static int pieceValue(char p) {
    switch (static_cast<char>(std::tolower(static_cast<unsigned char>(p)))) {
      case 'p': return 100;
      case 'n': return 320;
      case 'b': return 330;
      case 'r': return 500;
      case 'q': return 900;
      default: return 0;
    }
  }

  engine_components::search_arch::PhaseContext analyzePhase(const board::Board& b) const {
    engine_components::search_arch::PhaseContext c;
    int pieces = 0;
    for (char p : b.squares) {
      if (p == '.') continue;
      ++pieces;
      int pv = pieceValue(p);
      c.materialImbalance += std::isupper(static_cast<unsigned char>(p)) ? pv : -pv;
      if (std::tolower(static_cast<unsigned char>(p)) == 'q') c.threatDensity += 2;
      if (std::tolower(static_cast<unsigned char>(p)) == 'r') c.threatDensity += 1;
      if (std::tolower(static_cast<unsigned char>(p)) == 'p') c.pawnTropism += 1;
    }
    c.endgame = pieces <= 10;
    c.kingSafety = 10 - std::min(10, c.threatDensity);
    return c;
  }

  int evaluate(const board::Board& b, const engine_components::search_arch::PhaseContext& ctx) const {
    int score = 0;
    int pieces = 0;
    for (char p : b.squares) {
      if (p == '.') continue;
      ++pieces;
      score += std::isupper(static_cast<unsigned char>(p)) ? pieceValue(p) : -pieceValue(p);
    }
    if (handcrafted_) score += handcrafted_->score() / 20;
    if (endgame_ && endgame_->enabled) score += endgame_->evaluate(pieces <= 8);
    score += ctx.threatDensity * 2 + ctx.kingSafety - ctx.pawnTropism;
    return b.whiteToMove ? score : -score;
  }

  int tacticalScan(const board::Board& b, const movegen::Move& m) const {
    int score = 0;
    if (b.pieceAt(m.to) != '.') score += 350;
    if (m.promotion != '\0') score += 300;
    int fromRank = m.from / 8;
    int toRank = m.to / 8;
    if (std::abs(toRank - fromRank) >= 2) score += 40;
    if (m.to % 8 >= 3 && m.to % 8 <= 4) score += 20;
    return score;
  }

  int extensionForMove(const board::Board& b, const movegen::Move& m, const engine_components::search_arch::PhaseContext& ctx,
                       int ply) const {
    if (!features_.useThreatExtensions || ply > 20) return 0;
    int ext = 0;
    if (b.pieceAt(m.to) != '.') ext += 1;                    // recapture/capture pressure
    if (m.promotion != '\0') ext += 1;                       // promotion race
    if ((m.to / 8 == 6 || m.to / 8 == 1) &&                  // advanced passer-like push
        std::tolower(static_cast<unsigned char>(b.pieceAt(m.from))) == 'p')
      ext += 1;
    int threatScore = ctx.threatDensity + std::abs(ctx.materialImbalance) / 200 + ctx.pawnTropism;
    if (threatScore >= 8) ext += 1;
    return std::min(2, ext);
  }

  int quiescence(board::Board& b, int alpha, int beta, int qDepth, const engine_components::search_arch::PhaseContext& ctx) {
    ++nodes_;
    int standPat = evaluate(b, ctx);
    if (standPat >= beta) return beta;
    alpha = std::max(alpha, standPat);
    if (qDepth <= 0) return alpha;

    auto moves = movegen::generatePseudoLegal(b);
    std::vector<movegen::Move> forcing;
    forcing.reserve(moves.size());
    for (const auto& m : moves) {
      bool capture = b.pieceAt(m.to) != '.';
      bool promotion = m.promotion != '\0';
      bool forcingMove = tacticalScan(b, m) >= 120;
      int seeScore = see_ ? see_->estimate(m) : 0;
      if (!capture && !promotion && !forcingMove) continue;
      if (capture && seeScore < -10) continue;
      forcing.push_back(m);
    }

    orderMoves(forcing, 0, ctx, movegen::Move{});
    for (const auto& m : forcing) {
      if (shouldStop()) break;
      board::Board child = b;
      if (!child.applyMove(m.from, m.to, m.promotion)) continue;
      auto childCtx = analyzePhase(child);
      int nextQ = ctx.threatDensity >= 6 ? qDepth - 1 : qDepth - 2;
      int score = -quiescence(child, -beta, -alpha, std::max(0, nextQ), childCtx);
      if (score >= beta) return beta;
      alpha = std::max(alpha, score);
    }
    return alpha;
  }

  int pvs(board::Board& b, int depth, int alpha, int beta, int ply, const engine_components::search_arch::PhaseContext& ctx) {
    ++nodes_;
    if (shouldStop()) return evaluate(b, ctx);

    const std::uint64_t key = hashBoard(b);
    if (table_) {
      const tt::Probe probe = table_->probe(key);
      if (probe.hit && probe.entry.depth >= depth) {
        if (probe.entry.bound == tt::Bound::EXACT) return probe.entry.score;
        if (probe.entry.bound == tt::Bound::LOWER && probe.entry.score >= beta) return probe.entry.score;
        if (probe.entry.bound == tt::Bound::UPPER && probe.entry.score <= alpha) return probe.entry.score;
      }
    }

    if (depth <= 0) {
      int qDepth = ctx.threatDensity >= 6 ? 4 : 2;
      return features_.useQuiescence ? quiescence(b, alpha, beta, qDepth, ctx) : evaluate(b, ctx);
    }

    if (features_.usePhaseAwareFutility && features_.useFutility && depth == 1 && ctx.endgame && ctx.threatDensity < 3) {
      const int staticEval = evaluate(b, ctx);
      if (staticEval + 90 <= alpha) return staticEval;
    }

    if (features_.useMateDistancePruning) {
      alpha = std::max(alpha, -kMate + ply);
      beta = std::min(beta, kMate - ply);
      if (alpha >= beta) return alpha;
    }

    if (features_.useProbabilisticPruning && depth <= 2 && ctx.threatDensity < 3) {
      const int staticEval = evaluate(b, ctx);
      if (staticEval + 120 < alpha) return staticEval;
    }

    if (features_.useNullMove && depth >= 3 && !ctx.endgame && ctx.threatDensity < 6) {
      board::Board nullBoard = b;
      nullBoard.whiteToMove = !nullBoard.whiteToMove;
      auto nullCtx = analyzePhase(nullBoard);
      int R = 2 + depth / 4;
      int nullScore = -pvs(nullBoard, depth - 1 - R, -beta, -beta + 1, ply + 1, nullCtx);
      if (nullScore >= beta) return nullScore;
    }

    auto moves = movegen::generatePseudoLegal(b);
    if (moves.empty()) return evaluate(b, ctx) - ply;

    movegen::Move ttBest = table_ ? table_->probe(key).entry.bestMove : movegen::Move{};
    if (features_.useIID && ttBest.from < 0 && depth >= 4) {
      board::Board iidBoard = b;
      auto iidCtx = analyzePhase(iidBoard);
      (void)pvs(iidBoard, depth - 2, alpha, beta, ply + 1, iidCtx);
      if (table_) ttBest = table_->probe(key).entry.bestMove;
    }
    orderMoves(moves, ply, ctx, ttBest);

    const int originalAlpha = alpha;
    movegen::Move best{};
    bool found = false;

    for (std::size_t i = 0; i < moves.size(); ++i) {
      bool quietLate = (b.pieceAt(moves[i].to) == '.' && moves[i].promotion == '\0' && i > 8);
      if (features_.useLMP && depth <= 3 && ctx.threatDensity < 4 && quietLate) continue;
      board::Board child = b;
      if (!child.applyMove(moves[i].from, moves[i].to, moves[i].promotion)) continue;

      const auto childCtx = analyzePhase(child);
      int nextDepth = depth - 1;
      if (features_.useAdaptiveLmr && features_.useLMR && i >= 4 && depth >= 3) {
        int reduction = 1 + (childCtx.threatDensity <= 3 ? 1 : 0) - (childCtx.kingSafety <= 3 ? 1 : 0);
        reduction += std::abs(childCtx.materialImbalance) > 500 ? 1 : 0;
        nextDepth = std::max(1, nextDepth - std::max(0, reduction));
      }
      nextDepth += extensionForMove(b, moves[i], childCtx, ply);

      int score;
      if (features_.usePVS && found) {
        score = -pvs(child, nextDepth, -alpha - 1, -alpha, ply + 1, childCtx);
        if (score > alpha && score < beta) score = -pvs(child, nextDepth, -beta, -alpha, ply + 1, childCtx);
      } else {
        score = -pvs(child, nextDepth, -beta, -alpha, ply + 1, childCtx);
      }

      if (score > alpha) {
        alpha = score;
        best = moves[i];
        found = true;
      } else {
        updateHeuristics(ply, moves[i], false);
      }

      if (alpha >= beta) {
        if (killer_ && ply < static_cast<int>(killer_->killer.size())) {
          killer_->killer[ply][1] = killer_->killer[ply][0];
          killer_->killer[ply][0] = moves[i];
        }
        updateHeuristics(ply, moves[i], true);
        break;
      }
    }

    if (table_) {
      tt::Bound bound = tt::Bound::EXACT;
      if (alpha <= originalAlpha) bound = tt::Bound::UPPER;
      else if (alpha >= beta) bound = tt::Bound::LOWER;

      tt::NodeType type = tt::NodeType::QUIET;
      if (ply <= 1) type = tt::NodeType::PV;
      else if (ctx.threatDensity >= 5) type = tt::NodeType::TACTICAL;

      const std::uint16_t priority = static_cast<std::uint16_t>(std::min(65535, depth * 32 + ctx.threatDensity * 20));
      table_->store(key, depth, alpha, bound, best, type, priority);
    }

    return alpha;
  }

  void orderMoves(std::vector<movegen::Move>& moves, int ply, const engine_components::search_arch::PhaseContext& ctx,
                  const movegen::Move& ttMove) {
    std::vector<std::pair<int, movegen::Move>> scored;
    scored.reserve(moves.size());

    for (const auto& m : moves) {
      int score = 0;

      if (ttMove.from == m.from && ttMove.to == m.to) score += 1000000;                   // PV/TT first
      if (pvTable_ && ply < static_cast<int>(pvTable_->length.size()) &&
          pvTable_->length[ply] > 0 && pvTable_->pv[ply][0].from == m.from &&
          pvTable_->pv[ply][0].to == m.to)
        score += 980000;

      const bool capture = rootBoard_.pieceAt(m.to) != '.';
      if (capture) score += 900000 + (see_ ? see_->estimate(m) : 0);                      // winning captures
      if (m.promotion != '\0') score += 850000;                                           // promotions

      const int tactical = tacticalScan(rootBoard_, m);
      if (features_.useTacticalPrefilter && tactical >= 120) score += 800000;              // checks/forcing proxy

      if (killer_ && ply < static_cast<int>(killer_->killer.size())) {
        if (m.from == killer_->killer[ply][0].from && m.to == killer_->killer[ply][0].to) score += 750000;
        if (m.from == killer_->killer[ply][1].from && m.to == killer_->killer[ply][1].to) score += 740000;
      }

      if (counter_ && m.from >= 0 && m.to >= 0) {
        const auto& c = counter_->counter[m.from][m.to];
        if (c.from == m.from && c.to == m.to) score += 700000;
      }

      if (history_) score += history_->adaptiveScore(m.from, m.to);
      score += (4 - std::min(4, std::abs((m.to % 8) - 3))) * 8;                            // center pressure
      score += ctx.threatDensity * 3;
      if (dynamicPv_) score += dynamicPv_->get(openingKey_ + ":" + m.toUCI());

      scored.push_back({score, m});
    }

    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (std::size_t i = 0; i < moves.size(); ++i) moves[i] = scored[i].second;
  }

  void assignCandidateDepths(Result& out, int n, int rootDepth, const engine_components::search_arch::PhaseContext& ctx) const {
    out.candidateDepths.assign(static_cast<std::size_t>(n), rootDepth);
    if (!features_.useMultiRateThinking || n <= 1) return;

    for (int i = 1; i < n; ++i) {
      int reduction = ctx.endgame ? 1 : 0;
      reduction += ctx.threatDensity <= 3 ? 1 : 0;
      if (features_.useSelectiveDeepening && ctx.threatDensity >= 6 && i == 1) reduction = 0;  // hotspot bias
      out.candidateDepths[static_cast<std::size_t>(i)] = std::max(1, rootDepth - reduction);
    }
  }

  void updateHeuristics(int ply, const movegen::Move& m, bool success) {
    if (history_ && m.from >= 0 && m.to >= 0) {
      if (success) history_->recordSuccess(m.from, m.to, 16 + ply);
      else history_->recordFail(m.from, m.to);
    }
    if (counter_ && success && m.from >= 0 && m.to >= 0) counter_->counter[m.from][m.to] = m;
    if (pvTable_ && success && ply < static_cast<int>(pvTable_->length.size())) {
      pvTable_->pv[ply][0] = m;
      pvTable_->length[ply] = 1;
    }
  }
};

}  // namespace search

#endif
