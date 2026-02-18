#ifndef ENGINE_COMPONENTS_H
#define ENGINE_COMPONENTS_H

#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "movegen.h"

namespace engine_components {

namespace representation {
using Bitboard64 = std::uint64_t;
struct Bitboard128 {
  std::uint64_t lo = 0;
  std::uint64_t hi = 0;
};

struct AttackTables {
  std::array<Bitboard64, 64> rookAttacks{};
  std::array<Bitboard64, 64> bishopAttacks{};
  void initialize() {}
};

struct MagicTables {
  bool enabled = true;
  void initialize() {}
};
}  // namespace representation

namespace hashing {
struct Zobrist {
  std::array<std::array<std::uint64_t, 64>, 12> pieceSquare{};
  std::uint64_t sideToMove = 0;
  void initialize() {}
};

struct RepetitionTracker {
  std::vector<std::uint64_t> keys;
  int fiftyMoveClock = 0;

  void push(std::uint64_t key) { keys.push_back(key); }
  void clear() {
    keys.clear();
    fiftyMoveClock = 0;
  }
  bool isThreefold(std::uint64_t key) const {
    int count = 0;
    for (std::uint64_t k : keys) {
      if (k == key) ++count;
    }
    return count >= 3;
  }
};
}  // namespace hashing

namespace search_helpers {
struct KillerTable {
  std::array<std::array<movegen::Move, 2>, 128> killer{};
};

struct HistoryHeuristic {
  std::array<std::array<int, 64>, 64> score{};
};

struct CounterMoveTable {
  std::array<std::array<movegen::Move, 64>, 64> counter{};
};

struct PVTable {
  std::array<std::array<movegen::Move, 128>, 128> pv{};
  std::array<int, 128> length{};
};

struct SEE {
  int estimate(const movegen::Move&) const { return 0; }
};

struct SearchResultCache {
  std::unordered_map<std::string, std::string> openingBestMoves;

  std::string get(const std::string& key) const {
    auto it = openingBestMoves.find(key);
    return it == openingBestMoves.end() ? "" : it->second;
  }

  void put(const std::string& key, const std::string& move) { openingBestMoves[key] = move; }

  bool save(const std::string& path) const {
    std::ofstream out(path);
    if (!out) return false;
    for (const auto& [k, v] : openingBestMoves) out << k << ' ' << v << '\n';
    return true;
  }

  bool load(const std::string& path) {
    openingBestMoves.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string k;
    std::string v;
    while (in >> k >> v) openingBestMoves[k] = v;
    return true;
  }
};
}  // namespace search_helpers

namespace eval_model {
struct Handcrafted {
  int material = 0;
  int psqt = 0;
  int pawnStructure = 0;
  int kingSafety = 0;
  int mobility = 0;
  int space = 0;
  int bishopPair = 0;
  int rookActivity = 0;
  int tropism = 0;
  int tempo = 0;
  int initiative = 0;
  int timeAwareness = 0;

  int score() const {
    return material + psqt + pawnStructure + kingSafety + mobility + space + bishopPair + rookActivity + tropism +
           tempo + initiative + timeAwareness;
  }

  std::string breakdown() const {
    std::ostringstream oss;
    oss << "material=" << material << " psqt=" << psqt << " pawn=" << pawnStructure << " king=" << kingSafety
        << " mobility=" << mobility << " space=" << space << " bishopPair=" << bishopPair
        << " rookActivity=" << rookActivity << " tropism=" << tropism << " tempo=" << tempo
        << " initiative=" << initiative << " timeAwareness=" << timeAwareness;
    return oss.str();
  }
};

struct EndgameHeuristics {
  bool enabled = true;
  int kingPawnPattern = 0;
  int specializedScore = 0;
  int evaluate(bool deepEndgame) const { return deepEndgame ? kingPawnPattern + specializedScore : 0; }
};

struct NNUEConfig {
  int inputs = 1024;     // 768-1024 target range
  int hidden1 = 512;     // width target 256-512
  int hidden2 = 512;     // width target 256-512
};

struct NNUE {
  bool enabled = true;
  std::string weightsPath = "nnue.bin";
  NNUEConfig cfg{};
  std::vector<float> w1;
  std::vector<float> b1;
  std::vector<float> w2;
  std::vector<float> b2;
  std::vector<float> w3;
  float b3 = 0.0f;

  std::size_t parameterCount() const {
    return static_cast<std::size_t>(cfg.inputs) * cfg.hidden1 + static_cast<std::size_t>(cfg.hidden1) +
           static_cast<std::size_t>(cfg.hidden1) * cfg.hidden2 + static_cast<std::size_t>(cfg.hidden2) +
           static_cast<std::size_t>(cfg.hidden2) + 1;
  }

  void initializeWeights() {
    w1.assign(static_cast<std::size_t>(cfg.inputs * cfg.hidden1), 0.0f);
    b1.assign(static_cast<std::size_t>(cfg.hidden1), 0.0f);
    w2.assign(static_cast<std::size_t>(cfg.hidden1 * cfg.hidden2), 0.0f);
    b2.assign(static_cast<std::size_t>(cfg.hidden2), 0.0f);
    w3.assign(static_cast<std::size_t>(cfg.hidden2), 0.0f);

    for (std::size_t i = 0; i < w1.size(); ++i) w1[i] = static_cast<float>((static_cast<int>(i % 31) - 15) * 0.002f);
    for (std::size_t i = 0; i < w2.size(); ++i) w2[i] = static_cast<float>((static_cast<int>(i % 19) - 9) * 0.003f);
    for (std::size_t i = 0; i < w3.size(); ++i) w3[i] = static_cast<float>((static_cast<int>(i % 11) - 5) * 0.01f);
  }

  bool load(const std::string& path) {
    weightsPath = path;
    initializeWeights();

    std::ifstream in(path, std::ios::binary);
    if (in) {
      in.read(reinterpret_cast<char*>(w1.data()), static_cast<std::streamsize>(w1.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(b1.data()), static_cast<std::streamsize>(b1.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(w2.data()), static_cast<std::streamsize>(w2.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(b2.data()), static_cast<std::streamsize>(b2.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(w3.data()), static_cast<std::streamsize>(w3.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(&b3), static_cast<std::streamsize>(sizeof(b3)));
    }

    enabled = true;
    return true;
  }

  static std::vector<float> extractFeatures(const std::array<char, 64>& squares, bool whiteToMove, int inputSize) {
    std::vector<float> f(static_cast<std::size_t>(inputSize), 0.0f);
    auto piecePlane = [](char p) {
      switch (p) {
        case 'P': return 0; case 'N': return 1; case 'B': return 2; case 'R': return 3; case 'Q': return 4; case 'K': return 5;
        case 'p': return 6; case 'n': return 7; case 'b': return 8; case 'r': return 9; case 'q': return 10; case 'k': return 11;
        default: return -1;
      }
    };
    for (int sq = 0; sq < 64; ++sq) {
      int plane = piecePlane(squares[static_cast<std::size_t>(sq)]);
      if (plane < 0) continue;
      int idx = plane * 64 + sq;
      if (idx >= 0 && idx < inputSize) f[static_cast<std::size_t>(idx)] = 1.0f;
    }
    if (inputSize > 768) f[768] = whiteToMove ? 1.0f : -1.0f;
    return f;
  }

  int evaluate(const std::vector<float>& input) const {
    if (!enabled || input.empty() || w1.empty()) return 0;

    std::vector<float> h1(static_cast<std::size_t>(cfg.hidden1), 0.0f);
    std::vector<float> h2(static_cast<std::size_t>(cfg.hidden2), 0.0f);

    for (int h = 0; h < cfg.hidden1; ++h) {
      float acc = b1[static_cast<std::size_t>(h)];
      for (int i = 0; i < cfg.inputs; ++i) acc += input[static_cast<std::size_t>(i)] * w1[static_cast<std::size_t>(h * cfg.inputs + i)];
      h1[static_cast<std::size_t>(h)] = std::max(0.0f, acc);
    }
    for (int h = 0; h < cfg.hidden2; ++h) {
      float acc = b2[static_cast<std::size_t>(h)];
      for (int i = 0; i < cfg.hidden1; ++i) acc += h1[static_cast<std::size_t>(i)] * w2[static_cast<std::size_t>(h * cfg.hidden1 + i)];
      h2[static_cast<std::size_t>(h)] = std::max(0.0f, acc);
    }

    float out = b3;
    for (int i = 0; i < cfg.hidden2; ++i) out += h2[static_cast<std::size_t>(i)] * w3[static_cast<std::size_t>(i)];
    return static_cast<int>(std::lround(out * 100.0f));
  }
};

struct StrategyConfig {
  int planes = 24;         // 8x8x(16-32) target range
  int channels = 256;      // 128-256 target range
  int residualBlocks = 16; // 10-20 target range
  int policyOutputs = 64;  // simplified policy over destination squares
};

struct StrategyOutput {
  int valueCp = 0;
  std::vector<float> policy;
};

struct StrategyNet {
  bool enabled = true;
  std::string weightsPath = "strategy_large.nn";
  StrategyConfig cfg{};
  std::vector<float> stem;
  std::vector<float> blocks;
  std::vector<float> valueHead;
  float valueBias = 0.0f;
  std::vector<float> policyHead;

  std::size_t parameterCount() const {
    const std::size_t stemParams = static_cast<std::size_t>(cfg.planes) * cfg.channels * 9;
    const std::size_t blockParams = static_cast<std::size_t>(cfg.residualBlocks) * 2ULL * cfg.channels * cfg.channels;
    const std::size_t headParams = static_cast<std::size_t>(cfg.channels) * (1 + cfg.policyOutputs) + cfg.policyOutputs;
    return stemParams + blockParams + headParams;
  }

  void initializeWeights() {
    stem.assign(static_cast<std::size_t>(cfg.planes * cfg.channels), 0.001f);
    blocks.assign(static_cast<std::size_t>(cfg.residualBlocks * 2 * cfg.channels * cfg.channels), 0.0f);
    valueHead.assign(static_cast<std::size_t>(cfg.channels), 0.0f);
    policyHead.assign(static_cast<std::size_t>(cfg.channels * cfg.policyOutputs), 0.0f);
    for (std::size_t i = 0; i < blocks.size(); ++i) blocks[i] = static_cast<float>((static_cast<int>(i % 23) - 11) * 0.0005f);
    for (std::size_t i = 0; i < valueHead.size(); ++i) valueHead[i] = static_cast<float>((static_cast<int>(i % 13) - 6) * 0.01f);
    for (std::size_t i = 0; i < policyHead.size(); ++i) policyHead[i] = static_cast<float>((static_cast<int>(i % 17) - 8) * 0.0025f);
  }

  bool load(const std::string& path) {
    weightsPath = path;
    initializeWeights();

    std::ifstream in(path, std::ios::binary);
    if (in) {
      in.read(reinterpret_cast<char*>(stem.data()), static_cast<std::streamsize>(stem.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(blocks.data()), static_cast<std::streamsize>(blocks.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(valueHead.data()), static_cast<std::streamsize>(valueHead.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(&valueBias), static_cast<std::streamsize>(sizeof(valueBias)));
      in.read(reinterpret_cast<char*>(policyHead.data()), static_cast<std::streamsize>(policyHead.size() * sizeof(float)));
    }

    enabled = true;
    return true;
  }

  StrategyOutput evaluate(const std::vector<float>& planes) const {
    StrategyOutput out;
    out.policy.assign(static_cast<std::size_t>(cfg.policyOutputs), 0.0f);
    if (!enabled || planes.empty() || stem.empty()) return out;

    std::vector<float> state(static_cast<std::size_t>(cfg.channels), 0.0f);
    for (int c = 0; c < cfg.channels; ++c) {
      float acc = 0.0f;
      for (int p = 0; p < cfg.planes; ++p) {
        const std::size_t wIdx = static_cast<std::size_t>(p * cfg.channels + c);
        acc += planes[static_cast<std::size_t>(p)] * stem[wIdx];
      }
      state[static_cast<std::size_t>(c)] = std::max(0.0f, acc);
    }

    for (int b = 0; b < cfg.residualBlocks; ++b) {
      std::vector<float> next = state;
      const std::size_t offset = static_cast<std::size_t>(b * 2 * cfg.channels * cfg.channels);
      for (int c = 0; c < cfg.channels; ++c) {
        float mix = 0.0f;
        for (int i = 0; i < cfg.channels; ++i) {
          mix += state[static_cast<std::size_t>(i)] *
                 blocks[offset + static_cast<std::size_t>(c * cfg.channels + i)];
        }
        next[static_cast<std::size_t>(c)] = std::max(0.0f, state[static_cast<std::size_t>(c)] + mix);
      }
      state.swap(next);
    }

    float value = valueBias;
    for (int c = 0; c < cfg.channels; ++c) value += state[static_cast<std::size_t>(c)] * valueHead[static_cast<std::size_t>(c)];
    out.valueCp = static_cast<int>(std::lround(value * 100.0f));

    for (int m = 0; m < cfg.policyOutputs; ++m) {
      float logit = 0.0f;
      for (int c = 0; c < cfg.channels; ++c) {
        const std::size_t wIdx = static_cast<std::size_t>(c * cfg.policyOutputs + m);
        logit += state[static_cast<std::size_t>(c)] * policyHead[wIdx];
      }
      out.policy[static_cast<std::size_t>(m)] = logit;
    }

    return out;
  }
};

struct PolicyNet {
  bool enabled = false;
  std::vector<float> priors;
};

struct LossLearning {
  int lossCases = 0;
  int adversarialTests = 0;

  void recordLoss() { ++lossCases; }
  void runAdversarialSweep() { ++adversarialTests; }
};

struct TrainingInfra {
  bool selfPlayEnabled = false;
  bool supervisedEnabled = false;
  bool distillationEnabled = false;
  std::string replayBufferPath = "replay.bin";
  LossLearning lossLearning;
};
}  // namespace eval_model

namespace search_arch {
struct Features {
  bool usePVS = true;
  bool useAspiration = true;
  bool useQuiescence = true;
  bool useNullMove = true;
  bool useLMR = true;
  bool useFutility = true;
  bool useMateDistancePruning = true;
  bool useExtensions = true;
  bool useMultiPV = true;
  bool useMCTS = false;
  bool useParallel = false;
  bool useAsync = false;
  bool useMultiRateThinking = true;
  int multiPV = 1;
};

struct ParallelConfig {
  int threads = 1;
  bool rootParallel = false;
  bool treeSplit = false;
  bool hashSync = false;
  bool loadBalancing = false;
};

struct MCTSConfig {
  bool enabled = false;
  int simulations = 0;
};
}  // namespace search_arch

namespace opening {
struct Book {
  bool enabled = false;
  std::string format = "polyglot";
  std::string path = "book.bin";

  std::string probe() const { return ""; }
};

struct Novelty {
  std::unordered_map<std::string, int> seenPositions;
  bool isNovel(const std::string& key) {
    int& seen = seenPositions[key];
    ++seen;
    return seen == 1;
  }
};

struct BookBuilder {
  std::vector<std::pair<std::string, std::string>> lines;
  void addLine(const std::string& key, const std::string& move) { lines.push_back({key, move}); }
};

struct PrepModule {
  bool noveltySearch = false;
  bool pruneBook = false;
  Novelty novelty;
  BookBuilder builder;
};
}  // namespace opening

namespace timing {
struct Manager {
  int remainingMs = 0;
  int incrementMs = 0;
  std::string mode = "classical";
  bool pondering = false;
  int allocateMoveTimeMs(int fallback) const {
    if (remainingMs <= 0) return fallback;
    return remainingMs / 30 + incrementMs;
  }
};
}  // namespace timing

namespace tooling {
struct Formats {
  bool pgnEnabled = true;
  bool epdEnabled = true;
};

struct Integrity {
  bool antiCheatEnabled = false;
  bool checksumOk = true;
  bool verifyRuntime() const { return !antiCheatEnabled || checksumOk; }
};

struct TestHarness {
  bool regressionEnabled = true;
  bool eloEnabled = true;
  bool selfPlayTournaments = true;
  std::unordered_map<std::string, double> params;
};
}  // namespace tooling

}  // namespace engine_components

#endif
