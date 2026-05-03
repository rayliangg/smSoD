// ecga.cpp
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <thread>

#include "ecga.h"
#include "random.hpp"
#include "region.h"
#include "utility.hpp"

using std::size_t;

extern RandomGenerator randomGenerator;

namespace {

// Parallelize index loops only when large enough to amortize thread overhead.
constexpr std::size_t kParallelMinItems = 64;

template <typename F>
void parallel_index_range(std::size_t begin, std::size_t end, F f) {
    if (begin >= end) {
        return;
    }
    const std::size_t n = end - begin;
    if (n < kParallelMinItems) {
        for (std::size_t i = begin; i < end; ++i) {
            f(i);
        }
        return;
    }
    unsigned tc = std::thread::hardware_concurrency();
    if (tc == 0) {
        tc = 4;
    }
    if (static_cast<std::size_t>(tc) > n) {
        tc = static_cast<unsigned>(n);
    }
    const std::size_t chunk = (n + static_cast<std::size_t>(tc) - 1) / static_cast<std::size_t>(tc);
    std::vector<std::thread> threads;
    threads.reserve(tc);
    for (unsigned t = 0; t < tc; ++t) {
        const std::size_t lo = begin + static_cast<std::size_t>(t) * chunk;
        if (lo >= end) {
            break;
        }
        const std::size_t hi = std::min(lo + chunk, end);
        threads.emplace_back([lo, hi, f]() {
            for (std::size_t i = lo; i < hi; ++i) {
                f(i);
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
}

}  // namespace

// static member
std::mutex Ecga::csv_write_mutex_;

Ecga::Ecga() noexcept
    : population(Parameter::pop_size),
      mating_pool(Parameter::pop_size)
{
    std::size_t max_block = 0;
    for (const auto& sub : Parameter::dim_subproblems) {
        max_block = std::max(max_block, sub.size());
    }
    decoded_gene_buffer_.reserve(std::max<std::size_t>(1, max_block));

    for (auto& ind : population) {
        ind.random();
    }
}

void Ecga::adjustMinBoundFromTopQuartile() noexcept {
    const size_t popSz = population.size();
    const size_t k     = popSz / 4;
    std::nth_element(population.begin(), population.begin() + k, population.end(), compareChromosome);

    double sum = 0.0;
    for (size_t i = 0; i < k; ++i) sum += population[i].fitness;
    const double mu = (k > 0 ? sum / k : 0.0);

    double sq = 0.0;
    for (size_t i = 0; i < k; ++i) {
        const double d = population[i].fitness - mu;
        sq += d * d;
    }
    const double sigma = (k > 0 ? std::sqrt(sq / k) : 0.0);

    if (sigma < Parameter::min_bound) {
        Parameter::min_bound = std::max(Parameter::min_bound / 1.05, 1e-13);
    }
}

void Ecga::run() noexcept {
    for (size_t gen = 1; gen < Parameter::max_gen; ++gen) {
        evaluate();
        adjustMinBoundFromTopQuartile();
        tournamentSelection();

        preEncode(encode_buffer_by_individual_);
        transpose(encode_buffer_by_individual_, encode_buffer_by_subproblem_);

        std::vector<CodeTable> codeTables(Parameter::dim_subproblems.size());
        encode(encode_buffer_by_subproblem_, codeTables);

        transpose(encode_buffer_by_subproblem_, encode_buffer_by_individual_);
        auto groups = buildMPM(encode_buffer_by_individual_, codeTables);
        generatePopulation(groups, codeTables, encode_buffer_by_individual_);
    }
    evaluate();
    const auto it = std::min_element(population.begin(), population.end(), compareChromosome);
    if (it != population.end()) {
        writeCSV(*it);
    }
}

void Ecga::evaluate() noexcept {
    const std::size_t N = population.size();
    parallel_index_range(0, N, [this](std::size_t i) { population[i].evaluate(); });
}

size_t Ecga::tournamentWinner(const std::vector<size_t>& order,
                              size_t start,
                              size_t tournament_size) const noexcept
{
    size_t winner = order[start];
    const size_t end = std::min(start + tournament_size, order.size());
    for (size_t i = start + 1; i < end; ++i) {
        if (population[order[i]].fitness < population[winner].fitness) {
            winner = order[i];
        }
    }
    return winner;
}

void Ecga::tournamentSelection() noexcept {
    const size_t N = population.size();
    const size_t s = Parameter::tour_size;
    thread_local static std::vector<size_t> order;
    order.resize(N);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), randomGenerator.engine());

    size_t pick = 0;
    for (size_t i = 0; i < N; ++i) {
        if (pick + s > N) {
            std::shuffle(order.begin(), order.end(), randomGenerator.engine());
            pick = 0;
        }
        size_t win = tournamentWinner(order, pick, s);
        mating_pool[i] = population[win];
        pick += s;
    }
}

void Ecga::preEncode(std::vector<std::vector<Point>>& out) noexcept {
    const size_t N = population.size();
    const size_t D = Parameter::dim_subproblems.size();
    if (out.size() != N) {
        out.resize(N);
    }
    parallel_index_range(0, N, [&](std::size_t i) {
        auto& row = out[i];
        if (row.size() != D) {
            row.resize(D);
        }
        for (size_t j = 0; j < D; ++j) {
            Point& p = row[j];
            const auto& sub = Parameter::dim_subproblems[j];
            const size_t block = sub.size();
            if (p.components.size() != block) {
                p.components.resize(block);
            }
            for (size_t k = 0; k < block; ++k) {
                p.components[k] = mating_pool[i].genes[sub[k]];
            }
        }
    });
}

template <typename T>
void Ecga::transpose(const std::vector<std::vector<T>>& in,
                     std::vector<std::vector<T>>& out) noexcept
{
    const size_t R = in.size();
    if (R == 0) {
        out.clear();
        return;
    }
    const size_t C = in[0].size();
    if (out.size() != C) {
        out.resize(C);
    }
    parallel_index_range(0, C, [&](std::size_t c) {
        auto& col = out[c];
        if (col.size() != R) {
            col.resize(R);
        }
        for (size_t r = 0; r < R; ++r) {
            col[r] = in[r][c];
        }
    });
}

void Ecga::encode(std::vector<std::vector<Point>>& points,
                  std::vector<CodeTable>& codeTables) noexcept
{
    Region::maxPoints = static_cast<size_t>(Parameter::pop_size * Parameter::split_rate);
    for (size_t i = 0; i < points.size(); ++i) {
        Region::codeIncrement = 0;
        Region subRegion(points[i].front().components.size());
        subRegion.insertPoints(points[i]);
        subRegion.split(codeTables[i]);
    }
    Parameter::split_rate *= Parameter::decrease_rate;
}

void Ecga::generatePopulation(const std::vector<Group>& groups,
                              const std::vector<CodeTable>& codeTables,
                              const std::vector<std::vector<Point>>& points) noexcept
{
    const size_t popSize = population.size();
    const size_t numCross = static_cast<size_t>(popSize * Parameter::cross_prob);
    const size_t remain   = popSize - numCross;

    const auto& dimSub = Parameter::dim_subproblems;
    auto& popData       = population;
    const auto& pool    = mating_pool;

    for (size_t k = 0; k < numCross; ++k) {
        auto& child = popData[k];
        for (const auto& g : groups) {
            for (auto idx : g.indices) {
                const size_t parent = randomGenerator.uniform(0, static_cast<int>(popSize - 1));
                const auto& pt     = points[parent][idx];
                codeTables[idx].sampleInto(static_cast<std::size_t>(pt.code), decoded_gene_buffer_);
                const auto& sub    = dimSub[idx];
                for (size_t t = 0; t < sub.size(); ++t) {
                    child.genes[sub[t]] = decoded_gene_buffer_[t];
                }
            }
        }
    }

    // Select the `remain` best (lowest fitness) from mating pool without copying all chromosomes.
    thread_local static std::vector<size_t> pool_rank;
    pool_rank.resize(popSize);
    std::iota(pool_rank.begin(), pool_rank.end(), 0);
    if (remain >= popSize) {
        for (size_t i = 0; i < popSize; ++i) {
            popData[numCross + i] = pool[i];
        }
    } else if (remain > 0) {
        std::nth_element(pool_rank.begin(), pool_rank.begin() + remain, pool_rank.end(),
                         [&](size_t a, size_t b) { return pool[a].fitness < pool[b].fitness; });
        for (size_t i = 0; i < remain; ++i) {
            popData[numCross + i] = pool[pool_rank[i]];
        }
    }
}

void Ecga::writeCSV(const Chromosome& best) const noexcept {
    static const std::vector<std::string> header = {
        "m","k","length","mode","eva_fun","seed","error"
    };
    std::lock_guard<std::mutex> guard(csv_write_mutex_);
    std::ifstream infile(Parameter::output_file);
    const bool isEmpty = infile.peek() == std::ifstream::traits_type::eof();
    infile.close();

    std::ofstream outfile(Parameter::output_file, std::ios::app);
    if (!outfile.is_open()) return;

    if (isEmpty) {
        for (size_t i = 0; i < header.size(); ++i) {
            outfile << header[i] << (i + 1 < header.size() ? ',' : '\n');
        }
    }
    outfile << Parameter::m << ','
            << Parameter::k << ','
            << Parameter::len_chromo << ','
            << "smsod" << ','
            << Parameter::nfe.load() << ','
            << Parameter::seed << ','
            << std::scientific << std::setprecision(16)
            << (best.fitness - (0))
            << '\n';
}