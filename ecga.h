// ecga.h
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "chromosome.h"
#include "codetable.h"
#include "mpm.h"
#include "parameter.h"
#include "point.h"

class Ecga {
public:
    std::vector<Chromosome> population;
    std::vector<Chromosome> mating_pool;

    double previousStd{};
    double previousMean{};

    Ecga() noexcept;

    void adjustMinBoundFromTopQuartile() noexcept;
    void run() noexcept;
    void evaluate() noexcept;
    std::size_t tournamentWinner(const std::vector<std::size_t>& order,
                                  std::size_t start,
                                  std::size_t tournament_size) const noexcept;
    void tournamentSelection() noexcept;

    void preEncode(std::vector<std::vector<Point>>& out) noexcept;
    void encode(std::vector<std::vector<Point>>& points,
                std::vector<CodeTable>& codeTables) noexcept;
    void generatePopulation(const std::vector<Group>& groups,
                            const std::vector<CodeTable>& codeTables,
                            const std::vector<std::vector<Point>>& points) noexcept;

    void writeCSV(const Chromosome& best) const noexcept;

private:
    template <typename T>
    static void transpose(const std::vector<std::vector<T>>& in,
                          std::vector<std::vector<T>>& out) noexcept;

    static std::mutex csv_write_mutex_;

    std::vector<std::vector<Point>> encode_buffer_by_individual_;
    std::vector<std::vector<Point>> encode_buffer_by_subproblem_;
    std::vector<double> decoded_gene_buffer_;
};