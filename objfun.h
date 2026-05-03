// objfun.h — separable ridge-style test objective for SMSoD runs
#pragma once

#include <cstddef>
#include <vector>

#include "parameter.h"
#include "random.hpp"

extern RandomGenerator randomGenerator;

double objectiveFunction(const std::vector<double>& genes) noexcept {
    // Initialize random constants once
    static const auto constants = []() {
        std::vector<double> tmp;
        tmp.reserve(Parameter::m);
        for (std::size_t i = 0; i < Parameter::m; ++i) {
            tmp.push_back(randomGenerator.uniform(Parameter::lower, Parameter::upper));
        }
        return tmp;
    }();

    const std::size_t m = static_cast<std::size_t>(Parameter::m);
    const std::size_t k = static_cast<std::size_t>(Parameter::k);
    double fitness = 0.0;

    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t offset = i * k;
        const double d0 = genes[offset] - constants[i];
        fitness += d0 * d0;
        
        for (std::size_t j = 1; j < k; ++j) {
            const double diff = genes[offset + j] - genes[offset + j - 1];
            fitness += diff * diff;
        }
    }

    return fitness;
}
