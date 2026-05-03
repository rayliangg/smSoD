#pragma once

#include <vector>

#include "bound.h"
#include "random.hpp"

extern RandomGenerator randomGenerator;

class CodeTable {
public:
    std::vector< std::vector<Bound> > Table;
    
    CodeTable(){
        Table.reserve(100);
    }

    void add(std::vector<Bound>&& bounds) {
        Table.emplace_back(std::move(bounds));
    }

    std::vector<double> get(int index) const {
        const auto& bounds = Table[index];
        std::vector<double> tmp;
        tmp.reserve(bounds.size());
        for (const auto& b : bounds) {
            tmp.emplace_back(randomGenerator.uniform(b.lower, b.upper));
        }        
        return tmp;
    }

    /** Write random decoded values into out (reuses out's capacity; hot path). */
    void sampleInto(std::size_t index, std::vector<double>& out) const {
        const auto& bounds = Table[index];
        out.resize(bounds.size());
        for (std::size_t i = 0; i < bounds.size(); ++i) {
            out[i] = randomGenerator.uniform(bounds[i].lower, bounds[i].upper);
        }
    }
};
