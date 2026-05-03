#include <iomanip>
#include <iostream>

#include "chromosome.h"
#include "objfun.h"


extern RandomGenerator randomGenerator;
Chromosome::Chromosome(int len_chromo)
    : genes(len_chromo)
{}


void Chromosome::random() {
    for (auto& gene : genes) {
        gene = randomGenerator.uniform(Parameter::lower, Parameter::upper);
    }
}


void Chromosome::evaluate() {
    Parameter::nfe.fetch_add(1, std::memory_order_relaxed);
    this->fitness = objectiveFunction(this->genes);
}

std::ostream& operator<<(std::ostream& os, const Chromosome& chrom) {
    os << std::scientific << std::setprecision(8);    
    os << "Chromosome { fitness: " << chrom.fitness << ", genes: [";

    if (!chrom.genes.empty()) {
        os << chrom.genes[0];
        for (size_t i = 1; i < chrom.genes.size(); ++i) {
            os << ", " << chrom.genes[i];
        }
    }
    os << "] }" << std::endl;
    return os;
}
