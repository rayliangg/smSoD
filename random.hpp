#ifndef _random_hpp
#define _random_hpp

#include <random>

class RandomGenerator {
private:
    std::mt19937 engine_;

public:
    RandomGenerator() = default;
    ~RandomGenerator() = default;

    void seed(long value);
    std::mt19937& engine() { return engine_; }

    double uniform01();
    double uniform(double min_value, double max_value);
    int uniform(int min_value, int max_value);
    double exponential(double mean);
    bool bernoulli(double probability);
    bool flip(double probability) { return bernoulli(probability); }
    long geometric(double failure_probability);
    double normal01();
    double normal(double mean, double variance);
};

#endif
