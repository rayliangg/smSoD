#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>

#include "random.hpp"

void RandomGenerator::seed(long value)
{
	const auto u = static_cast<std::uint64_t>(static_cast<long long>(value));
	std::array<std::uint32_t, 2> parts{
		static_cast<std::uint32_t>(u),
		static_cast<std::uint32_t>(u >> 32u)};
	std::seed_seq seq(parts.begin(), parts.end());
	engine_.seed(seq);
}

double RandomGenerator::uniform01()
{
	// (0, 1), excluding 0 (safe for log() in derived samples).
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double u;
	do {
		u = dist(engine_);
	} while (u <= 0.0);
	return u;
}

double RandomGenerator::uniform(double min_value, double max_value)
{
	assert(max_value > min_value);
	return (max_value - min_value) * uniform01() + min_value;
}

int RandomGenerator::uniform(int min_value, int max_value)
{
	assert(max_value > min_value);
	std::uniform_int_distribution<int> dist(min_value, max_value);
	return dist(engine_);
}

double RandomGenerator::exponential(double mean)
{
	double u = uniform01();
	return -mean * std::log(u);
}

bool RandomGenerator::bernoulli(double probability)
{
	std::bernoulli_distribution dist(probability);
	return dist(engine_);
}

long RandomGenerator::geometric(double failure_probability)
{
	double u = uniform01();
	return static_cast<long>(std::log(u) / std::log(1.0 - failure_probability));
}

double RandomGenerator::normal01()
{
	double U1, U2, V1, V2, W, Y, X1, X2;

	do {
		U1 = uniform01();
		U2 = uniform01();
		V1 = 2 * U1 - 1;
		V2 = 2 * U2 - 1;
		W = V1 * V1 + V2 * V2;
	} while (W > 1);

	Y = std::sqrt((-2 * std::log(W)) / W);
	X1 = V1 * Y;
	X2 = V2 * Y;
	(void)X2;

	return X1;
}

double RandomGenerator::normal(double mean, double variance)
{
	double X = normal01();
	double stddev = std::sqrt(variance);
	return mean + stddev * X;
}
