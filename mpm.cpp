#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <queue>
#include <unordered_map>

#include "mpm.h"

using namespace std;

Group::Group() : num_counts{1}, modelComplexity{-1}, populationComplexity{-1} {}
size_t Group::size() const noexcept { return indices.size(); }
double Group::getModelComplexity() const noexcept { return modelComplexity; }
double Group::getPopulationComplexity() const noexcept { return populationComplexity; }

Group Group::merge(const Group& other) const {
    Group merged;
    merged.indices.reserve(size() + other.size());
    std::merge(indices.begin(), indices.end(),
               other.indices.begin(), other.indices.end(),
               std::back_inserter(merged.indices));

    std::uint64_t a = num_counts, b = other.num_counts;
    merged.num_counts = (b != 0 && a > UINT64_MAX / b)
                      ? UINT64_MAX
                      : a * b;
    return merged;
}

void count(Group& group,
           const std::vector<std::vector<Point>>& encodedPoints)
{
    const size_t N = static_cast<size_t>(Parameter::pop_size);
    const size_t D = group.size();
    const auto& idx = group.indices;

    // Reuse map across many count() calls inside buildMPM (avoids repeated heap churn).
    thread_local std::unordered_map<std::size_t, std::size_t> probs;
    probs.clear();
    probs.reserve(N);

    std::hash<int> hasher;
    for (size_t i = 0; i < N; ++i) {
        size_t h = 0;
        for (size_t k = 0; k < D; ++k) {
            int code = encodedPoints[i][idx[k]].getCode();
            h ^= hasher(code) + (h << 6) + (h >> 2);
        }
        ++probs[h];
    }

    double sum = 0.0;
    for (auto& [_, cnt] : probs) {
        double p = double(cnt) / N;
        sum -= p * std::log2(p);
    }
    group.populationComplexity = sum * N;
    group.modelComplexity = std::log2(N + 1.0) * (group.num_counts - 1);
}

struct MergeCand {
    size_t i, j;
    double gain;
};
struct Cmp {
    bool operator()(MergeCand const& a, MergeCand const& b) const {
        return a.gain < b.gain;  // max-heap
    }
};

std::vector<Group> buildMPM(
    const std::vector<std::vector<Point>>& encodedPoints,
    const std::vector<CodeTable>& codeTables)
{
    
    const size_t D = codeTables.size();

    std::vector<Group> groups;
    groups.reserve(D);
    for (size_t i = 0; i < D; ++i) {
        Group g;
        g.indices.push_back(i);
        g.num_counts = codeTables[i].Table.size();  
        count(g, encodedPoints);                     
        groups.push_back(std::move(g));
    }

    size_t G = groups.size();

    std::vector<bool> alive(G, true);

    std::vector<double> oldMDL(G);
    for (size_t i = 0; i < G; ++i) {
        oldMDL[i] = groups[i].getModelComplexity()
                  + groups[i].getPopulationComplexity();
    }

    std::priority_queue<MergeCand, std::vector<MergeCand>, Cmp> heap;
    for (size_t i = 0; i < G; ++i) {
        for (size_t j = i + 1; j < G; ++j) {
            Group tmp = groups[i].merge(groups[j]);
            count(tmp, encodedPoints);
            double newMDL = tmp.getModelComplexity() + tmp.getPopulationComplexity();
            double gain   = (oldMDL[i] + oldMDL[j]) - newMDL;
            if (gain > 0.0) {
                heap.push({i, j, gain});
            }
        }
    }

    while (!heap.empty()) {
        auto top = heap.top(); heap.pop();
        if (top.gain <= 0.0) break;
        if (!alive[top.i] || !alive[top.j]) continue;

        Group newG = groups[top.i].merge(groups[top.j]);
        count(newG, encodedPoints);

        double newMDL = newG.getModelComplexity() + newG.getPopulationComplexity();
        size_t newIdx = groups.size();
        groups.push_back(std::move(newG));
        alive.push_back(true);
        oldMDL.push_back(newMDL);

        alive[top.i] = alive[top.j] = false;

        for (size_t k = 0; k < newIdx; ++k) {
            if (!alive[k]) continue;
            Group tmp = groups[k].merge(groups[newIdx]);
            count(tmp, encodedPoints);
            double mdl2  = tmp.getModelComplexity() + tmp.getPopulationComplexity();
            double gain2 = (oldMDL[k] + oldMDL[newIdx]) - mdl2;
            if (gain2 > 0.0) {
                heap.push({k, newIdx, gain2});
            }
        }
    }

    std::vector<Group> result;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (alive[i]) {
            result.push_back(std::move(groups[i]));
        }
    }
    return result;
}