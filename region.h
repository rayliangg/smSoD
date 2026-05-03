// region.h
#pragma once

#include <cstdint>
#include <vector>

#include "bound.h"
#include "codetable.h"
#include "parameter.h"
#include "point.h"

class Region {
public:
    std::vector<Point*> points;
    std::size_t dimensions;
    std::vector<Bound> bounds;
    
    static std::uint32_t codeIncrement;
    static std::size_t maxPoints;

    Region(std::size_t dims);

    void insertPoints(std::vector<Point>& newPoints);
    void split(CodeTable& CT);
    Point getRandomPoint() const;
    std::vector<Region> splitRegion(const Point& splitPoint) const;
    Point findBestSplitPoint() const;
};