#include <iomanip>

#include "point.h"

Point::Point():code(-1){}

Point::Point(const std::vector<double>& comp):code(-1), components(comp) {}

Point::Point(std::vector<double>&& comp):code(-1), components(std::move(comp)) {}

void Point::setCode(int code) 
{
    this->code = code;
}

int Point::getCode() const 
{
    return this->code;
}

std::ostream& operator<<(std::ostream& os, const Point& point) 
{
    os << std::fixed << std::setprecision(2);    
    os << "Point " << "{ code: " << point.code << ", components: [";
    for (size_t i = 0; i < point.components.size(); i++) 
    {
        os << point.components[i];
        if (i < point.components.size() - 1) os << ", ";
    }
    os << "] }";
    return os;
}
