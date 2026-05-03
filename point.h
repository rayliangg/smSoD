#pragma once

#include <iostream>
#include <vector>

class Point 
{
public:
    int code;                 
    std::vector<double> components;

    //ctor
    Point();                              
    Point(const std::vector<double>& comp);        
    Point(std::vector<double>&& comp);             

    void setCode(int code);
    int getCode() const;

    friend std::ostream& operator<<(std::ostream& os, const Point& point);
};
