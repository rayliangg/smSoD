#include <random>

#include "ecga.h"
#include "parameter.h"
#include "random.hpp"

using namespace std;

RandomGenerator randomGenerator;

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

int main(int argc, char* argv[]) 
{    
    Parameter::parseArguments(argc, argv);
    Parameter::update();

    random_device rd;  
    Parameter::seed = static_cast<long>(rd());  
    randomGenerator.seed(Parameter::seed);
    
    Ecga ecga;
    ecga.run();
    return 0;
}
