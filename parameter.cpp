#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "parameter.h"

long Parameter::seed = 12;          

// ecga parameter
int Parameter::pop_size = 200;             
int Parameter::tour_size = 8;              
double Parameter::cross_prob = 0.975;      
int Parameter::max_gen = 2000;            
int Parameter::len_chromo = 10;              

// msod parameter
vector<vector<int>> Parameter::dim_subproblems({{0}}); 

// discretization parameter
double Parameter::split_rate = 0.5;        
double Parameter::decrease_rate = 0.998;        
double Parameter::min_bound = 1e-5;        
double Parameter::lower = -200;           
double Parameter::upper = 200;            

// problem parameter
std::atomic<int> Parameter::nfe{0};
int Parameter::k = 2;                   
int Parameter::m = 5;

// result output parameter
string Parameter::output_file = "../mk.csv"; 


// ./your_program --seed 123 --pop_size 300 --funNum 10 --output_file results.csv

void Parameter::parseArguments(int argc, char* argv[]) 
{
    for (int i = 1; i < argc; ++i) 
    {
        string arg = argv[i];
        if (arg == "--k" && i + 1 < argc) {
            Parameter::k = atoi(argv[++i]);
        }
        else if (arg == "--m" && i + 1 < argc) {
            Parameter::m = atoi(argv[++i]);
        }       
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n"
                      << "Usage: " << (argc > 0 ? argv[0] : "main")
                      << " [--k <int>] [--m <int>]\n";
            std::exit(1);
        }
    }
}

void Parameter::update() {
    Parameter::dim_subproblems.clear();
    Parameter::len_chromo = Parameter::k * Parameter::m;
    Parameter::dim_subproblems.resize(Parameter::m);
    // Parameter::max_gen = (Parameter::len_chromo * 5000) / Parameter::max_gen;
    for (int i = 0; i < Parameter::m; ++i){   
        int left = i * Parameter::k;
        Parameter::dim_subproblems[i].resize(Parameter::k);
        for (int j = 0; j < Parameter::k; ++j){
            Parameter::dim_subproblems[i][j] = (left + j);
        }
    }

    // for (const auto& dim_subproblem : dim_subproblems){
    //     for (const auto& index :dim_subproblem){
    //         cout << index << " ";
    //     }
    //     cout << endl;
    // }
}
