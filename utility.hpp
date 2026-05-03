// -*- c++ -*-
//
//  File:         utility.hpp
//
//  Description:  utility functions and procedures.
//
//  Author:       Fernando Lobo
//
//  Date:         June/1999
//
 
#ifndef _utility_hpp
#define _utility_hpp

#include <cstdio>
#include <fstream>

using namespace std;

void    error( char *msg );
void    errorcheck( char *str, bool condition );
double  sqr( double x );
double  min( double x );
int     min( int x, int y );
int     max( int x, int y );
double  log( double x, double b );
void    makeshuffle( int *shufflearray, const int n );
int     decode(int *schema,int S,int *index);

#endif

