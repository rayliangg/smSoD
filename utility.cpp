// -*- c++ -*-
//
//  File:         utility.cpp
//
//  Description:  utility functions and procedures.
//
//  Author:       Fernando Lobo
//
//  Date:         June/1999
//

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "random.hpp"
#include "utility.hpp"
using namespace std;

extern RandomGenerator randomGenerator;

// outputs an error msg and stops the program 
void error( char *msg )
{
	cout << msg << endl;
	exit(1);
}

// if 'condition' is false, print an error message and abort the program
void errorcheck( char *str, bool condition )
{
	if( !condition )
	{
		cout << "ERROR: " << str << endl;
		exit( 1 );
	}
}

double sqr( double x )
{
	return x*x;
}

int min( int x, int y )
{
	if (x<y) return x;
	else return y;
}

int max( int x, int y )
{
	if (x>y) return x;
	else return y;
}

// return log x base b
double log( double x, double b )
{
	return log(x)/log(b); 
}

// make a random n-permutation of the numbers 0,1,2,...,n-1
void makeshuffle( int *shufflearray, const int n )
{
	int i;

	// initialize
	for( i=0; i<n; i++ ) shufflearray[i] = i;
	// shuffle
	for( i=0; i<n-1; i++ )
	{
		int other = randomGenerator.uniform( i, n-1 );
		// swap( shufflearray[other], shufflearray[i] );
		int temp = shufflearray[other];
		shufflearray[other] = shufflearray[i];
		shufflearray[i] = temp;
	}
}

int decode(int *schema,int S,int *index)
{
	int addr=0;
	for(int i=0;i<S;++i)
		addr+=schema[i]*index[i];
	return addr;
}