#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <unistd.h>
#include <sys/time.h>

double randfrom(double min, double max);
double ** vector(int n, char type);
double *** initialMatrix(int n, int m);
void printV(double *** x, int n, int m);