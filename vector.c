
#include "headers.h"

// Generates random number between min and max
double randfrom(double min, double max)
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

// Generates vector of size n with random values
double ** vector(int n, char type)
{
	int i;
	double ** yy = (double **) malloc(sizeof(double *) * n);
	double * y = (double *) malloc(sizeof(double) * n) ;

	for (i = 0; i<n; i++)
	{
		yy[i] = &(y[i]);
        if (type == 'r')
		    y[i] = randfrom(0, 1);
        else 
		    y[i] = 0;
	}
	return yy;
}

// Generates array of n pointers to random m-dimensional vectors
double *** initialMatrix(int n, int m)
{
	int i;
	double *** yyy = (double ***) malloc(sizeof(double **) * n);
	
    yyy[0] = vector(m, 'r');
	for (i=1; i<n; i++) {
		yyy[i] = vector(m, 'z'); // vector of zeros
	}
	return yyy;
}

// Prints components of vector
void printV(double *** x, int n, int m) {
	int i, j;
	for (i=0; i<n; i++) {
		printf("(");
		for (j=0; j<m; j++)
		{
			printf("%.3f", *x[i][j]);
			if ((m > 1) && (j<m-1))
				printf(",    ");
		}
		printf(")\n");
	}
}