#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>


void printM(double ** A, int n, int m)
{
	int i,j;
	for (j = 0; j<m; j++)
	{
		for (i = 0; i<n; i++)
		{
			printf("%.3f, ", A[i][j]);
		}
		printf("\n");
	}
}

int main(int argc, char** argv) 
{
	int i, j, n, m, p;
    double start, end, D, AR, H0, Q0, V0, a, l, f, pi;
	double t0, dt, g, tf;
    struct timeval timecheck;	
	
	if (argc < 4)
		return 0;
	
	n = atoi(argv[1]);
	m = atoi(argv[2]);
	p = atoi(argv[3]);


	pi = 3.1416;
	D = 0.5; // [m]
	AR = pi*pow(D,2)/4; // [m^2]

	H0 = 1000; // [m]
	Q0 = 1.06;
	V0 = Q0 / AR; // [m/s]

	a = 2500; // [m/s]
	l = 1000; // [m]
	f = 0.02;

	t0 = 0; // [s] - This is not used, for every simulation, it is assumed t0 = 0
	dt = 1.*l/(a*n);
	tf = 1; // [s]
	g = 32.2; // [m/s]

	omp_set_num_threads(p);

    double ** H = (double **) malloc(sizeof(double *)*n+1);
    double ** V = (double **) malloc(sizeof(double *)*n+1);
    double * s = (double *) malloc(sizeof(double)*m);

	for (j = 0; j<m; j++)
	{
		s[j] = 1 - (1.*j)/10;
		if (s[j] < 0)
		{
			s[j] = 0;
		}
	}

	for(i = 0; i<n+1; i++)
	{
		H[i] = (double *) malloc(sizeof(double)*m);
		V[i] = (double *) malloc(sizeof(double)*m);
		H[i][0] = H0 - (i*l/n)*f*pow(V0,2)/(2*g*D);
		V[i][0] = V0;

		for (j = 1; j<m; j++)
		{
			H[i][j] = 0;
			V[i][j] = 0;
		}
	}


    // START > Wall-clock timing in [s]
    gettimeofday(&timecheck, NULL);
    start = (double)timecheck.tv_sec + (double)timecheck.tv_usec/1e6;

	for (j=1; j<m; j++)
	{
		H[0][j] = H[0][j-1]; // Reservoir
		V[0][j] = V[1][j-1] + g/a*(H0 - H[1][j-1]) - f*dt*V[1][j-1]*fabs(V[1][j-1])/(2*D);
		V[n][j] = V0 * s[j];
		H[n][j] = H[n-1][j-1] - a/g*(V[n][j] - V[n-1][j-1]) - a/g*(f*dt*V[n-1][j-1]*fabs(V[n-1][j-1])/(2*g));
		#pragma omp parallel for shared(V, H) schedule(static)
		for (i=1; i<n; i++)
		{
			V[i][j] = (V[i-1][j-1] + V[i+1][j-1] + g/a*(H[i-1][j-1] - H[i+1][j-1]) - f*dt/(2*D)*(V[i-1][j-1]*fabs(V[i-1][j-1]) + V[i+1][j-1]*fabs(V[i+1][j-1])))/2;
			H[i][j] = (a/g*(V[i-1][j-1] - V[i+1][j-1]) + H[i-1][j-1] + H[i+1][j-1] - a/g*(f*dt/(2*D)*(V[i-1][j-1]*fabs(V[i-1][j-1]) - V[i+1][j-1]*fabs(V[i+1][j-1]))))/2;
		}
	}
	// END
    gettimeofday(&timecheck, NULL);
    end = (double)timecheck.tv_sec + (double)timecheck.tv_usec/1e6;
    printf("SEQ: %lf seconds elapsed\n", (end - start));
	printf("\n");
	
	// printM(H, n+1, m);
	// printf("\n");
	// printM(V, n+1, m);

	for (i = 0; i<n+1; i++)
	{
		free(H[i]);
		free(V[i]);
	}

	free(H);
	free(V);
	free(s);

	return 0;
}