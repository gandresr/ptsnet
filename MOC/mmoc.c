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
    for (j = 0; j<2; j++) 
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

		for (j = 1; j<2; j++)
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
        int ii;
        int jj = (j % 2 == 0);
        ii = 1 - jj;

		H[0][ii] = H[0][jj]; // Reservoir
		V[0][ii] = V[1][jj] + g/a*(H0 - H[1][jj]) - f*dt*V[1][jj]*fabs(V[1][jj])/(2*D);
		V[n][ii] = V0 * s[j];
		H[n][ii] = H[n-1][jj] - a/g*(V[n][ii] - V[n-1][jj]) - a/g*(f*dt*V[n-1][jj]*fabs(V[n-1][jj])/(2*g));
		#pragma omp parallel for shared(V, H) schedule(static)
		for (i=1; i<n; i++)
		{
			V[i][ii] = (V[i-1][jj] + V[i+1][jj] + g/a*(H[i-1][jj] - H[i+1][jj]) - f*dt/(2*D)*(V[i-1][jj]*fabs(V[i-1][jj]) + V[i+1][jj]*fabs(V[i+1][jj])))/2;
			H[i][ii] = (a/g*(V[i-1][jj] - V[i+1][jj]) + H[i-1][jj] + H[i+1][jj] - a/g*(f*dt/(2*D)*(V[i-1][jj]*fabs(V[i-1][jj]) - V[i+1][jj]*fabs(V[i+1][jj]))))/2;
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