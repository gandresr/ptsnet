#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>
#include <sys/time.h>

int main(int argc, char* argv[])
{
    int i, j, n;
    float a[100][100] = {0};
    FILE *f;
    struct timeval start, stop;

    gettimeofday(&start, NULL);

    //f = fopen("test.txt", "w");
    n = 50;

    for (j=0; j<n; j++)
    {
        a[0][j] = j;
        //printf("%f,", a[0][j]);
    }
    
    //printf("\n");
    omp_set_num_threads(100);

    for (i=1; i<n; i++)
    {
        #pragma omp parallel for
        for (j=1; j<n-1; j++)
        {
            a[i][j] = a[i-1][j+1] - a[i-1][j-1];
            //printf("%f,", a[i][j]);
        }
        //printf( "\n");
    }

    gettimeofday(&stop, NULL);
    printf("took %lu\n", stop.tv_usec - start.tv_usec);;
    // fclose(f);
    return 0;
}