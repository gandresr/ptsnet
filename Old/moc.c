#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[])
{
    int i, j, n;
    float a[100][100] = {0};
    FILE *f;

    f = fopen("test.txt", "w");
    n = 10;

    for (j=0; j<n; j++)
    {
        a[0][j] = j;
        printf("%f\n", a[0][j]);
    }

    omp_set_num_threads(10);

    for (i=1; i<n; i++)
    {
        #pragma omp parallel for
        for (j=1; j<n; j++)
        {
            int tid;
            tid = omp_get_thread_num();
            a[i][j] = a[i-1][j+1] - a[i-1][j-1];
            fprintf(f, "i=%d, j=%d, a[i][j] = %f, thread = %d \n", i, j, a[i][j], tid);
        }
    }
    fclose(f);
    return 0;
}