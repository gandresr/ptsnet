#include "headers.h"

void baseSequential(int n, int m) 
{
    int i,j;
    double *** A = initialMatrix(n, m);

    // n: rows = time scale
    // m: cols = space scale

    printV(A, n, m);
    printf("\n");
    
    for (j = 1; j<n; j++)
    {
        for (i = 0; i<m; i++)
        {
            if ((i > 0) && (i < m-1)) {
                *A[j][i] = (*A[j-1][i-1] + *A[j-1][i+1])/2;
            }
            else if (i == 0) {
                *A[j][i] = *A[j-1][i+1];
            }
            else if (i == m-1) {
                *A[j][i] = *A[j-1][i-1];
            }
        }
    }

    printV(A, n, m);

    free(A);
}

int main(int argc, char ** argv)
{
    int n, m;

    if (argc == 2)
    {
        n = atoi(argv[1]);
    } 
    else if (argc > 2)
    {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
    } 
    else 
    {
        n = 10;
        m = 10;
    }

    baseSequential(n,m);

    return 0;
}