#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main()
{
    int n, m, p;
    char fname[100];
    FILE * f;

    char header[1000];
    

    for (p = 1; p<10; p++)
    {
        sprintf(header, "#!/bin/bash\n#----------------------------------------------------\n# Example SLURM job script to run pure OpenMP applications\n#----------------------------------------------------\n#SBATCH -J p%d        # Job name\n#SBATCH -o p%d.o%%j    # Name of stdout output file\n#SBATCH -e p%d.o%%j    # Name of stderr output file\n#SBATCH -p normal         # Queue name\n#SBATCH -N 1              # Total number of nodes requested\n#SBATCH -n %d             # Total number of mpi tasks requested\n#SBATCH -t 00:02:00       # Run time (hh:mm:ss) - 1.5 hours\n# The next line is required if the user has more than one project\n#SBATCH -A UT-CS395  # Project/allocation number\n\n# This example will run on 1 node with %d OpenMP threads\n\n# Please do set the number of threads by yourself!\nexport OMP_NUM_THREADS=%d\n"
    , p, p, p, 2*p, p, p);
        sprintf(fname, "run_moc_p%d.sh", p);
        
        f = fopen(fname, "w");

        fprintf(f, "%s", header);
        for (n = 1; n<6; n++)
        {
            for (m = 1; m<6; m++)
            {
                fprintf(f, "./moc %.0f %.0f %d\n", pow(10,n), pow(10,m), p);
            }
        }

        fclose(f);
    }
    return 0;
}