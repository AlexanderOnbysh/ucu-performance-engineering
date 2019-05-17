#include <mpi.h>
#include <stdio.h>

int main (int argc, char *argv[]) {
    int rank, size;

    //initialize MPI library
    MPI_Init (&argc, &argv);

    //get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //get my process id(rank)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //do something
    printf ("Print message from rank (process id) %d \n", rank);
    if (rank == 0) printf("MPI_WORLD size = %d processes\n", size);

    //MPI cleanup
    MPI_Finalize();
    return 0;
}