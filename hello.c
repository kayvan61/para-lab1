#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello, World.  I am %d of %d\n", rank, size);
    // MPI_Recv(void* data, int count, MPI_Datatype datatype, int from, int tag, MPI_Comm comm, MPI_Status* status);
    if(rank == 0) {
      int* a = malloc(sizeof(int));
      a[rank] = rank*3;
      MPI_Send(&rank, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
      MPI_Recv(a, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("The data recv'd is: %d\n", *a);
    }
    else if(rank == 1){
      int* a = malloc(sizeof(int));
      a[rank] = rank*3;
      MPI_Recv(a, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
      printf("The data recv'd is: %d\n", *a);
    }
    
    MPI_Finalize();
    return 0;
}
