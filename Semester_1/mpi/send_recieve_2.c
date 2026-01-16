#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 4) {
        if (world_rank == 0) {
            printf("Please run this program with 4 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    int next = (world_rank + 1) % world_size;
    int prev = (world_rank - 1 + world_size) % world_size;

    char message[100];

    if (world_rank == 0)
    {
        strcpy(message, "Hello from process 0");
        MPI_Send(message, strlen(message)+1, MPI_CHAR, next, 0, MPI_COMM_WORLD);
        MPI_Recv(message, 100, MPI_CHAR, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received message: %s\n", message);
    }
    else
    {
        MPI_Recv(message, 100, MPI_CHAR, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received message: %s\n", world_rank, message);
        sprintf(message, "Hello from process %d", world_rank);
        MPI_Send(message, strlen(message)+1, MPI_CHAR, next, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
