#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0)
    {
        char message[] = "Hello from process 0";
        MPI_Send(message, sizeof(message), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent message: %s\n", message);
    }
    else if (world_rank == 1)
    {
        char message[100];
        MPI_Recv(message, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received message: %s\n", message);
    }
    else if (world_rank == 2)
        {
        char message[100];
        MPI_Recv(message, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received message: %s\n", message);
    }

    MPI_Finalize();
    return 0;
}
