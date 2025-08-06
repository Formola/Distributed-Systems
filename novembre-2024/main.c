#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

int main(int argc, char *argv[])
{
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;

    if (size == (int)sqrt(size) * (int)sqrt(size))
    {
        n = (int)sqrt(size);
    }
    else
    {
        perror("Wrong size.\n");
        return 1;
    }

    int dims[2] = {n, n};
    int periods[2] = {1, 1};

    MPI_Comm topology;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &topology);

    srand(time(NULL) + rank);
    int value;

    value = rand() % 100;

    MPI_Barrier(topology);
    int source_rank, dest_rank;
    MPI_Cart_shift(topology, 1, 2, &source_rank, &dest_rank);
    printf("Sono il processo %d ed il rank source è %d e quello di dest %d, inoltre ho generato il valore %d\n", rank, source_rank, dest_rank, value);
    int buf;
    MPI_Status status;
    MPI_Sendrecv(&value, 1, MPI_INT, dest_rank, 0, &buf, 1, MPI_INT, source_rank, 0, topology, &status);

    value = buf;
    printf("Sono il processo %d ed ho appena ricevuto il valore %d dal processo di rank %d.\n", rank, value, source_rank);

    MPI_Comm row_comm;
    int my_row = rank / n;

    MPI_Comm_split(topology, my_row, rank, &row_comm);
    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    int sum;
    MPI_Reduce(&value, &sum, 1, MPI_INT, MPI_SUM, 0, row_comm);
    if (row_rank == 0)
    {
        printf("Sono il processo di rank %d e la somma dei valori della mia riga è %d\n", rank, sum);
    }

    MPI_Finalize();
    return 0;
}