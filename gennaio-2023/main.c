/*
Realizzare in mpi C un'applicazione con numero di processi nproc = n*n che effettui:

- realizzare una topologia bidimensionale n*n

- ciascun processo genera un numero casuale minore di 100

- i processi che, nella propria riga, hanno estratto il numero maggiore
  effettuano la somma degli interi della propria riga
  e inviano la somma all'elemento della propria riga posizionato sulla diagonale principale della topologia.

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

void print_array(int *v, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("%d ", v[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ricaviamo n da nproc
    if (sqrt(size) * sqrt(size) != size)
    {
        if (rank == 0)
        {
            printf("Number of processes must be a perfect square.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int n = (int)sqrt(size);

    MPI_Comm topology;

    int dims[2] = {n, n};
    int periods[2] = {1, 1}; // 1 true indica periodicità

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &topology);

    int coords[2];
    MPI_Cart_coords(topology, rank, 2, coords);
    printf("Process %d is at coordinates (%d, %d)\n", rank, coords[0], coords[1]);

    // ogni processo genera un num casuale minore di 100
    int seed = time(NULL) + rank;
    srand(seed);
    int value = rand() % 100;

    // creiamo row comm
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, coords[0], rank, &row_comm);

    int row_rank, row_size;
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(row_comm, &row_rank);

    printf("Process %d coords=(%d,%d) row_rank=%d value=%d\n", rank, coords[0], coords[1], row_rank, value);
    fflush(stdout);

    // solo chi ha il num maggiore sulla propria riga deve calcolare la somma dei valori di una riga
    // facciamo allgather affinche tutti abbiano i valori della prima riga,
    int *values_row = malloc(row_size * sizeof(int));

    MPI_Allgather(&value, 1, MPI_INT, values_row, 1, MPI_INT, row_comm);

    int max_value;
    MPI_Allreduce(&value, &max_value, 1, MPI_INT, MPI_MAX, row_comm);

    int my_sum;

    if (value == max_value)
    {
        printf("Process %d has the max value of %d in row %d\n", rank, max_value, coords[0]);

        // questo processo calcola la somma della riga
        int sum = 0;
        for (int i = 0; i < row_size; i++)
        {
            sum += values_row[i];
        }
        printf("Process %d has the sum of its row %d\n", rank, sum);
        fflush(stdout);

        // questi processi devono inviare tale somma al processo della propria riga
        // che si trova sulla diag principale della topologia.

        int diag_coords[2] = {coords[0], coords[0]}; // stessa riga e colonna
        int diag_rank;
        MPI_Cart_rank(topology, diag_coords, &diag_rank);

        MPI_Send(&sum, 1, MPI_INT, diag_rank, 0, topology);
    }

    // i processi che in una riga si trovano sulla diag devono ricevere
    if (coords[0] == coords[1])
    {

        MPI_Recv(&my_sum, 1, MPI_INT, MPI_ANY_SOURCE, 0, topology, MPI_STATUS_IGNORE);
        printf("Process %d received the sum %d\n", rank, my_sum);
        fflush(stdout);
    }

    printf("\n");
    MPI_Finalize();
    return 0;
}