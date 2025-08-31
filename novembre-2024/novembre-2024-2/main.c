/*

Scrivere in MPI C un programma con nproc = n*n che effettui:

- realizzare una topologia bidimensionale n*n

- ciascun processo genera un intero casuale minore di 100

- i processi su una stessa riga fanno scorrere verso destra di 2 posizioni il proprio valore

- viene calcolata la somma degli elementi di ciascuna riga.
  Solo i primi processi di ciascuna riga visualizzano ìa video la somma.

*/



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

    int n = (int)sqrt(size);
    if (n * n != size)
    {
        fprintf(stderr, "Number of processes must be a perfect square.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {n,n};
    int periods[2] = {1,1};
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &topology);
    MPI_Cart_coords(topology, rank, ndims, coords);

    printf("Process %d is at position (%d, %d)\n", rank, coords[0], coords[1]);
    fflush(stdout);

    srand(time(NULL) + rank);
    int my_value = rand() % 100;
    printf("Process %d generated value %d\n", rank, my_value);
    fflush(stdout);

    MPI_Comm row_comm;
    int remain_dims[2] = { 0 , 1 }; // 0 riga -> creiamo un comm per ogni riga
    int row_rank, row_size;
    MPI_Cart_sub(topology, remain_dims, &row_comm);
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    int source, dest;
    int recv_value;
    MPI_Cart_shift(topology, 1, 2, &source, &dest); // shift di 2 posizioni a destra

    MPI_Sendrecv(&my_value, 1, MPI_INT, dest, 0, &my_value, 1, MPI_INT, source, 0, topology, MPI_STATUS_IGNORE);

    printf("Process %d received value %d from process %d\n", rank, my_value, source);
    fflush(stdout);

    int row_sum = 0;
    MPI_Reduce(&my_value, &row_sum, 1, MPI_INT, MPI_SUM, 0, row_comm);

    if (row_rank == 0)
    {
        printf("Process %d is the first in its row and received value %d\n", rank, my_value);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}