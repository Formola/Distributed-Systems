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
        perror("Il numero di processi deve essere un quadrato perfetto\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {n, n};
    int periods[2] = {1, 1};
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &topology);
    MPI_Cart_coords(topology, rank, ndims, coords);

    printf("\nProcesso %d di coordinate (%d,%d)\n", rank, coords[0], coords[1]);
    fflush(stdout);

    srand(time(NULL) + rank);
    int value = rand() % 100;

    printf("\nProcesso %d di coordinate (%d,%d) ha generato il valore %d\n", rank, coords[0], coords[1], value);
    fflush(stdout);

    // -i processi su una stessa riga fanno scorrere verso destra
    // di 2 posizioni il proprio valore

    MPI_Comm row_comm;
    int remain_dims[2] = {0, 1};
    MPI_Cart_sub(topology, remain_dims, &row_comm);

    int source, dest;
    MPI_Cart_shift(topology, 1, 2, &source, &dest);

    int recv_value;
    MPI_Sendrecv(&value, 1, MPI_INT, dest, 0, &recv_value, 1, MPI_INT, source, 0, topology, MPI_STATUS_IGNORE);

    printf("\nProcesso %d di coordinate (%d,%d) ha ricevuto il valore %d dal processo %d\n", rank, coords[0], coords[1], recv_value, source);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    // si calcola la somma degli elementi in ogni riga
    // solo i primi processi di ogni riga stampano il risultato
    int row_sum;
    MPI_Allreduce(&recv_value, &row_sum, 1, MPI_INT, MPI_SUM, row_comm);

    if (coords[1] == 0)
    {
        printf("\nProcesso %d di coordinate (%d,%d) ha somma di riga %d\n", rank, coords[0], coords[1], row_sum);
        fflush(stdout);
    }

    printf("\n");
    MPI_Finalize();
    return 0;
}