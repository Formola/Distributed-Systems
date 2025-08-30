/*
Realizzare un programma mpi C con nproc=nxn che effettui:

- realizzare una topologia bidimensionale nxn.

- i processi della prima colonna leggono da file diversi
  un vettore di DIM interi e lo inviano a tutti i processi della propria riga.

- i processi effettuano uno shift facendo scorrere
  verso l'alto sulla colonna il proprio vettore.

- a questo punto, viene calcolato il vettore dei massimi per ogni riga.
  sta scritto male!.

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define DIM 4

void print_vector(int *v, int dim)
{

    for (int i = 0; i < dim; i++)
    {
        printf("%d ", v[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = DIM;

    int n = sqrt(size);
    if (n * n != size)
    {
        if (rank == 0)
            fprintf(stderr, "Error: number of processes must be a perfect square\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm topology; // top 2d

    int ndims = 2;
    int dims[2] = {n, n};    // n proc per dimensione
    int periods[2] = {1, 1}; // periodica su entrambe le dimensioni
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &topology);
    MPI_Cart_coords(topology, rank, ndims, coords);

    printf("Process %d is at coordinates (%d, %d)\n", rank, coords[0], coords[1]);

    int V[DIM];

    // processi sulla prima colonna della topologia leggono
    if (coords[1] == 0)
    {

        char filenames[][20] = {"input1.txt", "input2.txt", "input3.txt"};

        FILE *fp = fopen(filenames[coords[0]], "r");

        if (fp == NULL)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < dim; i++)
        {
            fscanf(fp, "%d", &V[i]);
        }
        fclose(fp);

        printf("Process %d read vector: ", rank);
        print_vector(V, dim);
        fflush(stdout);
    }

    MPI_Comm row_comm;
    int row_rank;
    int remain_dims[2] = {0, 1};
    MPI_Cart_sub(topology, remain_dims, &row_comm);
    MPI_Comm_rank(row_comm, &row_rank);

    // il proc in prima colonna di ogni riga distribuisce V
    MPI_Bcast(V, dim, MPI_INT, 0, row_comm);

    printf("Process %d received vector: ", rank);
    print_vector(V, dim);
    fflush(stdout);

    // i processi effettuano uno shift facendo scorrere verso l'alto in colonna il vettore V
    int source, dest;
    int recv_V[DIM];
    MPI_Cart_shift(topology, 0, -1, &source, &dest);
    MPI_Sendrecv(V, dim, MPI_INT, dest, 0, recv_V, dim, MPI_INT, source, 0, topology, MPI_STATUS_IGNORE);

    printf("Process %d sent vector to %d and received from %d: ", rank, dest, source);
    print_vector(recv_V, dim);
    fflush(stdout);

    // vettore dei massimi in ogni riga bah

    // find local max in each vector
    int max = recv_V[0];
    for (int i = 1; i < dim; i++)
    {
        if (recv_V[i] > max)
        {
            max = recv_V[i];
        }
    }

    printf("Process %d local max: %d\n", rank, max);
    fflush(stdout);

    int max_vec[n];

    MPI_Allgather(&max, 1, MPI_INT, &max_vec, 1, MPI_INT, row_comm);

    if (row_rank == 0)
    {
        printf("Vettore con i massimi per la riga %d: ", coords[0]);
        print_vector(max_vec, n);
    }

    printf("\n");
    MPI_Finalize();
    return 0;
}