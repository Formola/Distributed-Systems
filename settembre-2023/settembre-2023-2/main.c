/*
Realizzare in mpi C un'applicazione con numero di processi nproc = n*n che effettui:

- realizzare una topologia bidimensionale n*n.

- ciascun processo genera un num casuale minore di 100

- i processi che, nella propria colonna, hanno estratto
  il numero minore, effettuano il prodotto degli interi
  della propria colonna, e inviano il risultato
  all elemento (0,0) che effettua la somma globale.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

void print_vector(int *v, int dim)
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

    int n = (int)sqrt(size);

    if (n * n != size)
    {
        printf("Number of processes must be a perfect square\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {n, n};
    int periods[2] = {1, 1};
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &topology);
    MPI_Cart_coords(topology, rank, ndims, coords);

    printf("Process %d is at position (%d, %d)\n", rank, coords[0], coords[1]);
    fflush(stdout);

    srand(time(NULL) + rank);
    int my_num = rand() % 100;
    printf("Process %d generated number %d\n", rank, my_num);
    fflush(stdout);

    // processi che nella propria colonna hanno estratto il numero minore

    MPI_Comm col_comm;
    int remain_dims[2] = {1, 0}; // 1 means keep, 0 collapse, and create comm for each collapsed (column).
    MPI_Cart_sub(topology, remain_dims, &col_comm);
    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    // find min per ogni colonna

    typedef struct
    {
        int value;
        int rank;
    } minloc_t;

    minloc_t local_min, col_min;

    local_min.value = my_num;
    local_min.rank = col_rank;

    MPI_Allreduce(&local_min, &col_min, 1, MPI_2INT, MPI_MINLOC, col_comm);

    // proc che ha il min sulla sua colonna deve raccogliere tutti i val della sua colonna
    int col_min_values[n];
    MPI_Gather(&my_num, 1, MPI_INT, col_min_values, 1, MPI_INT, col_min.rank, col_comm);

    if (col_rank == col_min.rank)
    {

        // chi ha raccolto calcola il prodotto dei valori della sua colonna
        int prod_col = col_min_values[0];
        for (int i = 1; i < n; i++)
        {
            prod_col *= col_min_values[i];
        }

        printf("\nProcess %d in column %d has product %d\n", rank, coords[1], prod_col);
        fflush(stdout);

        int dest;
        int dest_coords[2] = {0, 0};
        MPI_Cart_rank(topology, dest_coords, &dest);
        // potevo pure passare dest = 0 direttamente.

        MPI_Send(&prod_col, 1, MPI_INT, dest, 0, topology);
    }

    // oppure rank == 0
    int global_sum = 0;
    int recv_prod;
    if (rank == 0)
    {

        for (int i = 0; i < n; i++)
        {

            MPI_Recv(&recv_prod, 1, MPI_INT, MPI_ANY_SOURCE, 0, topology, MPI_STATUS_IGNORE);
            global_sum += recv_prod;
        }

        printf("\nProc %d (%d,%d) Global sum is %d\n", rank, coords[0], coords[1], global_sum);
        fflush(stdout);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}