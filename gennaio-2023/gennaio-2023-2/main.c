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
        if (rank == 0)
        {
            fprintf(stderr, "Number of processes must be a perfect square.\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {n, n};    // proc per dim
    int periods[2] = {1, 1}; // 1 means periodic on that dim -> 0 1 2 -> 0 1 2
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &topology);
    MPI_Cart_coords(topology, rank, ndims, coords);

    printf("Process %d coordinates: (%d, %d)\n", rank, coords[0], coords[1]);
    fflush(stdout);

    // ogni processo genera un intero casuale minore di 100
    srand(time(NULL) + rank);
    int number = rand() % 100;
    printf("Process %d generated number: %d\n", rank, number);
    fflush(stdout);

    // i processi che nella propria riga hanno estratto il numero maggiore
    // devono effettuare la somma degli interi sulla propria riga
    // dopodiche inviano la somma al processo sulla propria riga posizionato sulla diagonale principale della topologia.

    MPI_Comm row_comm;
    int remain_dims[2] = {0, 1};
    int row_rank;
    MPI_Cart_sub(topology, remain_dims, &row_comm);
    MPI_Comm_rank(row_comm, &row_rank);

    typedef struct
    {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t my_max, row_max;
    my_max.value = number;
    my_max.rank = row_rank;

    MPI_Allreduce(&my_max, &row_max, 1, MPI_2INT, MPI_MAXLOC, row_comm);

    printf("Process %d (row-rank %d) has max %d from process (row-rank %d)\n",
           rank, row_rank, row_max.value, row_max.rank);
    fflush(stdout);

    // il processo che ha il max su una riga effettua la somma degli interi sulla sua riga
    int row_values[n];
    MPI_Gather(&number, 1, MPI_INT, row_values, 1, MPI_INT, row_max.rank, row_comm);

    int recv_sum;

    if (row_rank == row_max.rank)
    {
        int row_sum = 0;
        for (int i = 0; i < n; i++){
            row_sum += row_values[i];
        }
        printf("Process %d (row-rank %d) has row sum %d\n",
               rank, row_rank, row_sum);


        // inviare la somma all'elemento della propria riga che si trova sulla diag della topologia;
        int diag_coords[2] = {coords[0], coords[0]};
        int dest;
        MPI_Cart_rank(topology, diag_coords, &dest);
        MPI_Send(&row_sum, 1, MPI_INT, dest, 0, topology);
    }

    // i processi sulla diagonale ricevono
    if (coords[0] == coords[1]) {

        MPI_Recv(&recv_sum, 1, MPI_INT, MPI_ANY_SOURCE, 0, topology, MPI_STATUS_IGNORE);
        printf("Process %d (row-rank %d) received row sum %d\n",
               rank, row_rank, recv_sum);
    }

    printf("\n");
    MPI_Finalize();
    return 0;
}