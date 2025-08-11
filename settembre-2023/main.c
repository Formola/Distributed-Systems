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

    int n = (int)sqrt(size);

    // simple check on nproc and n.
    if (n * n != size)
    {
        perror("Number of process must be a perfect square");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm topology;
    int topology_rank;

    int nDims = 2;           // topologia bidimensionale
    int dims[2] = {n, n};    // num of proc for each dimension.
    int periods[2] = {1, 1}; // 1 is true, means periodicity on that dimension
    int reorder = 0;         // 0 is false, means no reordering of ranks

    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, nDims, dims, periods, reorder, &topology);
    MPI_Comm_rank(topology, &topology_rank);

    MPI_Cart_coords(topology, topology_rank, nDims, coords);

    printf("Process %d is at coordinates (%d, %d)\n", topology_rank, coords[0], coords[1]);
    fflush(stdout);

    // ogni proc genera un num casuale minore di 100
    srand(time(NULL) + rank);
    int value = rand() % 100;
    printf("Process %d generated value %d\n", topology_rank, value);
    fflush(stdout);

    // creazione communicator per colonne
    MPI_Comm col_comm;
    MPI_Comm_split(topology, coords[1], 0, &col_comm);

    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);

    printf("Process %d is in column communicator %d\n", topology_rank, coords[1]);
    fflush(stdout);

    int min_value;
    MPI_Allreduce(&value, &min_value, 1, MPI_INT, MPI_MIN, col_comm);

    int *col_values = NULL;
    MPI_Allgather(&value, 1, MPI_INT, col_values, 1, MPI_INT, col_comm);

    // se non vuoi fare allgather cosi che tutti hanno i valori di una colonna devi fare sta schifezza qua
    // calcoli il root e poi devi fare quella allreduce in cui aggiorni il root a tutti, altrimenti
    // se arrivato a gather con valori diversi in root (partono da -1) non funziona
    // int root = -1;
    // if (value == min_value)
    // {
    //     col_values = malloc(n * sizeof(int));
    //     root = col_rank;
    // }

    // MPI_Allreduce(MPI_IN_PLACE, &root, 1, MPI_INT, MPI_MAX, col_comm);
    // MPI_Gather(&value, 1, MPI_INT, col_values, 1, MPI_INT, root, col_comm);

    if (value == min_value)
    {
        int product = 1;
        for (int i = 0; i < n; i++)
        {
            product *= col_values[i];
        }

        int dest_rank;
        int dest_coords[2] = {0, 0};
        MPI_Cart_rank(topology, dest_coords, &dest_rank);

        MPI_Send(&product, 1, MPI_INT, dest_rank, 0, topology);
    }

    int rcv_product;
    if (topology_rank == 0)
    {
        int global_sum = 0;
        MPI_Status status;
        // ci sono n colonne quindi ci saranno n Send, ci aspettiamo n rcv dal processo (0,0)
        for (int i = 0; i < n; i++)
        {
            int rcv_product;
            MPI_Recv(&rcv_product, 1, MPI_INT, MPI_ANY_SOURCE, 0, topology, &status);
            global_sum += rcv_product;
            printf("Process 0 received product %d from process %d\n", rcv_product, status.MPI_SOURCE);
        }
        printf("Process 0 total sum = %d\n", global_sum);
    }

    MPI_Finalize();
    return 0;
}