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

    if (n * n != size){
        perror("Il numero di processi deve essere un quadrato perfetto\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {n,n};
    int periods[2] = {1,1};
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &topology);
    MPI_Cart_coords(topology, rank, ndims, coords);

    printf("Rank %d ha coordinate (%d,%d)\n", rank, coords[0], coords[1]);
    fflush(stdout);

    srand(time(NULL) + rank);
    int value = rand() % 100;

    printf("Rank %d ha estratto il numero %d\n", rank, value);
    fflush(stdout);

    // i processi che, nella propria riga, hanno estratto il numero maggiore effettuano
    // la somma degli interi della propria riga e inviano la somma
    // all'elemento della propria riga posizionato
    // sulla diagonale principale della topologia.

    MPI_Comm row_comm;
    int remain_dims[2] = {0,1};
    MPI_Cart_sub(topology, remain_dims, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    typedef struct {
        int value;
        int row_rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    int row_values[row_size];

    local_max.value = value;
    local_max.row_rank = row_rank;

    MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, row_comm);


    MPI_Gather(&value, 1, MPI_INT, row_values, 1, MPI_INT, global_max.row_rank, row_comm);
    
    int sum = 0; 
    if (row_rank == global_max.row_rank) {
        for (int i = 0; i < row_size; i++) {
            sum += row_values[i];
        }

        printf("\nRank %d (max della riga %d) ha la somma %d\n", rank, coords[0], sum);
        fflush(stdout);

        int diag_rank;
        int diag_coords[2] = {coords[0], coords[0]};

        MPI_Cart_rank(topology, diag_coords, &diag_rank);

        MPI_Send(&sum, 1, MPI_INT, diag_rank, 0, topology);
    }

    MPI_Barrier(topology);

     // i processi sulla diagonale principale ricevono la somma
     // e la stampano a video

    if (coords[0] == coords[1]) {

        int recv_sum;
        MPI_Recv(&recv_sum, 1, MPI_INT, MPI_ANY_SOURCE, 0, topology, MPI_STATUS_IGNORE);
        printf("\nRank %d (diagonale) ha ricevuto la somma %d dalla riga %d\n", rank, recv_sum, coords[0]);
        fflush(stdout);
    }

    printf("\n");
    MPI_Finalize();
    return 0;
}