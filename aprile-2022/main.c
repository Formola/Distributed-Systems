/*
Realizzare in mpi C un'applicazione in cui:

- il processo di rango 0 legge da file una matrice di interi
  A[DIMXDIM] con DIM multiplo intero k di nproc, e ne distribuisce a
  tutti i processi compreso se stesso le singole colonne in modalità round-robin.

- il singolo processo, data T[DIM x k],
  calcola un vettore V[DIM] prendendo
  il valore max di ciascuna riga

- solo i 6 processi con i valori più grandi
  in V[0] realizzano una topologia 2x3.

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "matrix.txt"

typedef struct
{
    int value;
    int rank;
} maxloc_t;

void print_matrix(int *A, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d ", A[i * col + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim;
    int *A = NULL;

    if (rank == 0)
    {

        FILE *fp = fopen(FILE_NAME, "r");

        if (fp == NULL)
        {
            printf("Error opening file %s\n", FILE_NAME);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // leggi dim da prima riga file
        fscanf(fp, "%d", &dim);

        if (dim % size != 0)
        {
            printf("Dimensione matrice deve essere multiplo di %d\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // leggere matrice dal file
        A = malloc(dim * dim * sizeof(int));

        for (int i = 0; i < dim * dim; i++)
        {
            fscanf(fp, "%d", &A[i]);
        }

        print_matrix(A, dim, dim);
    }

    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("\n");

    int k = dim / size; // numero di colonne per ogni processo

    int *T = malloc(dim * k * sizeof(int));

    // distribuire le colonne in round robin. serve datatype

    // num blocchi = dim ossia dim righe, elementi x blocco = 1, stride = dim distanza tra due blocchi
    // tra un elemento della colonna e quello sotto nella stessa colonna ci sono dim elementi
    MPI_Datatype column_type, recv_column_type;
    MPI_Type_vector(dim, 1, dim, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);

    // dim blocchi, 1 elemento per blocco, stride = k,
    MPI_Type_vector(dim, 1, k, MPI_INT, &recv_column_type);
    MPI_Type_commit(&recv_column_type);

    if (rank == 0)
    {
        for (int p = 0; p < dim; p++)
        {
            MPI_Send(&A[p], 1, column_type, p % size, 0, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < k; i++)
    {
        MPI_Recv(&T[i], 1, recv_column_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("Processo %d, colonne ricevute:\n", rank);
    print_matrix(T, dim, k);
    // fflush(stdout);

    int *my_v = malloc(dim * sizeof(int));

    // calcolo max su ogni riga e salvo in v
    for (int i = 0; i < dim; i++)
    {
        my_v[i] = T[i * k];
        for (int j = 1; j < k; j++)
        {
            if (T[i * k + j] > my_v[i])
            {
                my_v[i] = T[i * k + j];
            }
        }
    }
    printf("Processo %d, vettore max:\n", rank);
    for (int i = 0; i < dim; i++)
    {
        printf("%d ", my_v[i]);
    }

    maxloc_t *ranks = malloc(6 * sizeof(maxloc_t));

    maxloc_t my_max;
    my_max.value = my_v[0];
    my_max.rank = rank;

    // printf("Processo %d, valore max: %d\n", my_max.rank, my_max.value);

    for (int i = 0; i < 6; i++)
    {

        MPI_Allreduce(&my_max, &ranks[i], 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (ranks[i].rank == my_max.rank)
        {
            my_max.value = INT32_MIN; // escludo il massimo trovato
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        printf("Processo %d, valori massimi: ", ranks[0].rank);
        for (int i = 0; i < 6; i++)
        {
            printf("%d ", ranks[i].value);
        }
        printf("\n");
    }

    // creazione top 2x3.

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {2, 3};
    int periods[2] = {0, 0};

    MPI_Group world_group, max_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // creare gruppo con i 6 processi dei massimi
    int ranks_of_max[6];
    for (int i = 0; i < 6; i++)
        ranks_of_max[i] = ranks[i].rank;

    MPI_Group_incl(world_group, 6, ranks_of_max, &max_group);

    MPI_Comm max_comm;
    MPI_Comm_create(MPI_COMM_WORLD, max_group, &max_comm);

    if (max_comm != MPI_COMM_NULL)
    {
        MPI_Cart_create(max_comm, ndims, dims, periods, 0, &topology);

        int coords[2];
        int my_rank_in_topo;
        MPI_Comm_rank(topology, &my_rank_in_topo);
        MPI_Cart_coords(topology, my_rank_in_topo, ndims, coords);

        printf("Processo %d, coordinate nella topologia: (%d, %d)\n",
               rank, coords[0], coords[1]);
    }

    printf("\n");
    MPI_Type_free(&column_type);
    MPI_Type_free(&recv_column_type);

    MPI_Finalize();
    return 0;
}