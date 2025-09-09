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
#define DIM 18

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
    printf("\n");
}

void print_vector(int *V, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("%d ", V[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = DIM;

    int k = dim / size; // numero di colonne per processo

    int A[dim][dim];

    if (rank == 0)
    {
        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL)
        {
            perror("Errore nell'apertura del file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                fscanf(fp, "%d", &A[i][j]);
            }
        }
        fclose(fp);

        printf("Matrice letta dal file:\n");
        print_matrix((int *)A, dim, dim);
        fflush(stdout);
    }

    int T[dim][k];

    MPI_Datatype col_type;
    MPI_Type_vector(dim, 1, dim, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    // Distribuzione delle colonne in modalità round-robin
    if (rank == 0)
    {

        for (int p = 0; p < size; p++)
        {

            for (int c = 0; c < k; c++)
            {

                MPI_Send(&A[0][p + c * size], 1, col_type, p, 0, MPI_COMM_WORLD);
            }
        }
    }

    // tutti compreso proc0, ricevono ma devono memorizzare in colonna

    MPI_Datatype recv_type;
    MPI_Type_vector(dim, 1, k, MPI_INT, &recv_type);
    MPI_Type_commit(&recv_type);

    for (int c = 0; c < k; c++)
    {
        MPI_Recv(&T[0][c], 1, recv_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("Processo %d, matrice T ricevuta:\n", rank);
    print_matrix((int *)T, dim, k);
    fflush(stdout);

    // il singolo processo, data T[DIM x k],
    // calcola un vettore V[DIM] prendendo
    // il valore max di ciascuna riga

    int V[dim];

    for (int i = 0; i < dim; i++)
    {

        V[i] = T[i][0];
        for (int j = 1; j < k; j++)
        {
            if (T[i][j] > V[i])
            {
                V[i] = T[i][j];
            }
        }
    }

    printf("Processo %d, vettore V dei massimi in ogni riga:\n", rank);
    print_vector(V, dim);
    fflush(stdout);

    // solo i 6 processi con i valori più grandi
    // in V[0] realizzano una topologia 2x3.

    int ranks[6];

    int temp_V[dim];
    for (int i = 0; i < dim; i++)
    {
        temp_V[i] = V[i];
    }

    typedef struct
    {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    for (int i = 0; i < 6; i++)
    {

        local_max.value = temp_V[0];
        local_max.rank = rank;

        MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        ranks[i] = global_max.rank;

        // azzero il massimo trovato in questo giro
        if (global_max.rank == rank)
        {
            temp_V[0] = INT32_MIN;
        }
    }

    if (rank == 0)
    {
        printf("I 6 processi con i valori più grandi in V[0] sono: ");
        print_vector(ranks, 6);
        fflush(stdout);
    }

    MPI_Group world_group, new_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, 6, ranks, &new_group);
    MPI_Comm new_comm;
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {2, 3};
    int periods[2] = {1, 1};
    int coords[2];

    if (new_comm != MPI_COMM_NULL)
    {
        int new_rank;
        MPI_Cart_create(new_comm, ndims, dims, periods, 0, &topology);
        MPI_Comm_rank(topology, &new_rank);
        MPI_Cart_coords(topology, new_rank, ndims, coords);

        printf("Processo %d, coordiante nella topologia 2x3: (%d, %d)\n", rank, coords[0], coords[1]);
        fflush(stdout);
    }

    printf("\n");
    MPI_Type_free(&col_type);
    MPI_Type_free(&recv_type);
    MPI_Finalize();
    return 0;
}