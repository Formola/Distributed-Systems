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
    int k = dim / size;

    // ogni proc ricevera una matrice T[dim][k]
    // k sono il numero di colonne da distribuire in round robin 1 alla volta a ogni processo da proc0.

    if (dim % (k * size) != 0)
    {
        perror("dim deve essere un multiplo di k * size");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int A[dim][dim];

    if (rank == 0)
    {

        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL)
        {
            perror("Errore apertura file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                fscanf(fp, "%d", &A[i][j]);
            }
        }
        fclose(fp);

        printf("Matrice letta:\n");
        print_matrix(&A[0][0], dim, dim);
        printf("\n");
        fflush(stdout);
    }

    int T[dim][k]; // ognuno deve ricevere k colonne
    // si intende che vengano sia spedite le colonne che memorizzate le colonne

    // single column
    MPI_Datatype column_type;
    MPI_Type_vector(dim, 1, dim, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);

    // all column for a proc, so we send all columns to a proc with a single send
    MPI_Datatype columns_type;
    MPI_Aint stride = size * sizeof(int);
    MPI_Type_hvector(k, 1, stride, column_type, &columns_type);
    MPI_Type_commit(&columns_type);

    if (rank == 0)
    {

        for (int p = 0; p < size; p++)
        {
            MPI_Send(&A[0][p], 1, columns_type, p, 0, MPI_COMM_WORLD);
        }
    }

    // caso con k send e si manda 1 colonna alla volta
    // if (rank == 0)
    // {
    //     for (int p = 0; p < size; p++)
    //     {
    //         for (int j = 0; j < k; j++)
    //         {
    //             MPI_Send(&A[0][p + j * size], 1, column_type, p, 0, MPI_COMM_WORLD);
    //         }
    //     }
    // }

    // tutti, compreso p0, ricevono ma devono salvare lè colonne cosi come sono state inviate
    // serve datatype di recv
    MPI_Datatype col_recv_type;
    MPI_Type_vector(dim, 1, k, MPI_INT, &col_recv_type);
    MPI_Type_commit(&col_recv_type);

    MPI_Datatype recv_type;
    MPI_Type_hvector(k, 1, sizeof(int), col_recv_type, &recv_type);
    MPI_Type_commit(&recv_type);

    // caso con 1 sola recv per tutte le colonne.
    MPI_Recv(&T[0][0], 1, recv_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // caso con 1 solo datatype di recv per colonna e ricevo k colonne
    // for (int j = 0; j < k; j++)
    // {
    //     MPI_Recv(&T[0][j], 1, col_recv_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }

    printf("Matrice ricevuta da proc %d:\n", rank);
    print_matrix(&T[0][0], dim, k);
    fflush(stdout);

    // il singolo processo, dato T[dim][k]
    // si calcola un vettore V[dim]
    // che contiene i max di ogni riga

    int V[dim];

    for (int v = 0; v < dim; v++)
    { // ciclo per agg un elemento a v

        V[v] = T[v][0];
        // ciclo per trovare il max su ogni riga
        for (int i = 1; i < k; i++)
        {
            if (T[v][i] > V[v])
            {
                V[v] = T[v][i];
            }
        }
    }

    printf("Vettore dei massimi in ogni riga:\n");
    print_vector(V, dim);
    fflush(stdout);

    // i 6 processi con valori più grandi in V[0]
    // realizzano una topologia a matrice (2x3).

    int ranks[6];
    int max[6];

    typedef struct
    {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    local_max.value = V[0];
    local_max.rank = rank;

    for (int i = 0; i < 6; i++)
    {

        MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (global_max.value == local_max.value)
        {
            local_max.value = INT32_MIN;
            printf("Nuovo massimo globale trovato da processo %d: %d\n", rank, global_max.value);
            fflush(stdout);
        }

        // devono per forza farla tutti seno non vedrebbero gli stessi ranks
        ranks[i] = global_max.rank;
        max[i] = global_max.value;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Ranghi dei processi con i valori massimi:\n");
        fflush(stdout);
        for (int i = 0; i < 6; i++)
        {
            printf("Processo %d: %d\n", ranks[i], max[i]);
            fflush(stdout);
        }
    }

    // need to create group with top6ranks
    // then create a comm using that group
    // than creating the topology

    MPI_Group top6_group, world_group;
    MPI_Comm top6_comm;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, 6, ranks, &top6_group);
    MPI_Comm_create(MPI_COMM_WORLD, top6_group, &top6_comm);

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {2, 3};
    int periods[2] = {1, 1};
    int coords[2];

    if (top6_comm != MPI_COMM_NULL)
    {
        MPI_Cart_create(top6_comm, ndims, dims, periods, 0, &topology);

        int new_rank;
        MPI_Comm_rank(top6_comm, &new_rank);

        MPI_Cart_coords(topology, new_rank, ndims, coords);

        printf("Coordinate del processo %d nella topologia = (%d, %d) e new_rank = %d\n", rank, coords[0], coords[1], new_rank);
    }

    printf("\n");
    MPI_Type_free(&recv_type);
    MPI_Type_free(&col_recv_type);
    MPI_Type_free(&columns_type);
    MPI_Type_free(&column_type);
    MPI_Finalize();
    return 0;
}