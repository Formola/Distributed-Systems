// FATTA SU CARTA !!

/*
Realizzare in mpi C un'applicazione in cui:
- il processo di rango 0 legge da file una matrice di interi A[DIM][DIM]
  e ne distribuisce a tutti i processi compreso se stesso blocchi di k righe consecutive in modalità round-robin.
  DIM si suppone multiplo intero p di k * nproc.

- per ogni riga della matrice V, viene individuato il processo che presenta in prima colonna
  il valore max.

- i processi che hanno vinto almeno un turno costituiscono un nuovo gruppo
  e calcolano la somma delle proprie matrici V.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "input.txt"
#define DIM 12

typedef struct
{

    int value;
    int rank;
} maxloc_t;

void print_matrix(int *mat, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d ", mat[i * col + j]);
        }
        printf("\n");
    }
}

void print_vector(int *vec, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int rank, size;
    int dim = DIM;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int A[dim][dim];

    if (rank == 0)
    {
        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL)
        {
            printf("Error opening file %s\n", FILE_NAME);
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

        printf("Matrix A read from file:\n");
        print_matrix((int *)A, dim, dim);
        fflush(stdout);
    }

    int k = 2;

    if (dim % (k * size) != 0)
    {
        perror("DIM must be a multiple of k * nproc");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int blocks_per_proc = dim / (k * size);

    MPI_Datatype blocks_type;
    MPI_Type_vector(blocks_per_proc, k * dim, k * dim * size, MPI_INT, &blocks_type);
    MPI_Type_commit(&blocks_type);

    int num_rows = blocks_per_proc * k;
    int V[num_rows][dim];

    if (rank == 0)
    {

        for (int p = 0; p < size; p++)
        {
            MPI_Send(&A[p * k][0], 1, blocks_type, p, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Recv(&V[0][0], num_rows * dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("\nProcess %d received matrix V:\n", rank);
    print_matrix((int *)V, num_rows, dim);
    fflush(stdout);

    int winner_ranks[num_rows];
    maxloc_t local_max, global_max;

    int new_size = 0;

    for (int i = 0; i < num_rows; i++)
    {
        local_max.value = V[i][0];
        local_max.rank = rank;

        MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        winner_ranks[i] = global_max.rank;

        int current = winner_ranks[0];
        for (int j = 1; j < num_rows; j++)
        {
            if (winner_ranks[j] != current)
            {
                new_size++;
                current = winner_ranks[j];
            }
        }
    }

    printf("\nProcess %d winner ranks per row:\n", rank);
    print_vector(winner_ranks, num_rows);
    fflush(stdout);

    printf("\nProcess %d number of unique winners: %d\n", rank, new_size);
    fflush(stdout);

    MPI_Group world_group, new_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, new_size, winner_ranks, &new_group);

    MPI_Comm new_comm;
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    int sum_V[num_rows][dim];

    if (new_comm != MPI_COMM_NULL){
        MPI_Allreduce(&V, &sum_V, num_rows * dim, MPI_INT, MPI_SUM, new_comm);

        printf("\nProcess %d in new group with comm size %d has sum_V:\n", rank, new_size);
        print_matrix((int *)sum_V, num_rows, dim);
        fflush(stdout);
    }

    printf("\n");
    MPI_Type_free(&blocks_type);
    MPI_Finalize();
    return 0;
}
