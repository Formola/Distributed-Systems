// FATTA SU CARTA !!

/*
Realizzare in mpi C un'applicazione in cui:
- il processo di rango 0 legge da file una matrice di interi A[DIM][DIM]
  e ne distribuisce a tutti i processi compreso se stesso blocchi di k righe consecutive in modalità round-robin.
  DIM si suppone multiplo intero p di k * nproc.

- il singolo processo ordina in senso crescente le righe della
  propria matrice V, in base agli elementi che si trovano in prima colonna.


- il processo che presenta il valore max in V[0][0] dovrà
  inviare a tutti i processi la sua prima riga di V.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "matrix.txt"
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

    int V[blocks_per_proc * k][dim];

    if (rank == 0)
    {

        for (int p = 0; p < size; p++)
        {
            MPI_Send(&A[p * k][0], 1, blocks_type, p, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Recv(&V[0][0], blocks_per_proc * k * dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("\nProcess %d received matrix V:\n", rank);
    print_matrix((int *)V, blocks_per_proc * k, dim);
    fflush(stdout);

    for (int i = 0; i < (blocks_per_proc * k) - 1; i++)
    {
        for (int j = i + 1; j < (blocks_per_proc * k); j++)
        {
            if (V[i][0] > V[j][0])
            {
                for (int col = 0; col < dim; col++)
                {
                    int temp = V[i][col];
                    V[i][col] = V[j][col];
                    V[j][col] = temp;
                }
            }
        }
    }

    printf("\nProcess %d sorted matrix V:\n", rank);
    print_matrix((int *)V, blocks_per_proc * k, dim);
    fflush(stdout);

    int recv_V[dim];

    typedef struct
    {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    local_max.value = V[0][0];
    local_max.rank = rank;

    MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (rank == global_max.rank)
    {
        for (int i = 0; i < dim; i++)
        {
            recv_V[i] = V[0][i];
        }
    }

    MPI_Bcast(recv_V, dim, MPI_INT, global_max.rank, MPI_COMM_WORLD);

    printf("\nProcess %d received the max row from process %d: ", rank, global_max.rank);
    print_vector(recv_V, dim);
    fflush(stdout);

    printf("\n");
    MPI_Type_free(&blocks_type);
    MPI_Finalize();
    return 0;
}
