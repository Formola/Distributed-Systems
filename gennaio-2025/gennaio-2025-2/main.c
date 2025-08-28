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

int main(int argc, char *argv[])
{
    int rank, size;
    int dim = DIM;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *A = NULL;

    int k = 2;

    if (dim % (k * size) != 0)
    {
        if (rank == 0)
        {
            printf("Incompatible matrix dimensions and process configuration.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int num_blocks = DIM / k;

    int num_blocks_per_proc = num_blocks / size;

    int *V = malloc(num_blocks_per_proc * k * dim * sizeof(int));

    MPI_Datatype block_k_rows;
    MPI_Type_vector(k, dim, dim, MPI_INT, &block_k_rows);
    MPI_Type_commit(&block_k_rows);

    MPI_Datatype blocks_k_rows;
    MPI_Type_vector(num_blocks_per_proc, k * dim, size * k * dim, MPI_INT, &blocks_k_rows);
    MPI_Type_commit(&blocks_k_rows);

    if (rank == 0)
    {

        FILE *fp = fopen(FILE_NAME, "r");

        if (fp == NULL)
        {
            printf("Error opening file %s\n", FILE_NAME);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        A = malloc(dim * dim * sizeof(int));
        // reading matrix from file
        for (int i = 0; i < dim * dim; i++)
        {
            fscanf(fp, "%d", &A[i]);
        }

        printf("Matrix A read from file:\n");
        print_matrix(A, dim, dim);
        printf("\n");
        fflush(stdout);

        // esempio blocchi
        // 0 1 2 3 4 5 6 7 8 9 10 11
        // proc0 -> 0 1 -> 1 blocco da k righe
        // proc1 -> 2 3
        // proc2 -> 4 5
        // proc0 -> 6 7 -> 1 blocco da k righe
        // proc1 -> 8 9
        // proc2 -> 10 11

        // proc 0 copia i suoi blocchi, in partire prende i blocchi a 0, size, 2*size, ossia
        // quelli destinati a rank o
        // for (int b = 0; b < num_blocks; b+=size){
        //     int local_index =  b / size;
        //     memcpy(V + local_index * k * dim, A + b*k*dim, k*dim*sizeof(int));
        // }
        // printf("Process %d copied its blocks to V:\n", rank);
        // print_matrix(V, num_blocks_per_proc * k, dim);
        // fflush(stdout);
        // printf("\n");

        // // invia i blochi agli altri
        // for (int b = 1; b < num_blocks; b++){
        //     int dest = b % size;
        //     MPI_Send(&A[b*k*dim], 1, block_k_rows, dest, 0, MPI_COMM_WORLD);
        // }

        // proc0 copia i suoi blocchi
        for (int b = 0; b < num_blocks; b += size)
        {
            int local_index = b / size;
            memcpy(V + local_index * k * dim, A + b * k * dim, k * dim * sizeof(int));
        }

        printf("Process %d copied its blocks to V:\n", rank);
        print_matrix(V, num_blocks_per_proc * k, dim);
        fflush(stdout);
        printf("\n");

        for (int dest = 1; dest < size; dest++)
        {
            int first_block = dest;
            int *start = A + first_block * k * dim;
            MPI_Send(start, 1, blocks_k_rows, dest, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        // for (int b = 0; b < num_blocks_per_proc; b++) {
        //     MPI_Recv(&V[b*k*dim], 1, block_k_rows, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // }

        // printf("Process %d received matrix V:\n", rank);
        // print_matrix(V, num_blocks_per_proc * k, dim);
        // printf("\n");
        // fflush(stdout);

        MPI_Recv(V, num_blocks_per_proc * k * dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process %d received matrix V:\n", rank);
        print_matrix(V, num_blocks_per_proc * k, dim);
        fflush(stdout);
    }

    // ognuno ordina le righe di V in base ai valori sulla prima colonna

    for (int i = 0; i < (num_blocks_per_proc * k) - 1; i++)
    {
        for (int j = i + 1; j < (num_blocks_per_proc * k); j++)
        {

            if (V[i * dim] > V[j * dim])
            {

                // swamp full rows
                for (int k = 0; k < dim; k++)
                {
                    int temp = V[i * dim + k];
                    V[i * dim + k] = V[j * dim + k];
                    V[j * dim + k] = temp;
                }
            }
        }
    }

    printf("Process %d sorted matrix V:\n", rank);
    print_matrix(V, num_blocks_per_proc * k, dim);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    maxloc_t local_max, global_max;

    local_max.rank = rank;
    local_max.value = V[0];

    MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

    printf("Process %d has the maximum value %d at rank %d\n", rank, global_max.value, global_max.rank);

    int *first_row = malloc(dim * sizeof(int));

    if (rank == global_max.rank)
    {
        for (int j = 0; j < dim; j++)
        {
            first_row[j] = V[j];
        }
    }

    MPI_Bcast(first_row, dim, MPI_INT, global_max.rank, MPI_COMM_WORLD);

    printf("Process %d received first row of V:\n", rank);
    print_matrix(first_row, 1, dim);
    fflush(stdout);

    MPI_Type_free(&block_k_rows);
    MPI_Type_free(&blocks_k_rows);
    MPI_Finalize();
    return 0;
}
