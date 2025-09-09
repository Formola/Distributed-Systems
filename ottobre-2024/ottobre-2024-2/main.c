/*
Realizzare in mpi C un'applicazione in cui:

- il processo di rango 0 legge da file una matrice di interi A[DIM][DIM]
  e ne distribuisce a tutti i processi compreso se stesso blocchi di k righe consecutive
  in modalità round-robin.
  DIM si suppone multiplo di k*nproc.

- il singolo processo ordina in senso crescente le righe della propria matrice V,
  in base agli elementi che si trovano nella prima colonna.

- la riga che presenta il valore max in V[0][0] dovrà
  essere inviata a tutti i processi.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "matrix.txt"
#define DIM 12

void print_vector(int *v, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("%d ", v[i]);
    }
    printf("\n");
}

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

    int A[dim][dim];

    if (rank == 0){
        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < dim*dim; i++) {
            for (int j = 0; j < dim; j++) {
                fscanf(fp, "%d", &A[i][j]);
            }
        }
        fclose(fp);

        printf("Matrix A read from file:\n");
        print_matrix(&A[0][0], dim, dim);
        fflush(stdout);
    }

    int k = 2;
    
    if (dim % (k*size) != 0){
        perror("Error: DIM must be multiple of k*nproc");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int blocks_per_proc = dim / (k*size);

    int local_rows = blocks_per_proc * k;

    int V[local_rows][dim];

    MPI_Datatype rows_type;
    MPI_Type_vector(blocks_per_proc, dim*k, dim*size*k, MPI_INT, &rows_type);
    MPI_Type_commit(&rows_type);

    if (rank == 0) {
        for (int p = 0; p < size; p++) {
            MPI_Send(&A[p*k][0], 1, rows_type, p, 0, MPI_COMM_WORLD);
        }
    }

    // tutti compreso p0 ricevono

    MPI_Recv(&V[0][0], local_rows*dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("\nProcess %d received matrix V:\n", rank);
    print_matrix(&V[0][0], local_rows, dim);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    // il singolo processo ordina in senso crescente le righe della propria matrice V,
    // in base agli elementi che si trovano nella prima colonna.

    for (int i = 0; i < local_rows-1; i++) {
        for (int j = i+1; j < local_rows; j++) {

            if (V[i][0] > V[j][0]) {

                // swap rows
                for (int k = 0; k < dim; k++) {
                    int temp = V[i][k];
                    V[i][k] = V[j][k];
                    V[j][k] = temp;
                }
            }
        }
    }

    printf("\nProcess %d sorted matrix V:\n", rank);
    print_matrix(&V[0][0], local_rows, dim);
    fflush(stdout);

    // la riga che presenta il valore max in V[0][0] dovrà
    // essere inviata a tutti i processi.

    int max_row[dim];

    typedef struct {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    local_max.value = V[0][0];
    local_max.rank = rank;

    MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

    MPI_Bcast(&V[0], dim, MPI_INT, global_max.rank, MPI_COMM_WORLD);

    printf("\nProcess %d received max row from process %d:\n", rank, global_max.rank);
    print_vector(&V[0][0], dim);
    fflush(stdout);

    printf("\n");
    MPI_Type_free(&rows_type);
    MPI_Finalize();
    return 0;
}