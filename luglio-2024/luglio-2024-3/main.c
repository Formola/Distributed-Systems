/*
Realizzare in mpi C un'applicazione con numero di processi nproc = nxn in cui:

- realizzare una topologia bidimensionale n*n.

- i processi che nella topologia creata si trovano
  sulla prima colonna leggono, ciascuno da un
  file diverso, gli elementi di una matrice di
  interi A[k*k] e la inviano ai processi della
  loro stessa riga.

- ogni processo (i,j) calcola il minimo della
  diagonale principale della matrice A, moltiplcata
  per il valore j+1.

- solo i processi con un valore minore della media
  lo visualizzano a video

*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define DIM 4

void print_matrix(int *mat, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            printf("%d ", mat[i * dim + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dim = DIM;

    int n = (int)sqrt(size);
    if (n * n != size)
    {
        perror("Number of processes must be a perfect square\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {n, n};
    int periods[2] = {1, 1};
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &topology);
    MPI_Cart_coords(topology, rank, ndims, coords);

    printf("Rank %d has coordinates (%d,%d)\n", rank, coords[0], coords[1]);
    fflush(stdout);

    // i processi nella prima colonna della topologia leggono ognuno una matrice da un file diverso
    // e la inviano ai processi della loro stessa riga

    MPI_Comm row_comm;
    int row_rank;
    int remain_dims[2] = {0, 1};

    MPI_Cart_sub(topology, remain_dims, &row_comm);
    MPI_Comm_rank(row_comm, &row_rank);

    int A[dim][dim];

    if (coords[1] == 0)
    {

        char *filenames[] = {"input1.txt", "input2.txt", "input3.txt"};

        FILE *fp = fopen(filenames[coords[0]], "r");
        if (fp == NULL)
        {
            perror("Error opening file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // legge da file
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                fscanf(fp, "%d", &A[i][j]);
            }
        }

        fclose(fp);

        printf("Process %d read matrix from file %s:\n", rank, filenames[coords[0]]);
        print_matrix(&A[0][0], dim);
        fflush(stdout);
    }

    int source_row_coords[2] = {0, 0};
    int source_row_rank;
    MPI_Cart_rank(row_comm, source_row_coords, &source_row_rank);

    printf("source_row_rank for process %d is %d\n", rank, source_row_rank);
    fflush(stdout);

    // il processo sulla prima colonna invia A ai processi sulla sua riga
    MPI_Bcast(&A[0][0], dim*dim, MPI_INT, source_row_rank, row_comm);

    if (row_rank != source_row_rank)
    {
        printf("Process %d received matrix:\n", rank);
        print_matrix(&A[0][0], dim);
        fflush(stdout);
    }

    // ogni processo (i,j) calcola il minimo della diagonale principale della matrice A,
    // moltiplcata per il valore j+1.
    int local_min = INT32_MAX;
    for (int i = 0; i < dim; i++) {
        int current_diag_val = A[i][i] * (coords[1] + 1);
        if (local_min > current_diag_val) {
            local_min = current_diag_val;
        }
    }

    printf("Process %d with coords (%d,%d) has local_min %d\n", rank, coords[0], coords[1], local_min);
    fflush(stdout);

    // calcolo della media dei minimi
    int global_sum;
    MPI_Allreduce(&local_min, &global_sum, 1, MPI_INT, MPI_SUM, topology);

    float global_avg = (float)global_sum / size;

    if (local_min < global_avg) {
        printf("Process %d with coords (%d,%d) has local_min %d which is less than global_avg %.2f\n", rank, coords[0], coords[1], local_min, global_avg);
        fflush(stdout);
    }

    printf("\n");
    MPI_Finalize();
    return 0;
}
