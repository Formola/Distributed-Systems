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

    MPI_Init(&argc, &argv);
    int rank, size;
    int dim = DIM;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = sqrt(size);
    if (n * n != size)
    {
        if (rank == 0)
        {
            perror("Number of processes must be a perfect square");
        }
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

    int k = DIM; // k = 4

    int A[k][k]; // ognuno avrà questa matrice, i proc in prima colonna la leggono e la mandano agli altri sulla propria riga

    if (coords[1] == 0)
    {

        char *filenames[] = {"input1.txt", "input2.txt", "input3.txt"};
        int my_file_index = coords[0];
        FILE *fp = fopen(filenames[my_file_index], "r");
        if (fp == NULL)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // ognuno legge da file una matrice k x k
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < k; j++)
            {
                fscanf(fp, "%d", &A[i][j]);
            }
        }
    }

    MPI_Comm row_comm;
    int remain_dims[2] = {0, 1}; // 0 collassa 1 mantiene, crea un comm per le dim collassate -> un comm per ogni riga.
    int row_rank;
    MPI_Cart_sub(topology, remain_dims, &row_comm);
    MPI_Comm_rank(row_comm, &row_rank);

    // il processo sulla prima colonna ora manda la matrice agli altri sulla propria riga

    MPI_Bcast(A, k * k, MPI_INT, 0, row_comm);

    printf("\nProcess (%d,%d) received matrix:\n", coords[0], coords[1]);
    print_matrix(&A[0][0], k, k);
    fflush(stdout);

    // ogni processo (i,j) calcola il minimo della diagonale principale della matrice A, moltiplicata per il valore j+1.
    int min = A[0][0] * (coords[1] + 1); // j+1
    for (int i = 0; i < k; i++)
    {

        int current_value = A[i][i] * (coords[1] + 1);

        if (current_value < min)
        {
            min = current_value;
        }
    }
    // printf("Process (%d,%d) has minimum value: %d\n", coords[0], coords[1], min);

    // ognuno ha un valore min, bisogna che si calcoli una media visibile a tutti e solo chi ha min < media stampa il proprio min

    int all_min[size];

    MPI_Allgather(&min, 1, MPI_INT, all_min, 1, MPI_INT, topology);

    // tutti vedono all_min, si calcola una media
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += all_min[i];
    }
    float media = (float)sum / size;

    if (min < media)
    {
        printf("Process (%d,%d) has minimum value %d which is less than the average %f\n", coords[0], coords[1], min, media);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
