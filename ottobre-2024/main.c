/*
Realizzare in mpi C un'applicazione in cui:

- il processo di rango 0 legge da fule una matrice di interi A[DIM][DIM]
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

int *read_matrix_file(const char *filename, int *dim)
{

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("error opening file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // supponendo il primo elemento da leggere sia la dim della matrice
    // keep it simple, just read
    if (fscanf(file, "%d", dim) != 1)
    {
        fprintf(stderr, "Error reading matrix dimension\n");
        fclose(file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *matrix = (int *)malloc((*dim) * (*dim) * sizeof(int));

    // adesso leggiamo la matrice dal file

    for (int i = 0; i < *dim; i++)
    {

        for (int j = 0; j < *dim; j++)
        {

            if (fscanf(file, "%d", &matrix[i * (*dim) + j]) != 1)
            {

                fprintf(stderr, "Error reading matrix element at [%d][%d]\n", i, j);
                free(matrix);
                fclose(file);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    return matrix;
}

void print_matrix(int *matrix, int row, int col)
{

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d ", matrix[i * col + j]);
        }
        printf("\n");
    }
}

void swap_rows(int *matrix, int row1, int row2, int cols)
{
    for (int j = 0; j < cols; j++)
    {

        int temp = matrix[row1 * cols + j];
        matrix[row1 * cols + j] = matrix[row2 * cols + j];
        matrix[row2 * cols + j] = temp;
    }
}

void sort_rows(int *matrix, int rows, int cols)
{

    for (int i = 0; i < rows - 1; i++)
    {
        for (int j = i + 1; j < rows; j++)
        {

            if (matrix[i * cols] > matrix[j * cols])
            {

                swap_rows(matrix, i, j, cols);
            }
        }
    }
}


int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim;
    int *matrix = NULL;

    int k = 2;

    if (rank == 0)
    {
        // Read the matrix from file
        matrix = read_matrix_file(FILE_NAME, &dim);
        printf("Matrix read from file (dimension %d):\n", dim);

        if (dim % (k * size) != 0)
        {
            fprintf(stderr, "Matrix dimension must be divisible by number of processes\n");
            free(matrix);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        print_matrix(matrix, dim, dim);
        printf("\n");
    }

    // Broadcast the dimension of the matrix to all processes
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // brodcast k row to all processes

    int *my_rows = malloc(k * dim * sizeof(int));

    MPI_Scatter(matrix, k * dim, MPI_INT, my_rows, k * dim, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received rows:\n", rank);
    print_matrix(my_rows, k, dim);
    printf("\n");

    // Sort the rows of the local matrix
    sort_rows(my_rows, k, dim);

    printf("Process %d sorted rows:\n", rank);
    print_matrix(my_rows, k, dim);
    printf("\n");
    // Find the maximum value in the first column of the local matrix

    int my_val = my_rows[0 * dim + 0];
    int global_max;

    // chiamata collettiva sincrona, non serve una barrier dopo il sorting dato
    // che tutti i processi chiamano questa funzione e quindi hanno già finito di ordinare le loro righe
    MPI_Allreduce(&my_val, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    printf("Process %d knows max value in first column (considering all processes): %d\n", rank, global_max);

    int *max_row = malloc(dim * sizeof(int));

    int my_rank_global_max = (my_val == global_max) ? rank : -1;
    int global_max_rank;

    MPI_Allreduce(&my_rank_global_max, &global_max_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (rank == global_max_rank)
    {
        // Copy the row with the maximum value to max_row
        for (int j = 0; j < dim; j++)
        {
            max_row[j] = my_rows[0 * dim + j];
        }
        printf("\n");
    }

    // Broadcast the row with the maximum value to all processes
    MPI_Bcast(max_row, dim, MPI_INT, global_max_rank, MPI_COMM_WORLD);

    // Print the row with the maximum value
    printf("Process %d received the row with the maximum value:\n", rank);
    print_matrix(max_row, 1, dim);

    if (rank == 0)
    {
        free(matrix);
    }
    free(my_rows);
    free(max_row);

    printf("\n");
    MPI_Finalize();
    return 0;
}