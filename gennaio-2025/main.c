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
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "matrix.txt"

// Reads dim and matrix from file on rank 0
// Returns pointer to allocated matrix, sets *dim to dimension
int *read_matrix_from_file(const char *filename, int *dim)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror("Error opening file");
        return NULL;
    }

    // suppose the first line of the file contains the dimension of the matrix
    if (fscanf(fp, "%d", dim) != 1)
    {
        fprintf(stderr, "Failed to read dimension from file\n");
        fclose(fp);
        return NULL;
    }

    printf("Matrix dimension read from file: %d\n", *dim);
    printf("pointer is %p\n", (void *)dim);

    int *mat = malloc((*dim) * (*dim) * sizeof(int)); // mat[DIM][DIM] flattened to 1D array
    if (!mat)
    {
        perror("malloc failed");
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < *dim; i++)
    {
        for (int j = 0; j < *dim; j++)
        {
            if (fscanf(fp, "%d", &mat[i * (*dim) + j]) != 1) // matrix is stored in flattened form and in row-major order
            // so mat[i][j] is accessed as mat[i * dim + j], i*dim means the row offset, j means the column offset
            {
                fprintf(stderr, "Error reading matrix element [%d][%d]\n", i, j);
                free(mat);
                fclose(fp);
                return NULL;
            }
        }
    }

    fclose(fp);
    return mat;
}

void print_matrix(int *mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_array(int *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void swap_rows(int *mat, int row1, int row2, int cols)
{
    for (int i = 0; i < cols; i++)
    {
        int temp = mat[row1 * cols + i];
        mat[row1 * cols + i] = mat[row2 * cols + i];
        mat[row2 * cols + i] = temp;
    }
}

void sort_rows_by_first_column(int *mat, int rows, int cols)
{
    for (int i = 0; i < rows - 1; i++)
    {
        for (int j = i + 1; j < rows; j++)
        {
            if (mat[i * cols] > mat[j * cols])
            {
                swap_rows(mat, i, j, cols);
            }
        }
    }
}

int find_max_rank(int *max_first_pos, int size)
{
    int max_rank = 0;
    for (int i = 1; i < size; i++)
    {
        if (max_first_pos[i] > max_first_pos[max_rank])
        {
            max_rank = i;
        }
    }
    return max_rank;
}

int main(int argc, char *argv[])
{
    int rank, size;
    int dim;
    int *mat = NULL;
    int k = 2; // Number of rows per block (can be adjusted)

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        mat = read_matrix_from_file(FILE_NAME, &dim);

        if (dim % (k * size) != 0)
        {
            fprintf(stderr, "Error: dim (%d) must be multiple of k (%d) * number of processes (%d)\n", dim, k, size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (!mat)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Matrix (dim=%d) read from file:\n", dim);
        print_matrix(mat, dim, dim);
    }

    // Broadcast dim to all processes
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // only rank 0 has the dim, so if rank != 0 call bcast first,
    // mpi will block until all processes have called it and rank 0 will send the dim value to all processes

    int *my_rows = malloc(k * dim * sizeof(int)); // V mat di ogni processo

    // Distribute blocks of k rows to all processes in round-robin fashion
    MPI_Scatter(mat, k * dim, MPI_INT, my_rows, k * dim, MPI_INT, 0, MPI_COMM_WORLD);
    // mat is the source buffer, my_rows is the destination buffer for each process
    // k*dim is the number of elements in each block, MPI_INT is the type        // rank 0 sends the blocks to all processes, including itself

    printf("Rank %d received rows:\n", rank);
    print_matrix(my_rows, k, dim);

    // Sort the rows of my_rows based on the first column values
    sort_rows_by_first_column(my_rows, k, dim);
    printf("\nRank %d sorted rows:\n", rank);
    print_matrix(my_rows, k, dim);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes have sorted their rows before proceeding

    // find the process with the max value in my_rows[0][0]
    int my_max = my_rows[0];

    int *max_first_pos = malloc(size * sizeof(int));
    MPI_Allgather(&my_max, 1, MPI_INT, max_first_pos, 1, MPI_INT, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // print max_first_pos
        printf("Max first column values from all processes:\n");
        for (int i = 0; i < size; i++)
        {
            printf("Rank %d: %d\n", i, max_first_pos[i]);
        }
    }

    // Find the rank of the process with the maximum value in the first column
    int max_rank = find_max_rank(max_first_pos, size);

    if (rank == max_rank)
    {
        printf("Rank %d has the maximum value in the first column: %d\n", max_rank, my_rows[0]);
    }

    // Broadcast the first row of the process with the maximum value to all processes
    int rcv_first_row[dim];

    // Se sono il processo che ha la riga massima, copio la mia prima riga
    if (rank == max_rank)
    {
        for (int i = 0; i < dim; i++)
        {
            rcv_first_row[i] = my_rows[i];
        }
    }

    // Ora broadcast da max_rank a tutti i processi
    MPI_Bcast(rcv_first_row, dim, MPI_INT, max_rank, MPI_COMM_WORLD);

    printf("Process with rank %d received first row from rank %d:\n", rank, max_rank);
    print_array(rcv_first_row, dim);
    fflush(stdout);

    printf("\n");
    if (rank == 0)
    {
        free(mat);
    }
    free(my_rows);

    MPI_Finalize();
    return 0;
}
