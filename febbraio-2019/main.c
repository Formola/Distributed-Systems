/*
Realizzare in mpi C un'applicazione con nproc = n*n processi che:

- il proc di rango 0 legge da file una matrice A[dim][dim] e
  distribuisce a tutti, dim/nproc righe di A, una alla volta in round robin

- si crea una topologia 2d n*n. I processi sulla diagonale
  principale inviano usando una broadcast la propria porzione di matrice
  a tutti i processi della stessa riga.

- ogni processo calcola la matrice prodotto
  righe per colonne C[k][k], moltiplicando le proprie righe di A
  per la trasposta della matrice appena ricevuta.

- viene calcolata con k distinte operazioni di calcolo collettivo
  il massimo dei valori della diagonale di C.

- il processo che vince invia in broadcast il valore max ma non
  partecipa ai turni successivi.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define DIM 18

#define FILE_NAME "input.txt"

void print_vector(int *v, int dim)
{

    for (int i = 0; i < dim; i++)
    {
        printf("%d ", v[i]);
    }
    printf("\n");
}

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

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = DIM;

    int A[dim][dim];

    if (rank == 0)
    {

        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL)
        {
            printf("Error opening file %s\n", "FILE_NAME");
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

        printf("Matrix A:\n");
        print_matrix(&A[0][0], dim, dim);
        printf("\n");
        fflush(stdout);
    }

    int n = (int)sqrt(size);
    if (n * n != size)
    {
        perror("Number of processes must be a perfect square\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (dim % size != 0)
    {
        perror("Matrix dimension must be divisible by number of processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rows_per_proc = dim / size;

    MPI_Datatype rows_type;
    MPI_Type_vector(rows_per_proc, dim, dim * size, MPI_INT, &rows_type);
    MPI_Type_commit(&rows_type);

    if (rank == 0)
    {
        for (int p = 0; p < size; p++)
        {
            MPI_Send(&A[0][0] + p * dim, 1, rows_type, p, 0, MPI_COMM_WORLD);
        }
    }

    // tutti ricevono

    int local_A[rows_per_proc][dim];

    MPI_Recv(&local_A[0][0], rows_per_proc * dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("\nProcess %d received rows:\n", rank);
    print_matrix(&local_A[0][0], rows_per_proc, dim);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {n, n};
    int periods[2] = {1, 1};
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &topology);
    MPI_Cart_coords(topology, rank, ndims, coords);

    printf("Process %d has coordinates (%d,%d)\n", rank, coords[0], coords[1]);
    fflush(stdout);

    MPI_Comm row_comm;
    int remain_dims[2] = {0, 1};

    MPI_Cart_sub(topology, remain_dims, &row_comm);

    int local_A_T[dim][rows_per_proc]; // trasposta della matrice ricevuta

    // Se sono sulla diagonale: preparo la trasposta
    if (coords[0] == coords[1])
    {
        for (int i = 0; i < rows_per_proc; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                local_A_T[j][i] = local_A[i][j];
            }
        }

        printf("\n");

        printf("Process %d on diagonal has A_T:\n", rank);

        print_matrix(&local_A_T[0][0], dim, rows_per_proc);
        fflush(stdout);
    }

    // // Adesso il diagonale fa broadcast della trasposta sulla sua riga
    MPI_Bcast(local_A_T, dim * rows_per_proc, MPI_INT, coords[0], row_comm);

    // Debug
    printf("\nProc %d in riga %d ha ricevuto A_T:\n", rank, coords[0]);
    print_matrix(&local_A_T[0][0], dim, rows_per_proc);
    fflush(stdout);

    // ognuno calcola il prodotto
    int local_C[rows_per_proc][rows_per_proc];
    for (int i = 0; i < rows_per_proc; i++)
    {
        for (int j = 0; j < rows_per_proc; j++)
        {
            local_C[i][j] = 0;
            for (int k = 0; k < dim; k++)
            {
                local_C[i][j] += local_A[i][k] * local_A_T[k][j];
            }
        }
    }

    printf("\nProc %d computed local C:\n", rank);
    print_matrix(&local_C[0][0], rows_per_proc, rows_per_proc);
    fflush(stdout);

    // calcolo, con k distinte op
    // il massimo dei valori della diagonale di C

    // il processo che vince invia in broadcast il valore max ma non
    // partecipa ai turni successivi.

    int v_diag_max[rows_per_proc];
    int active[size]; // 1 = attivo, 0 = inattivo

    // tutti inizialmente attivi
    for (int i = 0; i < size; i++)
        active[i] = 1;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int t = 0; t < rows_per_proc; t++)
    {
        struct
        {
            int value;
            int rank;
        } local_data, global_data;

        if (active[rank])
            local_data.value = local_C[t][t];
        else
            local_data.value = INT32_MIN;

        local_data.rank = rank;

        // Reduce a root = 0
        MPI_Reduce(&local_data, &global_data, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

        // Root comunica chi è il vincitore a tutti (anche se rank = 0)
        MPI_Bcast(&global_data, 1, MPI_2INT, 0, MPI_COMM_WORLD);

        // Solo il vincitore fa broadcast del valore massimo
        MPI_Bcast(&global_data.value, 1, MPI_INT, global_data.rank, MPI_COMM_WORLD);

        // Aggiorna stato processi
        if (rank == global_data.rank)
            active[rank] = 0;

        v_diag_max[t] = global_data.value;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("\nVector of max on diagonal of C:\n");
        print_vector(v_diag_max, rows_per_proc);
        fflush(stdout);
    }

    printf("\n");
    MPI_Type_free(&rows_type);
    MPI_Finalize();
    return 0;
}
