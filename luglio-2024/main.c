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

void print_vec(int *v, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("%d ", v[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = (int)sqrt(size);

    MPI_Comm topology;

    int dims[2] = {n, n};
    int periods[2] = {1, 1};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &topology);

    int coords[2];
    MPI_Cart_coords(topology, rank, 2, coords);

    // Communicator per righe
    MPI_Comm row_comm;
    MPI_Comm_split(topology, coords[0], rank, &row_comm);

    int row_comm_rank;
    MPI_Comm_rank(row_comm, &row_comm_rank);

    if (rank == 0)
    {
        printf("Topologia %dx%d creata con %d processi.\n\n", n, n, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Processo %d: coordinate (%d, %d), rank in row_comm: %d\n", rank, coords[0], coords[1], row_comm_rank);

    // Nomi file per prima colonna
    char *filenames[] = {
        "input1.txt", "input2.txt", "input3.txt"};

    int *A = NULL;
    int dim;

    // Solo processi prima colonna leggono file
    if (coords[1] == 0)
    {
        FILE *fp = fopen(filenames[coords[0]], "r");
        if (fp == NULL)
        {
            perror("Errore apertura file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(fp, "%d", &dim);

        A = malloc(dim * dim * sizeof(int));
        for (int i = 0; i < dim * dim; i++)
        {
            fscanf(fp, "%d", &A[i]);
        }
        fclose(fp);

        printf("Processo (%d, %d) ha letto matrice %dx%d dal file '%s':\n", coords[0], coords[1], dim, dim, filenames[coords[0]]);
        print_matrix(A, dim, dim);
        printf("\n");
    }

    // Broadcast dimensione matrice sulla riga
    MPI_Bcast(&dim, 1, MPI_INT, 0, row_comm);

    // Allocazione per gli altri processi della riga
    if (coords[1] != 0)
    {
        A = malloc(dim * dim * sizeof(int));
    }

    // Broadcast matrice sulla riga
    MPI_Bcast(A, dim * dim, MPI_INT, 0, row_comm);

    // Conferma ricezione matrice
    printf("Processo (%d, %d) ha ricevuto matrice %dx%d dalla riga %d:\n", coords[0], coords[1], dim, dim, coords[0]);
    print_matrix(A, dim, dim);
    printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);

    // Calcolo minimo sulla diagonale principale moltiplicata per (j+1)
    int local_min_diag = INT32_MAX;

    for (int i = 0; i < dim; i++)
    {
        int val = A[i * dim + i] * (coords[1] + 1);
        printf("Processo (%d, %d): elemento diagonale A[%d,%d]=%d, moltiplicato per %d = %d\n", coords[0], coords[1], i, i, A[i * dim + i], coords[1] + 1, val);
        if (val < local_min_diag)
        {
            local_min_diag = val;
        }
    }

    printf("Processo (%d, %d): minimo calcolato sulla diagonale principale moltiplicata = %d\n\n", coords[0], coords[1], local_min_diag);

    MPI_Barrier(MPI_COMM_WORLD);

    //punto 4 fatto con all gather e poi tutti si calcolano avg
    //Raccolta di tutti i minimi
    int *all_min_diag = malloc(size * sizeof(int));
    MPI_Allgather(&local_min_diag, 1, MPI_INT, all_min_diag, 1, MPI_INT, topology);

    // Tutti calcolano la media localmente
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += all_min_diag[i];
    }
    float avg = (float)sum / size;

    if (rank == 0)
    {
        printf("Minimi locali raccolti da tutti i processi:\n");
        print_vec(all_min_diag, size);
        printf("Media dei minimi locali: %f\n\n", avg);
    }

    MPI_Barrier(MPI_COMM_WORLD);


    if (local_min_diag < avg)
    {
        printf("Processo (%d, %d) stampa: valore min locale %d < media %f\n", coords[0], coords[1], local_min_diag, avg);
    }

    // PUNTO 4 fatto con allreduce

    // int global_sum;

    // MPI_Allreduce(&local_min_diag, &global_sum, 1, MPI_INT, MPI_SUM, topology);

    // float avg = (float)global_sum / size;

    // if (local_min_diag < avg)
    // {
    //     printf("Processo (%d, %d) stampa: valore min locale %d < media %f\n", coords[0], coords[1], local_min_diag, avg);
    // }

    free(A);
    // free(all_min_diag);

    MPI_Finalize();
    return 0;
}
