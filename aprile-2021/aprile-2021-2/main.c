/*
Realizzare in mpi C un'applicazione con nproc in cui:

- il processo di rango 0 legge da file nproc matrici di interi
  A_i[dim][dim] e invia a ciascuno degli altri processi di rango i
  la diagonale principale della matrice (và utilizzata un'unica send per processo).

- ogni processo memorizza gli elementi ricevuti in un vettore V[dim] che ordina in senso decrescente.

- il processo 0 deve calcolare il valore max globale
  dei vettori V dei soli processi pari, mentre
  il processo 1 deve calcolare il valore min globale
  dei vettori V dei soli processi pari.

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "matrix.txt"
#define DIM 4 // dimensione fissa della matrice

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
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = DIM;

    int A[size][dim][dim]; // solo il rank 0 le legge tutte da file, poi le distribuirà.

    if (rank == 0){

        FILE *fp = fopen(FILE_NAME, "r");

        if (fp == NULL){
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int p = 0; p < size; p++) {
            
            for (int i = 0; i < dim; i++){
                for (int j = 0; j < dim; j++) {
                    fscanf(fp, "%d", &A[p][i][j]);
                }
            }
        }

        printf("Matrices read from file:\n");
        for (int p = 0; p < size; p++) {
            printf("Matrix %d:\n", p);
            print_matrix(&A[p][0][0], dim, dim);
            printf("\n");
            fflush(stdout);
        }

        fclose(fp);
        printf("\n");
    }

    int V[dim];

    // proc0 deve mandare a ogni processo la diagonale della matrice A_i

    // usiamo datatype di send per mandare una diagonale tramite una singola send per processo
    
    MPI_Datatype diag_type;
    MPI_Type_vector(dim, 1, dim+1, MPI_INT, &diag_type);
    MPI_Type_commit(&diag_type);

    if (rank == 0){

        for (int p = 0; p < size; p++) {

            MPI_Send(&A[p][0][0], 1, diag_type, p, 0, MPI_COMM_WORLD);
        }

    }

    // tutti ricevono anche p0

    MPI_Recv(V, dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Process %d received diagonal: ", rank);
    print_vector(V, dim);
    fflush(stdout);

    // ogni processo ordina memorizza la diagonale in V e la ordina in senso decrescente

    for (int i = 0; i < dim-1; i++) {
        for (int j = i+1; j < dim; j++) {

            if (V[i] < V[j]) {

                //swamp
                int temp = V[i];
                V[i] = V[j];
                V[j] = temp;
            }
        }
    }

    printf("\nProcess %d sorted diagonal: ", rank);
    print_vector(V, dim);
    fflush(stdout);

    // creazione gruppo e comm per processi con rank pari
    MPI_Group world_group, even_group;
    MPI_Comm even_comm;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int even_size;

    if (size % 2 != 0) {
        even_size = size / 2 + 1;
    } else {
        even_size = size / 2;
    }

    int even_ranks[even_size];

    for (int i = 0; i < even_size; i++) {
        even_ranks[i] = i * 2;
    }

    MPI_Group_incl(world_group, even_size, even_ranks, &even_group);
    MPI_Comm_create(MPI_COMM_WORLD, even_group, &even_comm);


    int max_values[even_size];
    int min_values[even_size];

    if (even_comm != MPI_COMM_NULL) {
        printf("\nProcess %d is in even_comm\n", rank);
        fflush(stdout);

        int local_max = V[0]; // V è ordinato in senso decrescente, quindi il massimo è il primo elemento
        int local_min = V[dim-1]; // il minimo è l'ultimo elemento

        // MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, even_comm);
        MPI_Gather(&local_max, 1, MPI_INT, max_values, 1, MPI_INT, 0, even_comm);

        // gli even mandano i min singolarmente a proc 1
        MPI_Send(&local_min, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
    }


    if (rank == 0){

        printf("\nProcess 0 calculating global max from even ranks:\n");
        print_vector(max_values, even_size);
        fflush(stdout);

        // calcolo il max globale
        int global_max = max_values[0];
        for (int i = 1; i < even_size; i++) {
            if (max_values[i] > global_max) {
                global_max = max_values[i];
            }
        }

        printf("\nProcess 0 has computed global max: %d\n", global_max);
        fflush(stdout);


    } else if (rank == 1){

        for (int p = 0; p < even_size; p++) {

            int local_min;
            MPI_Recv(&local_min, 1, MPI_INT, even_ranks[p], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            min_values[p] = local_min;
        }

        printf("\nProcess 1 calculating global min from even ranks:\n");
        print_vector(min_values, even_size);
        fflush(stdout);

        // calcolo il min globale

        int global_min = min_values[0];
        for (int i = 1; i < even_size; i++) {
            if (min_values[i] < global_min) {
                global_min = min_values[i];
            }
        }

        printf("\nProcess 1 has computed global min: %d\n", global_min);
        fflush(stdout);

    } 

    printf("\n");
    MPI_Type_free(&diag_type);
    MPI_Finalize();
    return 0;
}