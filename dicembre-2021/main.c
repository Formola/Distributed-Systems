/*
Realizzare in mpi C un'applicazione in cui:

- il processo di rango 0 legge da file juna matrice di interi
  A[DIMxDIM] con DIM multiplo intero di nproc, e ne distribuisce
  a tutti i processi compreso se stesso k righe, una alla volta
  in modalità round-robin.

- il singolo processo moltipla la matrice T[k*dim] per la
  prima riga della stessa matrice vista come una colonna di
  dim elementi.

- attraverso successive operazioni di calcolo collettivo, 
  viene calcolato un vettore che contiene i k elementi più
  grandi a partire dai valori contenuti in tutte le colonne
  ottenute dall'operazione precedente.

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "matrix.txt"

void print_mat(int *A, int row, int col)
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

void print_col(int *A, int row, int col){
    for (int i = 0; i < row; i++)
    {
        printf("%d\n", A[i * col]);
    }
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *A = NULL;
    int dim;

    if (rank == 0)
    {

        // legge da file

        FILE *fp = fopen(FILE_NAME, "r");

        if (fp == NULL)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // leggi dim
        fscanf(fp, "%d", &dim);

        if (dim % size != 0)
        {
            perror("Dimensione della matrice deve essere multipla del numero di processi");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // leggi mat dimxdim

        A = malloc(dim * dim * sizeof(int));
        for (int i = 0; i < dim * dim; i++)
        {
            fscanf(fp, "%d", &A[i]);
        }

        printf("Matrix read from file:\n");
        print_mat(A, dim, dim);
        printf("\n");
    }

    // send dim to other too
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // distribuzione in round robin di k righe una alla volta
    int k = dim / size; // k righe per processo

    // k righe * dim colonne
    int *local_A = malloc(k * dim  * sizeof(int));

    for (int i = 0; i < k; i++){
        MPI_Scatter(A+i*dim*size, dim, MPI_INT, local_A+i*dim, dim, MPI_INT, 0, MPI_COMM_WORLD);
    }

    printf("Local matrix for process %d:\n", rank);
    print_mat(local_A, k, dim);

    // local first row transposed
    // local_A[0][dim] 
    int *my_col = malloc(dim*sizeof(int));
    for (int i = 0; i < dim; i++) {
        my_col[i] = local_A[0*dim + i];
    }

    printf("Transposed first row for process %d:\n", rank);
    print_col(my_col, dim, 1);

    // each process computes the scalar product (k*dim)x(dim*1) = (k*1)

    int *my_product = malloc(k*sizeof(int));

    for (int i = 0; i < k; i++) {
        my_product[i] = 0;
        for (int j = 0; j < dim; j++) {
            my_product[i] += local_A[i*dim + j] * my_col[j];
        }
    }

    printf("Local product for process %d:\n", rank);
    print_col(my_product, k, 1);


    // calcolare un vettore che contiene i k
    // valori più grandi presenti in tutte le colonne my_product

    
    int *top_k_global = malloc(k * sizeof(int));

    int *tmp_values = malloc(k * sizeof(int));
    memcpy(tmp_values, my_product, k * sizeof(int));
    
    int global_max;

    for (int i = 0; i < k; i++) {
        


        MPI_Allreduce(tmp_values, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        top_k_global[i] = global_max;
        

        // remove global_max from tmp_values
        for (int j = 0; j < k; j++) {
            if (tmp_values[j] == global_max) {
                tmp_values[j] = INTMAX_MIN;
                break;
            }
        }

    }

    printf("Top %d values across all processes:\n", k);
    for (int i = 0; i < k; i++) {
        printf("%d ", top_k_global[i]);
    }
    printf("\n");
    



    free(local_A);
    free(my_col);
    free(my_product);
    free(tmp_values);
    printf("\n");
    MPI_Finalize();
    return 0;
}