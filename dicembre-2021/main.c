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

#define DIM 6

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

void print_vector(int *v, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("%d ", v[i]);
    }
    printf("\n");
}

void print_colonna(int *col, int dim){
    for (int i = 0; i < dim; i++)
    {
        printf("%d\n", col[i]);
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

    int k = dim / size;

    if (dim % (k * size) != 0)
    {
        if (rank == 0)
            printf("dim non multiplo di nproc\n");
        MPI_Finalize();
        return -1;
    }

    int A[dim][dim];

    if (rank == 0) {

        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL) {
            perror("Errore apertura file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                fscanf(fp, "%d", &A[i][j]);
            }
        }

        printf("Matrice letta:\n");
        print_mat(&A[0][0], dim, dim);
        printf("\n");
        fflush(stdout);
    }

    // ogni processo riceve k righe, una alla volta in round robin
    int T[k][dim];

    // usiamo un datatype di send per mandare tutte le k righe insieme
    MPI_Datatype k_rows_type;
    MPI_Type_vector(k, dim, size*dim, MPI_INT, &k_rows_type);
    MPI_Type_commit(&k_rows_type);

    if (rank == 0){

        for (int p = 0; p < size; p++) {
            MPI_Send(&A[p][0], 1, k_rows_type, p, 0, MPI_COMM_WORLD);
        }

    }

    // tutti, compreso proc0, ricevono in T
    MPI_Recv(&T[0][0], k*dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Process %d ha ricevuto la matrice T:\n", rank);
    print_mat(&T[0][0], k, dim);
    fflush(stdout);

    // bisogna fare il prodotto della prima riga ma vista come colonna
    int my_first_col[dim];

    for (int i = 0; i < dim; i++){
        my_first_col[i] = T[0][i];
    }

    printf("Process %d ha calcolato la prima riga vista come colonna per il prodotto:\n", rank);
    print_colonna(my_first_col, dim);
    fflush(stdout);

    // prodotto matrice * colonna
    // [k][dim] * [dim][1] = [k][1]

    int my_prod[k];

    for (int i = 0; i < k; i++) {
        my_prod[i] = 0;
        for (int j = 0; j < dim; j++) {

            my_prod[i] += T[i][j] * my_first_col[j];
        }
    }

    printf("Process %d ha calcolato il prodotto:\n", rank);
    print_colonna(my_prod, k);
    fflush(stdout);


    // attraverso operazioni di calcolo collettiver successive,
    // viene calcolato un vettore che contiene i k elementi
    // più grandi a partire dai calori contenuti in 
    // tutte le colonne dell'operazione di prodotto

    int max_values[k];

    typedef struct {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    int my_prod_copy[k];
    for (int i = 0; i < k; i++){
        my_prod_copy[i] = my_prod[i];
    }

    int my_max;

    for (int i = 0; i < k; i++){

        my_max = my_prod_copy[0];
        for (int j = 1; j < k; j++){
            if (my_prod_copy[j] > my_max){
                my_max = my_prod_copy[j];
            }
        }

        local_max.value = my_max;
        local_max.rank = rank;

        MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);


        // questo controllo potrebbe far si che ci siano doppioni di valori max nel vettore finale
        if (rank == global_max.rank){
            
            // devo 'rimuovere' il max dal prossimo giro
            for (int j = 0; j < k; j++){

                if (my_prod_copy[j] == global_max.value){
                    my_prod_copy[j] = INT32_MIN;
                }
            }
        }

        max_values[i] = global_max.value;

    }

    printf("Process %d ha calcolato i k massimi:\n", rank);
    print_colonna(max_values, k);
    fflush(stdout);


    MPI_Type_free(&k_rows_type);
    MPI_Finalize();
    return 0;
}