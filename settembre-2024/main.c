/*
Realizzare in MPI C un programma in cui:

- il processo di rango 0 legge da file un vettore
  di interi A[DIM] e ne distribuisce a tutti i
  processi compreso se stesso blocchi di k elementi consecutivi
  in modalità round-robin. DIM si suppone multiplo di k*nproc.

- il singolo processo, ordina in senso crescente il proprio vettore V.

- il processo 0 raccoglie in un vettore i 5 valori più grandi
  diversi tra gli elementi dell'intero vettore attraverso
  ripetute operazioni di calcolo collettive.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define FILE_NAME "input.txt"
#define DIM 12

int print_vector(int *v, int dim)
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dim = DIM;

    int k = 2;

    if ( dim % (k*size) != 0 )
    {
        if (rank == 0)
            printf("dim must be multiple of k*nproc. change DIM or k.\n");
        MPI_Finalize();
        return -1;
    }

    int num_blocks = dim / k;
    int blocks_per_proc = num_blocks / size;
    int blocks_size = blocks_per_proc * k;

    // create datatype for send all blocks to a process with a single send

    MPI_Datatype blocks_type;
    MPI_Type_vector(blocks_per_proc, k, size*k, MPI_INT, &blocks_type);
    MPI_Type_commit(&blocks_type);

    int *V = malloc(blocks_size * sizeof(int));

    if (rank == 0) {

        FILE *fp = fopen(FILE_NAME, "r");

        if (fp == NULL){
            perror("Error opening file");
            MPI_Finalize();
            return -1;
        }

        // proc0 legge dal file

        int *A = malloc(dim * sizeof(int));

        for (int i = 0; i < dim; i++){
            fscanf(fp, "%d", &A[i]);
        }

        printf("\nVettore letto da file:\n");
        print_vector(A, dim);
        fflush(stdout);
        fclose(fp);


        // proc0 deve distribuire usando il datatype, ma prima
        // deve copiarsi i suoi blocchi da tenere
        for (int b = 0; b < num_blocks; b+=size){
            for (int i = 0; i < k; i++){
                int index = (b/size)*k + i;
                V[index] = A[b*k+i];
            }
        }

        printf("\nVettore dei blocchi di proc0:\n");
        print_vector(V, blocks_size);
        fflush(stdout);

        // now proc0 can sends to others
        for (int p = 1; p < size; p++) {
            MPI_Send(A+p*k, 1, blocks_type, p, 0, MPI_COMM_WORLD);
        }

    } else {

        MPI_Recv(V, blocks_per_proc *k, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("\nVettore dei blocchi di proc%d:\n", rank);
        print_vector(V, blocks_size);
        fflush(stdout);

    }

    // ogni processo ordina il proprio V in senso crescente

    for (int i = 0; i < (blocks_size)-1; i++){
        for (int j = i+1; j < (blocks_size); j++) {

            if (V[i] > V[j]) {

                //swap
                int temp = V[i];
                V[i] = V[j];
                V[j] = temp;
            }
        }
    }

    printf("\nVettore ordinato di proc%d:\n", rank);
    print_vector(V, blocks_size);
    fflush(stdout);

    // we are gonna use this for computing max and remove max for next round
    int *temp_V = malloc(blocks_size * sizeof(int));
    for (int i = 0; i < blocks_size; i++){
        temp_V[i] = V[i];
    }

    typedef struct {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;


    int *max_values = malloc(5 * sizeof(int));

    for (int i = 0; i < 5; i++) {

        // find local max
        local_max.value = -1;
        local_max.rank = rank;

        for (int i = 0; i < blocks_size; i++) {
            if (temp_V[i] > local_max.value) {
                local_max.value = temp_V[i];
            }
        }

        // un alternativa qui poteva essere fare reduce al proc0 solo del global_max, e
        // poi proc0 doveva mandare il globalmax in bcast a tutti cosi gli altri possono escluderlo
        MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (rank == 0){

            max_values[i] = global_max.value;
        }

        if (rank == global_max.rank) {
            // this proc has the max, so we need to replace it in temp_V with INT32_MIN
            for (int i = 0; i < blocks_size; i++){
                if (temp_V[i] == global_max.value) {
                    temp_V[i] = INT32_MIN;
                }
            }
        }

    }

    if (rank == 0){
        printf("\n");
        printf("Proc %d has max values vector:\n", rank);
        print_vector(max_values, 5);
        fflush(stdout);
    }


    printf("\n");
    free(V);
    MPI_Type_free(&blocks_type);
    MPI_Finalize();
    return 0;
}
