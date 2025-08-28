/*
Scrivere in C un programma mpi con nproc processi che effettuano:

- il processo di rango 0 legge da file un vettore di interi V(dim) e li distribuisce
  secondo la modalità round-robin agli altri processi compreso se stesso. ne distribuisce
  k (dim/nproc) elementi in due blocchi non consecutivi di k/2 elementi, facendo
  l'ipotesi che k sia un intero pari.

- ogni processo, memorizza gli elementi ricevuti
  in una matrice A[2][k/2] e poi somma i i primi elementi delle due righe.

- i tre processi con il valore più grande di somma
  costituiscono un nuovo gruppo e ciascuno
  invia la propria matrice A agli altri due
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "input.txt"
#define DIM 20

void print_matrix(int *A, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", A[i * cols + j]);
        }
        printf("\n");
    }
}

void print_vector(int *V, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("%d ", V[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Group group_world;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);

    int dim = DIM;

    // check dim / nproc is integer
    if (rank == 0)
    {
        if (dim % (2 * size) != 0)
        {
            printf("Error: dim/size would not be even. Change dim or size.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int k = dim / size;
    int num_blocks = 2; // ogni blocco ha k/2 elementi
    int block_size = k / num_blocks;

    int *V = NULL;

    int *A = malloc(num_blocks * block_size * sizeof(int)); // 2 righe, k/2 colonne

    // datatype for send all blocks with one send to a proc
    MPI_Datatype blocks_per_proc;
    MPI_Type_vector(num_blocks, block_size, size * block_size, MPI_INT, &blocks_per_proc);
    MPI_Type_commit(&blocks_per_proc);

    // datatype for receiving all blocks together and save them in A[2][k/2]
    MPI_Datatype recv_type;
    MPI_Type_vector(num_blocks, block_size, block_size, MPI_INT, &recv_type);
    MPI_Type_commit(&recv_type);

    if (rank == 0)
    {

        FILE *fp = fopen(FILE_NAME, "r");

        if (fp == NULL)
        {
            printf("Error opening file %s\n", FILE_NAME);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // reading vector
        V = malloc(dim * sizeof(int));

        for (int i = 0; i < dim; i++)
        {
            fscanf(fp, "%d", &V[i]);
        }

        fclose(fp);

        printf("\n");
        printf("Vector read from file %s:\n", FILE_NAME);
        print_vector(V, dim);
        printf("\n");
        fflush(stdout);

        // proc0 needs to save his own A matrix

        for (int b = 0; b < num_blocks; b++)
        {

            for (int i = 0; i < block_size; i++)
            {
                A[b * block_size + i] = V[b * block_size * size + i];
            }
        }
        printf("Matrix A of proc %d:\n", rank);
        print_matrix(A, num_blocks, block_size);
        fflush(stdout);

        // now proc0 sends the two block in each send to processes in round robin
        for (int p = 1; p < size; p++)
        {
            MPI_Send(&V[p * block_size], 1, blocks_per_proc, p, 0, MPI_COMM_WORLD);
        }
    }
    else
    {

        // ogni altro processo fa 1 recv di tutti i suoi blocchi insieme
        // ma già in ricezione deve salvarli in A[2][k/2], usiamo datatype

        MPI_Recv(A, 1, recv_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Matrix A of proc %d:\n", rank);
        print_matrix(A, num_blocks, block_size);
        fflush(stdout);
    }

    // ogni processo somma i primi elementi delle due righe
    int my_sum = 0;
    my_sum = A[0] + A[block_size];
    printf("Sum of elements in A[0][0] and A[1][0] of proc %d: %d\n", rank, my_sum);
    fflush(stdout);

    // i 3 processi con il valore più grande di somma creano un nuovo gruppo
    // e ciascuno nel gruppo invia la propria matrice A agli altri due
    typedef struct
    {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    int my_sum_copy = my_sum;

    int ranks[3] = {-1,-1,-1};
    // int rank_max = -1;
    // int sum_max;

    local_max.value = my_sum_copy;
    local_max.rank = rank;

    // approccio con maxloc
    for (int i = 0; i < 3; i++) {
        

        MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        ranks[i] = global_max.rank;

        if (rank == global_max.rank) {
            local_max.value = -1; // assegno un valore minimo per non essere più il massimo
        }

    }

    // approccio senza maxloc
    // for (int i = 0; i < 3; i++)
    // {

    //     MPI_Allreduce(&my_sum_copy, &sum_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    //     int is_max = (my_sum_copy == sum_max) ? rank : -1;
    //     if (is_max != -1)
    //     {
    //         my_sum_copy = INT32_MIN;
    //     }

    //     int max_rank;
    //     MPI_Allreduce(&is_max, &max_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    //     ranks[i] = max_rank;
    // }

    printf("Ranks of processes with max sum: %d %d %d\n", ranks[0], ranks[1], ranks[2]);

    // abbiamo i ranks, possiamo costruire il nuovo gruppo
    // potevamo fare anche direttamente con comm_split con color ?? la traccia vuole per forza gruppo??

    // (rank == ranks[0] || rank == ranks[1] || rank == ranks[2]) ? 1 : MPI_UNDEFINED
    // int color = (rank == ranks[0] || rank == ranks[1] || rank == ranks[2]) ? 1 : MPI_UNDEFINED;

    // for key, we want that the top rank in ranks starts from 0 in the new comm for example
    // int key = (rank == ranks[0]) ? 0 : (rank == ranks[1]) ? 1 : 2;

    // MPI_Comm new_comm;
    // MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_comm);

    // int new_rank;
    // if (new_comm != MPI_COMM_NULL) {
    //     MPI_Comm_rank(new_comm, &new_rank);
    //     printf("New rank of process %d in new_comm: %d\n", rank, new_rank);
    // }


    // creazione gruppo manuale
    MPI_Group new_group;
    MPI_Group_incl(group_world, 3, ranks, &new_group);

    MPI_Comm new_comm;
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    int new_rank;

    if (new_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(new_comm, &new_rank);
        printf("New rank of process %d in new_comm: %d\n", rank, new_rank);


        int *recv_A = malloc(2 * block_size * sizeof(int));

        int dest = (new_rank + 1) % 3;
        int source = (new_rank + 2) % 3;
        // ogni processo nel nuovo gruppo invia la propria matrice A agli altri due
        MPI_Sendrecv(A, 2*block_size, MPI_INT, dest, 0, recv_A, 2*block_size, MPI_INT, source, 0, new_comm, MPI_STATUS_IGNORE);

        printf("Received matrix A in process %d from %d:\n", new_rank, source);
        print_matrix(recv_A, 2, block_size);
        fflush(stdout);

    }

    MPI_Type_free(&blocks_per_proc);
    MPI_Type_free(&recv_type);
    printf("\n");
    MPI_Finalize();
    return 0;
}