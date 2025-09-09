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
    int dim = DIM;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (dim % (2 * size) != 0)
    {
        perror("dim/size must be even");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int k = dim / size;
    int blocks_per_proc = 2;
    int block_size = k / blocks_per_proc; // k/2

    int V[dim];

    if (rank == 0)
    {
        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < dim; i++)
        {
            fscanf(fp, "%d", &V[i]);
        }
        fclose(fp);
        printf("Process %d read vector V: ", rank);
        print_vector(V, dim);
        printf("\n");
        fflush(stdout);
    }

    int A[blocks_per_proc][block_size]; // 2 x k/2

    MPI_Datatype blocks_type;
    MPI_Type_vector(blocks_per_proc, block_size, size * block_size, MPI_INT, &blocks_type);
    MPI_Type_commit(&blocks_type);

    if (rank == 0)
    {
        for (int p = 0; p < size; p++)
        {
            MPI_Send(&V[p * block_size], 1, blocks_type, p, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Recv(A, blocks_per_proc * block_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("\nProcess %d received matrix A:\n", rank);
    print_matrix((int *)A, blocks_per_proc, block_size);
    fflush(stdout);

    int my_sum = A[0][0] + A[1][0];

    typedef struct
    {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    local_max.value = my_sum;
    local_max.rank = rank;

    int ranks[3];

    for (int i = 0; i < 3; i++)
    {
        MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        ranks[i] = global_max.rank;
        if (rank == global_max.rank)
        {
            local_max.value = INT32_MIN; // Exclude this rank in the next iteration
        }
    }

    printf("Process %d found top ranks: %d, %d, %d\n", rank, ranks[0], ranks[1], ranks[2]);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Group world_group, new_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, 3, ranks, &new_group);

    MPI_Comm new_comm;
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    int recvs_A[2][blocks_per_proc][block_size];

    if (new_comm != MPI_COMM_NULL)
    {
        int new_rank;
        MPI_Comm_rank(new_comm, &new_rank);

        // for (int i = 0; i < 3; i++) {
        //     if ( i != new_rank) {
        //         MPI_Send(A, blocks_per_proc*block_size, MPI_INT, i, 0, new_comm);
        //     }
        // }

        // // ognuno fa 2 recv
        // for (int i = 0; i < 2; i++) {
        //     MPI_Recv(recvs_A[i], blocks_per_proc*block_size, MPI_INT, MPI_ANY_SOURCE, 0, new_comm, MPI_STATUS_IGNORE);
        // }

        // MPI_Barrier(new_comm);

        // printf("\nProcess %d in new group received matrices:\n", rank);
        // for (int i = 0; i < 3; i++) {
        //     printf("From process %d:\n", ranks[i != new_rank ? i : 2]);
        //     print_matrix((int *)recvs_A[i], blocks_per_proc, block_size);
        //     fflush(stdout);
        // }

        // invii: invio la mia matrice agli altri due processi usando modulo
        MPI_Send(A, blocks_per_proc * block_size, MPI_INT, (new_rank + 1) % 3, 0, new_comm);
        MPI_Send(A, blocks_per_proc * block_size, MPI_INT, (new_rank + 2) % 3, 0, new_comm);

        // ricezioni: ricevo dagli altri due processi usando modulo
        for (int i = 1; i <= 2; i++)
        {
            int src = (new_rank - i + 3) % 3; // calcolo del mittente
            MPI_Recv(recvs_A[i - 1], blocks_per_proc * block_size, MPI_INT, src, 0, new_comm, &status);
            printf("\nProcess %d received matrix from process %d:\n", rank, ranks[src]);
            print_matrix((int *)recvs_A[i - 1], blocks_per_proc, block_size);
            fflush(stdout);
        }
    }

    printf("\n");
    MPI_Type_free(&blocks_type);
    MPI_Finalize();
    return 0;
}