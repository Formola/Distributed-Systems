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

void print_vector(int *V, int dim)
{

    for (int i = 0; i < dim; i++)
    {
        printf("%d ", V[i]);
    }
}

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

// Returns 1 (true) if rank is present in the vector, 0 (false) otherwise
int rank_in_vector(int rank, int *vec, int len)
{
    for (int i = 0; i < len; i++)
    {
        if (vec[i] == rank)
            return 1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Group group_world;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);

    int dim = 0;
    int *V = NULL;

    // proc 0 legge da file

    if (rank == 0)
    {
        FILE *file = fopen(FILE_NAME, "r");
        if (file == NULL)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fscanf(file, "%d", &dim);

        if (dim % size != 0)
        {
            fprintf(stderr, "Dimension %d is not divisible by number of processes %d\n", dim, size);
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        V = malloc(dim * sizeof(int));

        for (int i = 0; i < dim; i++)
        {
            fscanf(file, "%d", &V[i]);
        }

        fclose(file);

        // Print the vector read from file
        printf("Vector read from file: ");
        print_vector(V, dim);
        printf("\n\n");
    }

    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int k = dim / size;     // number of elements per process. size = nproc traccia.
    int block_size = k / 2; // size of each block to send

    int *local_vector = malloc(k * sizeof(int));

    MPI_Scatter(V, block_size, MPI_INT, local_vector, block_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(V + size * block_size, block_size, MPI_INT, local_vector + block_size, block_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Print the local vector for each process
    printf("Process %d received vector: ", rank);
    print_vector(local_vector, k);

    int A[2][block_size];
    for (int i = 0; i < block_size; i++)
    {
        A[0][i] = local_vector[i];
        A[1][i] = local_vector[i + block_size];
    }
    printf("\nProcess %d matrix A:\n", rank);
    print_matrix((int *)A, 2, block_size);

    int my_sum = A[0][0] + A[1][0];
    printf("Process %d sum of first elements: %d\n", rank, my_sum);

    // 3.point
    // i 3 processi con la somma maggiore costituiscono un nuovo gruppo da 3 e ognuno manda la propria
    // matrice A agli altri due

    // find the 3 process with highest value in my_sum
    MPI_Group new_group;
    MPI_Comm new_comm;

    int *all_sums = NULL;

    if (rank == 0)
    {
        all_sums = malloc(size * sizeof(int));
    }

    MPI_Gather(&my_sum, 1, MPI_INT, all_sums, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("All sums: ");
        for (int i = 0; i < size; i++)
        {
            printf("%d ", all_sums[i]);
        }
        printf("\n");
    }

    // Fase 2: calcolare i top 3
    int top3[3];
    if (rank == 0)
    {
        for (int i = 0; i < 3; i++)
        {
            int max_idx = -1;
            int max_val = -999999;
            for (int j = 0; j < size; j++)
            {
                if (all_sums[j] > max_val)
                {
                    max_val = all_sums[j];
                    max_idx = j;
                }
            }
            top3[i] = max_idx;
            all_sums[max_idx] = -999999; // esclude il trovato
        }
    }

    // Fase 3: broadcast dei top 3 a tutti
    MPI_Bcast(top3, 3, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Group_incl(group_world, 3, top3, &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    if (new_comm != MPI_COMM_NULL)
    {
        int new_size;
        MPI_Comm_size(new_comm, &new_size);

        // buffer per ricevere tutte le matrici
        int *rcv_A = malloc(new_size * 2 * block_size * sizeof(int));

        // raccoglie le matrici da tutti i processi del nuovo gruppo
        MPI_Allgather(A, 2 * block_size, MPI_INT,
                      rcv_A, 2 * block_size, MPI_INT,
                      new_comm);

        int new_rank;
        MPI_Comm_rank(new_comm, &new_rank);
        printf("Process %d (in new_comm rank %d) received matrices:\n", rank, new_rank);
        MPI_Barrier(new_comm);
        for (int p = 0; p < new_size; p++)
        {
            printf("From process %d in new group:\n", p);
            print_matrix(rcv_A + p * 2 * block_size, 2, block_size);
        }
        free(rcv_A);
    }

    printf("\n");
    printf("\n");
    MPI_Finalize();
    return 0;
}