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

    if (dim % (k * size) != 0)
    {
        perror("DIM must be multiple of k*nproc");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int blocks_per_proc = dim / (k * size);

    int A[dim];
    int V[k * blocks_per_proc];

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
            fscanf(fp, "%d", &A[i]);
        }

        fclose(fp);

        printf("Initial vector A:\n");
        print_vector(A, dim);
        printf("\n");
        fflush(stdout);
    }

    // need to send blocks_per_proc blocks of k elements in round-robin

    MPI_Datatype blocks_type;
    MPI_Type_vector(blocks_per_proc, k, size * k, MPI_INT, &blocks_type);
    MPI_Type_commit(&blocks_type);

    if (rank == 0)
    {

        for (int p = 0; p < size; p++)
        {
            MPI_Send(&A[p * k], 1, blocks_type, p, 0, MPI_COMM_WORLD);
        }
    }

    // tutti ricevono, compreso p0

    MPI_Recv(V, k * blocks_per_proc, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("\nProcess %d received vector V:\n", rank);
    print_vector(V, k * blocks_per_proc);
    fflush(stdout);

    // il singolo processo, ordina in senso crescente il proprio vettore V.

    for (int i = 0; i < (k * blocks_per_proc) - 1; i++)
    {
        for (int j = i + 1; j < k * blocks_per_proc; j++)
        {

            if (V[i] > V[j])
            {
                // Swap

                int temp = V[i];
                V[i] = V[j];
                V[j] = temp;
            }
        }
    }

    printf("\nProcess %d sorted vector V:\n", rank);
    print_vector(V, k * blocks_per_proc);
    fflush(stdout);

    // il processo 0 raccoglie in un vettore i 5 valori più grandi
    //  diversi tra gli elementi dell'intero vettore attraverso
    //  ripetute operazioni di calcolo collettive.

    int top5[5];
    int temp_V[k * blocks_per_proc];

    for (int i = 0; i < k * blocks_per_proc; i++)
    {
        temp_V[i] = V[i];
    }

    typedef struct
    {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    for (int i = 0; i < 5; i++)
    {

        // find local max
        local_max.value = temp_V[0];
        for (int j = 1; j < k * blocks_per_proc; j++)
        {

            if (temp_V[j] > local_max.value)
            {
                local_max.value = temp_V[j];
                local_max.rank = rank;
            }
        }

        // find global max
        MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (rank == 0)
        {
            top5[i] = global_max.value;
        }

        // remove global max from all arrays
        for (int j = 0; j < k * blocks_per_proc; j++)
        {

            if (temp_V[j] == global_max.value)
            {
                temp_V[j] = INT32_MIN;
            }
        }
    }

    if (rank == 0)
    {
        printf("\nTop 5 distinct max values:\n");
        print_vector(top5, 5);
        fflush(stdout);
    }

    printf("\n");
    MPI_Type_free(&blocks_type);
    MPI_Finalize();
    return 0;
}
