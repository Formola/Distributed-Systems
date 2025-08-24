/*
Realizzare in mpi C un'applicazione con numero di processi nproc in cui:

- il processo di rango 9 legge da due file un vettore di interi
  A[dim] e B[dim]

- il processo di rango 0 distribuisce A
  agli altri processi, compreso se stesso, in blocchi
  consecutivi di dim/nproc elementi, si suppone
  dim/nproc sia intero k.

- il processo 0 distribuisce B agli altri, compreso se stesso
  in round-robin in blocchi consecutivi di m interi
  si suppone dim/(nproc*m) sia intero

- ogni processo calcola il prod scalare di Ai*Bi e lo distribuisce
  usando un operazione di com collettiva a tutti,
  cosi che tutti i processi abbiamo un vettore T[nproc].

- in base ai valori di T si crea una pipe virtuale
  tra i processi, nel senso che il processo con
  T[rank] massimo sarà il primo della pipe, poi il secondo ecc.

- il primo processo della pipe, dovrà inviare il proprio
  vettore Ai al secondo processo della pipe, il quale lo somma al proprio B e lo
  fa scorrere nella pipe. L'operazione si ripete fino all'ultimo di processo
  della pipe, che trasmette in broadcast il vettore risultato a tutti.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define DIM 16

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

    char *filenames[2] = {"vector_A.txt", "vector_B.txt"};

    int dim = DIM;

    int A[dim], B[dim];

    if (rank == 0)
    {

        FILE *fp_A = fopen(filenames[0], "r");
        FILE *fp_B = fopen(filenames[1], "r");
        if (fp_A == NULL || fp_B == NULL)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // reading A
        for (int i = 0; i < dim; i++)
        {
            fscanf(fp_A, "%d", &A[i]);
        }
        // reading B
        for (int i = 0; i < dim; i++)
        {
            fscanf(fp_B, "%d", &B[i]);
        }

        print_vector(A, dim);
        print_vector(B, dim);

        fclose(fp_A);
        fclose(fp_B);
    }

    int M = 2;
    int k = DIM / size;
    int p = DIM / (size * M);

    int lvec_a[k];
    int lvec_b[M * p];

    MPI_Scatter(A, k, MPI_INT, lvec_a, k, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received A: ", rank);
    print_vector(lvec_a, k);
    fflush(stdout);

    MPI_Datatype block;
    MPI_Type_vector(p, M, M * size, MPI_INT, &block);
    MPI_Type_commit(&block);

    if (rank == 0)
    {
        for (int i = 0; i < size; i++)
        {
            MPI_Send(&B[i * M], 1, block, i, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Recv(lvec_b, M * p, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Process %d received B: ", rank);
    print_vector(lvec_b, M * p);
    fflush(stdout);

    int T[size];

    if (k != M * p)
    {
        fprintf(stderr, "Dimensione di A e B ricevuti non compatibile, cambia m\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ognuno calcola prod scalare
    // prod scalare x definzione hanno stessa dim
    // anche se l'abbiamo calcolata diversa, è la stessa
    int local_prod_scal = 0;
    for (int i = 0; i < k; i++)
    {
        local_prod_scal += lvec_a[i] * lvec_b[i];
    }

    printf("Process %d computed local_prod_scal: %d\n", rank, local_prod_scal);
    fflush(stdout);

    MPI_Allgather(&local_prod_scal, 1, MPI_INT, T, 1, MPI_INT, MPI_COMM_WORLD);

    printf("Process %d computed T: ", rank);
    print_vector(T, size);
    fflush(stdout);

    // Creazione della pipe virtuale
    int orders[size];

    for (int i = 0; i < size; i++)
    {
        int max = INT32_MIN;
        int max_index = -1;
        for (int j = 0; j < size; j++)
        {
            if (max < T[j])
            {
                max = T[j];
                max_index = j;
            }
        }
        orders[i] = max_index;
        T[max_index] = INT32_MIN;
    }
    printf("Process %d computed orders: ", rank);
    print_vector(orders, size);
    fflush(stdout);

    // creiamo la topologia con questo orders, prima bisogna creare gruppo e comm

    MPI_Group world_group, new_group;
    MPI_Comm new_comm;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int ranks[size];
    for (int i = 0; i < size; i++)
    {
        ranks[i] = orders[i];
    }

    MPI_Group_incl(world_group, size, ranks, &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
    int new_rank;
    MPI_Comm_rank(new_comm, &new_rank);

    int ndims = 1;
    int dims[1] = {size};
    int periods[1] = {0};

    MPI_Comm pipe;

    MPI_Cart_create(new_comm, ndims, dims, periods, 0, &pipe);
    int coords[1];
    MPI_Cart_coords(pipe, new_rank, ndims, coords);

    printf("Process %d in new_comm has coordinates: (%d)\n", rank, coords[0]);

    // ora bisogna passare lvec_A al secondo, che somma e invia al prossimo ecc

    int local_sum[k];
    int final_vector[k];

    int rank_source, rank_dest;

    if (pipe != MPI_COMM_NULL)
    {
        MPI_Cart_shift(pipe, 0, 1, &rank_source, &rank_dest);
    }

    if (new_rank == ranks[0])
    {

        MPI_Send(lvec_a, k, MPI_INT, rank_dest, 0, pipe);
    }
    else if (new_rank >= 1 && new_rank < size - 1)
    {
        MPI_Recv(local_sum, k, MPI_INT, rank_source, 0, pipe, MPI_STATUS_IGNORE);

        for (int i = 0; i < k; i++)
        {
            local_sum[i] += lvec_b[i];
        }

        MPI_Send(local_sum, k, MPI_INT, rank_dest, 0, pipe);
    }
    else
    {
        // last one in pipe
        MPI_Recv(local_sum, k, MPI_INT, rank_source, 0, pipe, MPI_STATUS_IGNORE);

        for (int i = 0; i < k; i++)
        {
            local_sum[i] += lvec_b[i];
            final_vector[i] = local_sum[i];
        }
    }
    if (pipe != MPI_COMM_NULL)
        MPI_Barrier(pipe);

    MPI_Bcast(final_vector, k, MPI_INT, ranks[size - 1], MPI_COMM_WORLD);

    printf("Process %d received final_vector: ", rank);
    print_vector(final_vector, k);
    fflush(stdout);

    MPI_Type_free(&block);
    MPI_Finalize();
    return 0;
}
