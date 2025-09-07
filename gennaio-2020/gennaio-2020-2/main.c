/*
Realizzare in mpi C un'applicazione con numero di processi nproc in cui:

- il processo di rango 0 legge da due file un vettore di interi
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

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = DIM;

    int A[DIM], B[DIM];

    if (rank == 0)
    {

        char *filenames[] = {"vector_A.txt", "vector_B.txt"};

        FILE *file_A = fopen(filenames[0], "r");
        FILE *file_B = fopen(filenames[1], "r");

        if (file_A == NULL || file_B == NULL)
        {
            printf("Error opening file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < dim; i++)
        {
            fscanf(file_A, "%d", &A[i]);
            fscanf(file_B, "%d", &B[i]);
        }
        fclose(file_A);
        fclose(file_B);

        printf("Vector A: ");
        print_vector(A, dim);
        fflush(stdout);

        printf("Vector B: ");
        print_vector(B, dim);
        fflush(stdout);

        printf("\n");
    }

    if (dim % size != 0)
    {
        perror("dim must be multiple of size");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int k = dim / size; // block size for A

    // proc0 deve distribuire A in blocchi consecutivi di k elementi
    // quindi ci basta scatter

    int local_A[k];

    MPI_Scatter(A, k, MPI_INT, local_A, k, MPI_INT, 0, MPI_COMM_WORLD);

    printf("\nRank %d received A: ", rank);
    print_vector(local_A, k);
    fflush(stdout);

    // proc0 distribuisce B in round-robin in blocchi consecutivi di m interi

    int m = 2; // block size for B
    if (dim % (size * m) != 0)
    {
        perror("dim must be multiple of size*m");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int num_blocks_per_proc = dim / (size * m); // number of blocks per process
    int local_B[num_blocks_per_proc * m];

    MPI_Datatype block_type;
    MPI_Type_vector(num_blocks_per_proc, m, size * m, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);

    if (rank == 0)
    {
        for (int p = 0; p < size; p++)
        {
            MPI_Send(&B[p * m], 1, block_type, p, 0, MPI_COMM_WORLD);
        }
    }

    // tutti, compreso proc0, ricevono B
    MPI_Recv(local_B, num_blocks_per_proc * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("\nRank %d received B: ", rank);
    print_vector(local_B, num_blocks_per_proc * m);
    fflush(stdout);

    if (k != num_blocks_per_proc * m)
    {
        perror("k must be equal to num_blocks_per_proc*m");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ogni processo calcola il prod scalare di Ai*Bi
    int local_prod = 0;
    for (int i = 0; i < k; i++)
    {
        local_prod += local_A[i] * local_B[i];
    }

    printf("\nRank %d local product: %d\n", rank, local_prod);
    fflush(stdout);

    // ogni processo distribuisce il suo prodotto scalare a tutti usando una operazione collettiva
    int T[size];

    MPI_Allgather(&local_prod, 1, MPI_INT, T, 1, MPI_INT, MPI_COMM_WORLD);

    printf("\nRank %d received T: ", rank);
    print_vector(T, size);
    fflush(stdout);

    // in base ai valori di T, si crea una pipe virtuale tra i processi
    // il processo con T[rank] massimo sarà il primo della pipe, poi il secondo

    int pipe_ranks[size];

    int T_copy[size];

    for (int i = 0; i < size; i++)
    {
        T_copy[i] = T[i];
    }

    for (int i = 0; i < size; i++) {
        int max_index = 0;
        for (int j = 1; j < size;j++) {
            if (T_copy[j] > T_copy[max_index]) {
                max_index = j;
            }
        }
        pipe_ranks[i] = max_index;
        T_copy[max_index] = INT32_MIN; // Invalidate the max element
    }

    printf("\nRank %d pipe ranks: ", rank);
    print_vector(pipe_ranks, size);
    fflush(stdout);

    MPI_Group world_group, pipe_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, size, pipe_ranks, &pipe_group);
    MPI_Comm pipe_comm;
    MPI_Comm_create(MPI_COMM_WORLD, pipe_group, &pipe_comm);

    int pipe_rank;
    MPI_Comm_rank(pipe_comm, &pipe_rank);

    printf("\nRank %d in pipe has rank %d\n", rank, pipe_rank);
    fflush(stdout);


    // i primo processo della pipe deve inviare il proprio vettore A al secondo
    // che lo somma al proprio B e lo fa scorrere.
    // L'operazione si ripete fino all'ultimo processo della pipe
    // che trasmette in broadcast il vettore risultato a tutti

    int result[k];

    if (pipe_rank == 0){

        MPI_Send(local_A, k, MPI_INT, pipe_rank+1, 0, pipe_comm);
        
    } else if (pipe_rank != size - 1) {
        
        int temp[k];
        MPI_Recv(temp, k, MPI_INT, pipe_rank -1, 0, pipe_comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < k; i++) {
            result[i] = temp[i] + local_B[i];
        }

        MPI_Send(result, k, MPI_INT, pipe_rank+1, 0, pipe_comm);
    } else {

        int temp[k];
        MPI_Recv(temp, k, MPI_INT, pipe_rank-1, 0, pipe_comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < k; i++) {
            result[i] = temp[i] + local_B[i];
        }
    }

    MPI_Bcast(result, k, MPI_INT, size - 1, MPI_COMM_WORLD);

    printf("\nRank %d final result: ", rank);
    print_vector(result, k);
    fflush(stdout);



    printf("\n");
    MPI_Type_free(&block_type);
    MPI_Finalize();
    return 0;
}
