/*
Realizzare in mpi C un'applicazione in cui:

- il processo di rango 0 legge da file una matrice di interi A[DIM][DIM]
  e ne distribuisce a tutti i processi compreso se stesso blocchi di k righe consecutive
  in modalità round-robin.
  DIM si suppone multiplo di k*nproc.

- il singolo processo ordina in senso crescente le righe della propria matrice V,
  in base agli elementi che si trovano nella prima colonna.

- la riga che presenta il valore max in V[0][0] dovrà
  essere inviata a tutti i processi.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "matrix.txt"
#define DIM 12

void print_vector(int *v, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("%d ", v[i]);
    }
    printf("\n");
}

void print_matrix(int *mat, int row, int col)
{

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d ", mat[i * col + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    int dim = DIM;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int k = 2;

    if (dim % (k * size) != 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Errore: DIM deve essere un multiplo intero di k*size\n");
        }
        MPI_Finalize();
        return 1;
    }

    int num_blocks = dim / k;
    int blocks_per_proc = num_blocks / size;
    int block_size = k * dim;
    int blocks_size = block_size * blocks_per_proc;

    int *V = malloc(blocks_per_proc * k * dim * sizeof(int));

    // datatype for sending blocks of k rows
    MPI_Datatype blocks_type;
    MPI_Type_vector(blocks_per_proc, block_size, size*k*dim, MPI_INT, &blocks_type);
    MPI_Type_commit(&blocks_type);

    //datatype for recv and store in V[blocks_per_proc*k][DIM]
    // non necessario tanto le righe sono memorizzante in maniera contigua.
    MPI_Datatype recv_type;
    MPI_Type_vector(blocks_per_proc, block_size, block_size, MPI_INT, &recv_type);
    MPI_Type_commit(&recv_type);

    if (rank == 0)
    {

        FILE *fp = fopen(FILE_NAME, "r");

        if (fp == NULL)
        {
            perror("Errore apertura file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int *A = malloc(dim * dim * sizeof(int));

        // lettura da file matrix
        for (int i = 0; i < dim * dim; i++)
        {
            fscanf(fp, "%d", &A[i]);
        }

        printf("Matrice A letta da file:\n");
        print_matrix(A, dim, dim);
        printf("\n");
        fflush(stdout);
        fclose(fp);

        // proc0 memorizes his blocks before distributing
        for (int b = 0; b < num_blocks; b += size)
        {
            int index = (b / size) * block_size;
            for (int i = 0; i < dim * k; i++)
            {
                V[index + i] = A[b * k * dim + i];
            }
        }

        printf("Matrice V for process %d:\n", rank);
        print_matrix(V, blocks_per_proc * k, dim);

        // now proc0 can sends blocks
        // with datatype of all blocks per proc, we just need size-1 send
        for (int p = 1; p < size; p++) {

            MPI_Send(&A[p*k*dim], 1, blocks_type, p, 0, MPI_COMM_WORLD);

        }
    } else {


        // MPI_Recv(V, 1, recv_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(V, blocks_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Matrice V for process %d:\n", rank);
        print_matrix(V, blocks_per_proc * k, dim);
        fflush(stdout);
        printf("\n");
    }


    // ogni processo ordina le righe di V in senso crescente in base ai valori
    // nella prima colonna

    for (int i = 0; i < (blocks_per_proc*k)-1; i++) {

        for (int j = i+1; j < (blocks_per_proc*k); j++) {

            if (V[i*dim] > V[j*dim]) {

                //swap whole row
                for (int k = 0; k < dim; k++) {

                    int temp = V[i*dim + k];
                    V[i*dim+k] = V[j*dim +k];
                    V[j*dim+k] = temp;
                }
            }
        }
    }

    printf("Matrice V ordinata per processo %d:\n", rank);
    print_matrix(V, blocks_per_proc * k, dim);


    // la riga col max in V[0][0] và inviata a tutti

    typedef struct {

        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    local_max.value = V[0];
    local_max.rank = rank;

    MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);


    int *recv_V = malloc(blocks_per_proc * k * dim * sizeof(int));

    if ( rank == global_max.rank) {

        int *v_first_row = malloc(dim * sizeof(int));
        for ( int i = 0; i < dim; i++) {
            v_first_row[i] = V[i];
        }

        printf("riga V da inviare %d:\n", rank);
        print_matrix(v_first_row, 1, dim);
        fflush(stdout);

        // sono io che devo inviare la mia riga
        for (int p = 0; p < size; p++) {

            if ( p == rank ) continue;
            MPI_Send(v_first_row, dim, MPI_INT, p, 1, MPI_COMM_WORLD);
        }
        free(v_first_row);
    } else {

        MPI_Recv(recv_V, dim, MPI_INT, global_max.rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("riga V ricevuta da %d:\n", rank);
        print_matrix(recv_V, 1, dim);
        fflush(stdout);
        printf("\n");
    }

    free(recv_V);

    printf("\n");
    MPI_Type_free(&blocks_type);
    MPI_Type_free(&recv_type);
    MPI_Finalize();
    return 0;
}