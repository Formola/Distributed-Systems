/*
Realizzare in mpi C un'applicazione con nproc in cui:

- il processo di rango 0 legge da file nproc matrici di interi
  A_i[dim][dim] e invia a ciascuno degli altri processi di rango i
  la diagonale principale della matrice (và utilizzata un'unica send per processo).

- ogni processo memorizza gli elementi ricevuti in un vettore V[dim] che ordina in senso decrescente.

- il processo 0, usando operazioni di calcolo collettivo,
  si costruisce un vettore Vmax[dim] che contiene
  i primi dim elementi di valore più alto.

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "matrix.txt"
#define DIM 4

typedef struct
{
    int value;
    int rank;
} maxloc_t;

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

    int dim = DIM;

    int *A = malloc(dim * dim * sizeof(int));
    int *V = malloc(dim * sizeof(int));

    if (rank == 0)
    {

        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL)
        {
            perror("Errore apertura file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Datatype diag_type;
        MPI_Type_vector(dim, 1, dim + 1, MPI_INT, &diag_type);
        MPI_Type_commit(&diag_type);

        for (int p = 0; p < size; p++)
        {

            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    fscanf(fp, "%d", &A[i * dim + j]);
                }
            }

            printf("Matrice %d:\n", p);
            print_matrix(A, dim, dim);
            fflush(stdout);

            if (p == 0)
            {
                // compute diag
                for (int i = 0; i < dim; i++)
                {
                    V[i] = A[i * dim + i];
                }
                printf("Process 0 computed diagonal: ");
                print_vector(V, dim);
                fflush(stdout);
            }
            else
            {
                MPI_Send(A, 1, diag_type, p, 0, MPI_COMM_WORLD);
            }
        }

        fclose(fp);
        MPI_Type_free(&diag_type);
    }
    else
    {

        // MPI_Datatype simple_block;
        // MPI_Type_vector(dim, 1, 1, MPI_INT, &simple_block);
        // MPI_Type_commit(&simple_block);

        MPI_Recv(V, dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process %d received diagonal: ", rank);
        print_vector(V, dim);
        fflush(stdout);
        // free(&simple_block);
    }

    // ogni processo ordina v in maniera decrescente

    for (int i = 0; i < dim - 1; i++)
    {
        for (int j = i + 1; j < dim; j++)
        {
            if (V[i] < V[j])
            {
                int temp = V[i];
                V[i] = V[j];
                V[j] = temp;
            }
        }
    }

    printf("Process %d sorted diagonal: ", rank);
    print_vector(V, dim);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    int *V_max = malloc(dim * sizeof(int));

    maxloc_t local_max, recv_max;

    // ciclo esterno per v_max
    for (int i = 0; i < dim; i++)
    {
        // APPROCCIO CON MAXLOC
        // local_max.value = INT32_MIN;
        // local_max.rank = rank;

        // for (int j = 0; j < dim; j++)
        // {
        //     if (V[j] > local_max.value)
        //     {
        //         local_max.value = V[j];
        //     }
        // }

        // MPI_Allreduce(&local_max, &recv_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // printf("Process %d found new max %d\n", rank, recv_max.value);

        // if (rank == recv_max.rank)
        // {

        //     printf("Process %d found new max %d\n", rank, recv_max.value);
        //     fflush(stdout);
        //     // dobb trovare il valore in V e mettergli INT_MIN
        //     int found = 0;
        //     for (int k = 0; k < dim; k++)
        //     {

        //         if (recv_max.value == V[k] && found == 0)
        //         {
        //             V[k] = INT32_MIN;
        //             found = 1;
        //         }
        //     }
        // }

        // if (rank == 0)
        // {
        //     V_max[i] = recv_max.value;
        //     printf("Process 0 found new max %d at iteration %d\n", V_max[i], i);
        //     fflush(stdout);
        // }

        int local_max = INT32_MIN;
        for (int i = 0; i < dim; i++)
        {
            if (V[i] > local_max)
                local_max = V[i];
        }

        // massimo globale
        int global_max;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // se il processo possiede il massimo
        if (local_max == global_max)
        {
            // trova l'indice e azzera
            for (int i = 0; i < dim; i++)
            {
                if (V[i] == global_max)
                {
                    V[i] = INT32_MIN;
                    break;
                }
            }
        }

        if (rank == 0)
        {
            V_max[i] = global_max;
            printf("Process 0 found new max %d at iteration %d\n", V_max[i], i);
            fflush(stdout);
        }
    }

    if (rank == 0)
    {
        printf("Process 0 found all max values: ");
        print_vector(V_max, dim);
    }

    free(V_max);
    free(V);
    free(A);
    MPI_Finalize();
    return 0;
}