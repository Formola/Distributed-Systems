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

    int A[size][dim][dim]; // una matrice dimxdim per ogni processo

    if (rank == 0)
    {
        FILE *fp = fopen(FILE_NAME, "r");

        if (fp == NULL)
        {
            perror("Errore nell'apertura del file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // lettura delle matrici dal file
        for (int p = 0; p < size; p++)
        {
            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    fscanf(fp, "%d", &A[p][i][j]);
                }
            }
        }

        fclose(fp);

        printf("\nMatrice lette dal file:\n");
        for (int p = 0; p < size; p++)
        {
            printf("Matrice %d:\n", p);
            print_matrix((int *)A[p], dim, dim);
            printf("\n");
            fflush(stdout);
        }
        printf("\n");
    }

    // proc 0 deve inviare la diagonale principale di ogni matrice al processo di rango i
    // tramite una sola send per processo
    int V[dim]; // vettore per memorizzare la diagonale principale

    MPI_Datatype diag_type;
    MPI_Type_vector(dim, 1, dim + 1, MPI_INT, &diag_type);
    MPI_Type_commit(&diag_type);

    if (rank == 0)
    {

        for (int p = 0; p < size; p++)
        {
            MPI_Send(&A[p][0][0], 1, diag_type, p, 0, MPI_COMM_WORLD);
        }
    }

    // tutti, compreso il proc 0, ricevono la diagonale principale
    MPI_Recv(&V[0], dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Processo %d ha ricevuto la diagonale principale: ", rank);
    print_vector(V, dim);
    fflush(stdout);

    // ogni processo ordina V in senso decrescente

    for (int i = 0; i < dim - 1; i++)
    {
        for (int j = i + 1; j < dim; j++)
        {
            if (V[i] < V[j])
            {
                // swap
                int temp = V[i];
                V[i] = V[j];
                V[j] = temp;
            }
        }
    }

    printf("Processo %d ha ordinato il vettore in senso decrescente: ", rank);
    print_vector(V, dim);
    fflush(stdout);

    // il processo 0 tramite operazioni di calcolo collettivo
    // si costruisce un vettore Vmax[dim] che contiene i primi dim elementi di valore più alto

    int Vmax[dim];
    int temp_V[dim];

    for (int i = 0; i < dim; i++)
    {
        temp_V[i] = V[i];
    }

    typedef struct
    {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    local_max.rank = rank;

    for (int i = 0; i < dim; i++)
    {

        // find my local max
        local_max.value = temp_V[0];
        for (int j = 1; j < dim; j++)
        {
            if (temp_V[j] > local_max.value)
            {
                local_max.value = temp_V[j];
            }
        }

        MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (rank == global_max.rank)
        {
            for (int j = 0; j < dim; j++)
            {
                if (temp_V[j] == global_max.value)
                {
                    temp_V[j] = INT32_MIN;
                }
            }
        }

        if (rank == 0)
        {
            Vmax[i] = global_max.value;
        }
    }

    if (rank == 0)
    {
        printf("\nProc %d: Il vettore Vmax con i primi %d elementi di valore più alto è: ", rank, dim);
        print_vector(Vmax, dim);
        fflush(stdout);
    }

    printf("\n");
    MPI_Type_free(&diag_type);
    MPI_Finalize();
    return 0;
}