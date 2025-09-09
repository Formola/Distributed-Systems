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

    int A[size][dim][dim]; // una mat per processo

    if (rank == 0)
    {
        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

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

        printf("Processo %d ha letto le seguenti matrici:\n", rank);
        for (int p = 0; p < size; p++)
        {
            printf("Matrice %d:\n", p);
            print_matrix((int *)A[p], dim, dim);
            printf("\n");
            fflush(stdout);
        }
        printf("\n");
    }

    // p0 invia a p_i la diagonale principale di A_i tramite una singola send per processo

    int V[dim];

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

    // tutti, compreso p0, ricevono la diagonale principale di A_i e la salvano in V
    MPI_Recv(V, dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("\nProcesso %d ha ricevuto la diagonale principale:\n", rank);
    print_vector(V, dim);
    fflush(stdout);

    // ogni processo memorizza gli elementi ricevuti in un vettore V[dim] che ordina in senso decrescente.
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

    printf("\nProcesso %d ha ordinato V in senso decrescente:\n", rank);
    print_vector(V, dim);
    fflush(stdout);


     // il processo 0, usando operazioni di calcolo collettivo,
     // si costruisce un vettore Vmax[dim] che contiene
     // i primi dim elementi di valore più alto.

     int Vmax[dim];

    int temp_V[dim];

    for (int i = 0; i < dim; i++)
    {
        temp_V[i] = V[i];
    }

    typedef struct {
        int value;
        int rank;
    } maxloc_t;

    maxloc_t local_max, global_max;

    for (int i = 0; i < dim; i++) {

        local_max.value = temp_V[0];
        local_max.rank = rank;
        for (int j = 1; j < dim; j++) {
            if (temp_V[j] > local_max.value) {
                local_max.value = temp_V[j];
            }
        }

        MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (rank == 0){
            Vmax[i] = global_max.value;
        }

        if (rank == global_max.rank) {

            for (int k = 0; k < dim; k++) {
                if (temp_V[k] == global_max.value) {
                    temp_V[k] = INT32_MIN;
                    break;
                }
            }
        }

    }


    if (rank == 0) {
        printf("Processo %d ha calcolato Vmax:\n", rank);
        print_vector(Vmax, dim);
        fflush(stdout);
    }

    printf("\n");
    MPI_Type_free(&diag_type);
    MPI_Finalize();
    return 0;
}