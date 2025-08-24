/*
Realizzare in mpi C un'applicazione con nproc in cui:

- il processo di rango 0 legge da file nproc matrici di interi
  A_i[dim][dim] e invia a ciascuno degli altri processi di rango i
  la diagonale principale della matrice (và utilizzata un'unica send per processo).

- ogni processo memorizza gli elementi ricevuti in un vettore V[dim] che ordina in senso decrescente.

- il processo 0 deve calcolare il valore max globale
  dei vettori V dei soli processi pari, mentre
  il processo 1 deve calcolare il valore min globale
  dei vettori V dei soli processi pari.

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "matrix.txt"
#define DIM 4 // dimensione fissa della matrice
// se la scriviamo del file, dovremmo fare delle send in più.

void print_matrix(int *A, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d ", A[i * col + j]);
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

    int *A = NULL;
    int *diag = malloc(DIM * sizeof(int));

    if (rank == 0)
    {
        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL)
        {
            perror("Errore apertura file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // datatype per inviare la diagonale con una singola send per processo
        MPI_Datatype diag_type;
        MPI_Type_vector(DIM, 1, DIM + 1, MPI_INT, &diag_type);
        MPI_Type_commit(&diag_type);

        for (int proc = 0; proc < size; proc++)
        {
            A = malloc(DIM * DIM * sizeof(int));

            for (int i = 0; i < DIM; i++)
            {
                for (int j = 0; j < DIM; j++)
                {
                    fscanf(fp, "%d", &A[i * DIM + j]);
                }
            }

            printf("Matrice %d:\n", proc);
            print_matrix(A, DIM, DIM);

            if (proc == 0)
            {
                for (int i = 0; i < DIM; i++)
                {
                    diag[i] = A[i * DIM + i];
                }
                printf("Processo %d ha la diagonale:\n", proc);
                print_vector(diag, DIM);
            }
            else
            {
                MPI_Send(A, 1, diag_type, proc, 1, MPI_COMM_WORLD);
            }

            free(A);
        }
        MPI_Type_free(&diag_type);
        fclose(fp);
    }
    else
    {
        MPI_Recv(diag, DIM, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Processo %d ha ricevuto la diagonale:\n", rank);
        print_vector(diag, DIM);
    }

    int *V = malloc(DIM * sizeof(int));
    for (int i = 0; i < DIM; i++)
    {
        V[i] = diag[i];
    }

    // ordinamento decrescente
    for (int i = 0; i < DIM - 1; i++)
    {
        for (int j = i + 1; j < DIM; j++)
        {
            if (V[i] < V[j])
            {
                int tmp = V[i];
                V[i] = V[j];
                V[j] = tmp;
            }
        }
    }

    printf("Processo %d ha il vettore ordinato:\n", rank);
    print_vector(V, DIM);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank % 2 == 0 && rank != 0)
    {
        int local_max = V[0];
        int local_min = V[DIM - 1];

        MPI_Send(&local_max, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&local_min, 1, MPI_INT, 1, 3, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        int global_max = V[0];

        for (int i = 2; i < size; i += 2)
        {
            int local_max;
            MPI_Recv(&local_max, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (local_max > global_max)
            {
                global_max = local_max;
            }
        }

        printf("Processo 0 ha il massimo globale: %d\n", global_max);
    }

    if (rank == 1)
    {
        int global_in = V[DIM - 1];

        for (int i = 2; i < size; i += 2)
        {
            int local_min;
            MPI_Recv(&local_min, 1, MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (local_min < global_in)
            {
                global_in = local_min;
            }
        }

        printf("Processo 1 ha il minimo globale: %d\n", global_in);
    }

    // secondo approccio con even comm, i processi in even comm fanno la reduce, poi 0 manda a 1 il min
    // MPI_Comm even_comm;
    // MPI_Comm_split(MPI_COMM_WORLD, (rank % 2 == 0) ? 0 : MPI_UNDEFINED, rank, &even_comm);

    // int local_max = V[0];
    // int local_min = V[dim - 1];
    // int global_max, global_min;

    // // Tutti i processi pari partecipano
    // if (rank % 2 == 0)
    // {
    //     MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, even_comm);
    //     MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, even_comm);
    // }

    // // Poi il processo 0 deve stampare il max
    // if (rank == 0)
    // {
    //     printf("Processo 0 ha il massimo globale: %d\n", global_max);
    // }
    // // e il processo 1 deve stampare il min → lo mandiamo da 0 a 1
    // if (rank == 0)
    // {
    //     MPI_Send(&global_min, 1, MPI_INT, 1, 77, MPI_COMM_WORLD);
    // }
    // if (rank == 1)
    // {
    //     MPI_Recv(&global_min, 1, MPI_INT, 0, 77, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     printf("Processo 1 ha il minimo globale: %d\n", global_min);
    // }

    printf("\n");
    free(diag);
    MPI_Finalize();
    return 0;
}