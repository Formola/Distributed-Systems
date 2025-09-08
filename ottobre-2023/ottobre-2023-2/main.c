/*
Scrivere in C un programma MPI con nproc processi che effettuano:

- il processo di rango 0 legge da file una matrice quadrata A[dim][dim]
  di interi.

- il vettore delle colonne di A viene suddiviso tra i processi
  in modo che ciascun processo riceva k = dim / nproc colonne,
  non necessariamente consecutive (round robin di colonne).

- ogni processo riceve una sottomatrice di dimensione [dim][k]
  e ne costruisce una matrice trasposta [k][dim].

- da questa matrice trasposta, ogni processo costruisce un vettore v di dimensione dim,
  in cui ogni elemento è il massimo dei valori corrispondenti nelle righe di questa matrice.

- tutti i vettori v vengono raccolti da tutti i processi tramite una comunicazione collettiva.

- ciascun processo calcola la media di tutti gli elementi di tutti i vettori raccolti.

- ogni processo si assegna un colore in base al confronto tra il proprio vettore v e la media:
  se almeno un elemento di v è minore della media, il processo assume colore 1, altrimenti colore 0.

- i processi si dividono in due gruppi secondo il colore assegnato.

- all’interno di ciascun gruppo, viene effettuata una riduzione collettiva sui vettori v:
  i processi con colore 0 sommano elemento per elemento i loro vettori,
  quelli con colore 1 moltiplicano elemento per elemento i loro vettori.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILE_NAME "input.txt"
#define DIM 16

void print_matrix(int rows, int cols, int mat[rows][cols])
{

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d\t", mat[i][j]);
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = DIM;

    int A[dim][dim];

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
            for (int j = 0; j < dim; j++)
            {
                fscanf(fp, "%d", &A[i][j]);
            }
        }
        fclose(fp);

        printf("Process %d - Matrix A read from file:\n", rank);
        print_matrix(dim, dim, A);
        printf("\n");
        fflush(stdout);
    }

    // il vettore delle colonne di A viene suddiviso tra i processi
    // in modo che ciascun processo riceva k = dim / nproc colonne,
    // non necessariamente consecutive(round robin di colonne).

    int k = dim / size; // number of columns per process

    int local_A[k][dim]; // each process will receive a submatrix of size [dim][k]

    MPI_Datatype col_type;
    MPI_Type_vector(dim, 1, dim, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    MPI_Datatype cols_type;
    MPI_Aint stride = size * sizeof(int);
    MPI_Type_create_hvector(k, 1, stride, col_type, &cols_type);
    MPI_Type_commit(&cols_type);

    if (rank == 0)
    {

        for (int p = 0; p < size; p++)
        {
            MPI_Send(&A[0][p], 1, cols_type, p, 0, MPI_COMM_WORLD);
        }
    }

    // tutti, compreso proc0, ricevono le loro k colonne

    MPI_Recv(&local_A[0][0], dim * k, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("\nProcess %d - Received local_A:\n", rank);
    print_matrix(k, dim, local_A);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    // -da questa matrice trasposta, ogni processo costruisce un vettore v di dimensione dim,
    // in cui ogni elemento è il massimo dei valori corrispondenti nelle righe di questa matrice.

    int V[dim];

    for (int j = 0; j < dim; j++)
    {

        int max_col = local_A[0][j];
        for (int i = 1; i < k; i++)
        {
            if (local_A[i][j] > max_col)
            {
                max_col = local_A[i][j];
            }
        }
        V[j] = max_col;
    }

    printf("\nProcess %d - Computed vector V:\n", rank);
    print_vector(V, dim);
    fflush(stdout);

    // tutti i vettori v vengono raccolti da tutti i processi tramite una comunicazione collettiva.

    int all_V[size * dim];

    MPI_Allgather(V, dim, MPI_INT, all_V, dim, MPI_INT, MPI_COMM_WORLD);

    // ciascun processo calcola la media di tutti gli elementi di tutti i vettori raccolti.

    int sum = 0;
    float avg = 0.0;

    for (int i = 0; i < size * dim; i++)
    {
        sum += all_V[i];
    }
    avg = (float)sum / (size * dim);

    printf("\nProcess %d - Average of all elements in all V: %.2f\n", rank, avg);
    fflush(stdout);

    // ogni processo si assegna un colore in base al confronto
    // tra il proprio vettore v e la media:
    // se almeno un elemento di v è minore della media,
    // il processo assume colore 1, altrimenti colore 0.

    int color;

    for (int i = 0; i < dim; i++)
    {
        if (V[i] < avg)
        {
            color = 1;
            break;
        }
        else
        {
            color = 0;
        }
    }

    // -i processi si dividono in due gruppi secondo il colore assegnato.

    MPI_Comm color_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &color_comm);

    // all’interno di ciascun gruppo, viene effettuata una riduzione collettiva sui vettori v.
    // i processi con colore 0 sommano elemento per elemento i loro vettori,
    // quelli con colore 1 moltiplicano elemento per elemento i loro vettori.

    int result_V[dim];

    if (color == 0) {

        MPI_Allreduce(V, result_V, dim, MPI_INT, MPI_SUM, color_comm);
        printf("\nProcess %d (color 0) - Result of element-wise SUM in color group:\n", rank);
        print_vector(result_V, dim);
        fflush(stdout);

    } else if (color == 1){

        MPI_Allreduce(V, result_V, dim, MPI_INT, MPI_PROD, color_comm);
        printf("\nProcess %d (color 1) - Result of element-wise PROD in color group:\n", rank);
        print_vector(result_V, dim);
        fflush(stdout);

    }

    printf("\n");
    MPI_Type_free(&col_type);
    MPI_Type_free(&cols_type);
    MPI_Finalize();
    return 0;
}
