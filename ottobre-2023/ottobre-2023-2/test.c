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

        for (int i = 0; i < DIM; i++)
        {
            for (int j = 0; j < DIM; j++)
            {
                fscanf(fp, "%d", &A[i][j]);
            }
        }
        fclose(fp);

        printf("Matrix A:\n");
        print_matrix(DIM, DIM, A);
        fflush(stdout);
    }

    int k = dim / size; // colonne da distribuire a ogni processo. es. k=4

    int mat[k][dim]; // ognuno dovrebbe ricevere mat[dim][k] e ne fa la trasposta quindi mat[k][dim]
    // salviamo direttamente la trasposta che possiamo ottenere facilmente
    // se utilizziamo correttamente il datatype di send delle colonne

    // singola colonna
    MPI_Datatype column_type;
    MPI_Type_vector(dim, 1, dim, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);

    // si poteva fare anche senza questo, ci voleva una send per ogni colonna
    // calcolando l'indice della colonna da mandare: index = proc + i_col*size
    MPI_Datatype columns_type;
    MPI_Aint stride = size * sizeof(int);
    MPI_Type_hvector(k, 1, stride, column_type, &columns_type);
    MPI_Type_commit(&columns_type);

    // non è chiaro se la traccia vuole che in ricezione venga salvata
    // prima come [dim][k] quindi per COLONNE e poi fare la trasposta a mano SAREBBE CRAZY
    // in tal caso ci vuole anche un datatype di recv e poi calcolo trasposta manuale.

    // MPI_Datatype col_recv_type;
    // MPI_Type_vector(dim, 1, k, MPI_INT, &col_recv_type);
    // MPI_Type_commit(&col_recv_type);

    // MPI_Datatype recv_type;
    // MPI_Type_hvector(k, 1, sizeof(int), col_recv_type, &recv_type);
    // MPI_Type_commit(&recv_type);

    if (rank == 0)
    {

        // proc0 farà le send di tutte le colonne a ogni proc

        for (int p = 0; p < size; p++)
        {
            MPI_Send(&A[0][p], 1, columns_type, p, 0, MPI_COMM_WORLD);
        }
    }

    // ognuno compreso proc0 riceve la matrice praticamente già trasposta [k][dim]
    MPI_Recv(&mat, k * dim, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // MPI_Recv(&mat[0][0], 1, recv_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // printf("\nProcess %d received matrix:\n", rank);
    // print_matrix(k, dim, mat);
    // fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    // stampa sequenziale per chiarezza
    for (int p = 0; p < size; p++)
    {
        if (rank == p)
        {
            printf("\n=== Process %d received matrix (k=%d x dim=%d) ===\n", rank, k, dim);
            print_matrix(k, dim, mat);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ogni processo costruisce un vettore v di dimensione dim,
    // in cui ogni elemento è il massimo dei valori corrispondenti nelle righe di questa matrice

    int v_max[dim];
    for (int j = 0; j < dim; j++)
    {
        v_max[j] = mat[0][j];
        for (int i = 1; i < k; i++)
        {
            if (mat[i][j] > v_max[j])
            {
                v_max[j] = mat[i][j];
            }
        }
    }
    printf("\nProcess %d - Vector of max from rows:\n", rank);
    print_vector(v_max, dim);
    fflush(stdout);

    // tutti i vettori v vengono raccolti da tutti i processi tramite una comunicazione collettiva.

    int all_v_max[dim * size];
    MPI_Allgather(v_max, dim, MPI_INT, all_v_max, dim, MPI_INT, MPI_COMM_WORLD);

    // ciascun processo calcola la media di tutti gli elementi di tutti i vettori raccolti.

    float avg = 0.0;
    for (int i = 0; i < dim * size; i++)
    {
        avg += all_v_max[i];
    }

    avg /= (dim * size);
    printf("Process %d - Average of all max values: %.2f\n", rank, avg);
    fflush(stdout);

    // ogni processo si assegna un colore in base al confronto tra il proprio vettore v e la media:
    // se almeno un elemento di v è minore della media, il processo assume colore 1, altrimenti colore 0.

    int color = 0;

    for (int i = 0; i < dim; i++)
    {
        if (v_max[i] < avg)
        {
            color = 1;
            break;
        }
    }

    // stampa chiara prima della divisione in gruppi
    printf("Process %d - Vector v_max: ", rank);
    print_vector(v_max, dim);
    printf("Process %d - Average: %.2f\n", rank, avg);
    printf("Process %d - Assigned color: %d\n", rank, color);
    fflush(stdout);

    // divisione in gruppi in base a color
    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
    int new_rank, new_size;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);

    // riduzione all'interno del gruppo
    int result[dim];
    if (color == 0)
    {
        MPI_Allreduce(v_max, result, dim, MPI_INT, MPI_SUM, new_comm);
    }
    else
    {
        MPI_Allreduce(v_max, result, dim, MPI_INT, MPI_PROD, new_comm);
    }

    // stampa chiara dei risultati di gruppo
    if (new_rank == 0)
    {
        printf("\n=== Group with color %d ===\n", color);
        printf("Number of processes in this group: %d\n", new_size);
        printf("Result vector after reduction: ");
        print_vector(result, dim);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    printf("\n");
    MPI_Type_free(&column_type);
    MPI_Type_free(&columns_type);
    MPI_Finalize();
    return 0;
}
