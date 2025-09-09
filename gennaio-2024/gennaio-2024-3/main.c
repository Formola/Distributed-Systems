/*
Scrivere in C un programma MPI con nproc processi che effettuano:

- il processo di rango 0 legge da file un vettore di interi V(dim)
  e lo suddivide tra i processi in round robin.
  Ogni processo riceve k = dim / nproc elementi,
  organizzati in una matrice A[2][k/2], cioè due righe di k/2 elementi ciascuna.

- ogni processo ordina in ordine crescente ciascuna delle due righe di A,
  poi confronta il primo elemento della prima riga con quello della seconda riga
  e assegna a sé stesso un colore: 0 se il primo elemento della prima riga è minore,
  1 altrimenti.

- i processi si suddividono in due gruppi in base al colore assegnato.
  Ogni processo seleziona il vettore corrispondente alla riga di A associata al proprio colore
  (prima riga per colore 0, seconda riga per colore 1).

- all’interno di ogni gruppo viene effettuata una riduzione collettiva su questi vettori:
  il gruppo 0 somma elemento per elemento i vettori,
  il gruppo 1 moltiplica elemento per elemento i vettori.
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

    int V[dim];

    if (rank == 0)
    {
        FILE *fp = fopen(FILE_NAME, "r");
        if (fp == NULL)
        {
            perror("Errore nell'apertura del file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < dim; i++)
        {
            fscanf(fp, "%d", &V[i]);
        }
        fclose(fp);

        printf("Vettore letto dal file:\n");
        print_vector(V, dim);
        fflush(stdout);
    }

    int k = dim / size; // elementi per processo

    if (dim % (k * size) != 0)
    {
        perror("dim non divisibile per nproc");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int A[2][k / 2];

    int rows_per_proc = 2;

    MPI_Datatype rows_type;
    MPI_Type_vector(rows_per_proc, k / 2, (k / 2) * size, MPI_INT, &rows_type);
    MPI_Type_commit(&rows_type);

    if (rank == 0)
    {
        for (int p = 0; p < size; p++)
        {
            MPI_Send(&V[p * (k / 2)], 1, rows_type, p, 0, MPI_COMM_WORLD);
        }
    }

    // tutti compreso p0 ricevono
    MPI_Recv(A, 2 * k / 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Processo %d ha ricevuto la matrice A:\n", rank);
    print_matrix(2, k / 2, A);
    fflush(stdout);

    // ogni processo ordina in ordine crescente ciascuna delle due righe di A,

    for (int i = 0; i < 2; i++)
    {

        for (int j = 0; j < (k / 2) - 1; j++)
        {
            for (int l = j + 1; l < k / 2; l++)
            {
                if (A[i][j] > A[i][l])
                {
                    int temp = A[i][j];
                    A[i][j] = A[i][l];
                    A[i][l] = temp;
                }
            }
        }
    }

    printf("Processo %d ha ordinato la matrice A:\n", rank);
    print_matrix(2, k / 2, A);
    fflush(stdout);

    // confronta il primo elemento della prima riga con quello della seconda riga
    // e assegna a sé stesso un colore : 0 se il primo elemento della prima riga è minore,
    // 1 altrimenti.

    int color;

    if (A[0][0] < A[1][0])
    {
        color = 0;
    }
    else
    {
        color = 1;
    }

    // i processi si suddividono in due gruppi in base al colore assegnato
    // ogni processo seleziona il vettore corrispondente alla riga di A
    // associata al proprio colore(prima riga per colore 0, seconda riga per colore 1).

    MPI_Comm color_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &color_comm);

    int selected_row[k / 2];

    for (int i = 0; i < k / 2; i++)
    {
        selected_row[i] = A[color][i];
    }

    printf("Processo %d con color %d ha selezionato la riga di A:\n", rank, color);
    print_vector(selected_row, k / 2);
    fflush(stdout);

    // all’interno di ogni gruppo viene effettuata una riduzione collettiva su questi vettori:
    // il gruppo 0 somma elemento per elemento i vettori,
    // il gruppo 1 moltiplica elemento per elemento i vettori.

    int result[k / 2];

    if (color == 0){
        MPI_Allreduce(selected_row, result, k/2, MPI_INT, MPI_SUM, color_comm);

        printf("Processo %d con color %d ha il risultato della somma:\n", rank, color);
        print_vector(result, k / 2);
        fflush(stdout);
    } else if (color == 1){
        MPI_Allreduce(selected_row, result, k/2, MPI_INT, MPI_PROD, color_comm);

        printf("Processo %d con color %d ha il risultato della moltiplicazione:\n", rank, color);
        print_vector(result, k / 2);
        fflush(stdout);
    }

    printf("\n");
    MPI_Type_free(&rows_type);
    MPI_Finalize();
    return 0;
}
