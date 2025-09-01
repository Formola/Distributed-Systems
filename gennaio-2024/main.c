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
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < dim; i++)
        {
            fscanf(fp, "%d", &V[i]);
        }

        printf("Proc %d ha letto da file:\n ", rank);
        print_vector(V, dim);

        fclose(fp);
    }

    int k = dim / size;
    int block_size = k / 2;
    int num_blocks = 2;

    // datatype per round robin di blocchi. mandiamo tutti i blocchi insieme in una sola send.
    MPI_Datatype blocks_type;
    MPI_Type_vector(num_blocks, block_size, size * block_size, MPI_INT, &blocks_type);
    MPI_Type_commit(&blocks_type);

    int A[2][block_size]; // tutti salvano qui i blocchi ricevuti

    // proc0 fa la distribuzione
    if (rank == 0)
    {

        for (int i = 0; i < size; i++)
        {
            MPI_Send(&V[i * block_size], 1, blocks_type, i, 0, MPI_COMM_WORLD);
        }
    }

    // tutti, compreso 0, ricevono e organizzano in A come richiesto
    // la matrice è memorizzata per righe quindi per come riceviamo i dati
    // dal datatype, non ce ne serve un altro per riorganizzarlo, arrivano
    // già consecutivi i blocchi.
    MPI_Recv(A, num_blocks * block_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("\nProc %d ha ricevuto:\n", rank);
    print_matrix(2, block_size, A);
    fflush(stdout);

    // ogni processo ordina le righe in senso crescente
    for (int i = 0; i < 2; i++)
    { // per ogni riga
        for (int j = 0; j < block_size - 1; j++)
        { // per ogni elemento della colonna tranne l'ultimo
            for (int k = j + 1; k < block_size; k++)
            { // scorre tutta la colonna

                if (A[i][j] > A[i][k])
                {

                    // swap
                    int temp = A[i][j];
                    A[i][j] = A[i][k];
                    A[i][k] = temp;
                }
            }
        }
    }

    printf("Proc %d ha ordinato:\n", rank);
    print_matrix(2, block_size, A);
    fflush(stdout);

    // ogni processo controlla il primo elemento della prima riga
    // con il primo elemento della seconda riga, e assegna il colore 0
    // se quello della prima riga è minore, 1 altrimenti
    int color;

    if (A[0][0] < A[1][0])
    {
        color = 0;
    }
    else
    {
        color = 1;
    }

    printf("\nProc %d ha assegnato il colore: %d (%s)\n",
           rank, color, color == 0 ? "Gruppo SOMMA" : "Gruppo MOLTIPLICAZIONE");
    fflush(stdout);

    // i processi si suddivisono in due gruppi in base al colore
    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
    int new_rank;
    MPI_Comm_rank(new_comm, &new_rank);

    int my_vector[block_size];
    for (int i = 0; i < block_size; i++)
    {
        my_vector[i] = A[color][i];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // all’interno di ogni gruppo viene effettuata una riduzione collettiva
    // su questi vettori:
    // il gruppo 0 somma elemento per elemento i vettori,
    // il gruppo 1 moltiplica elemento per elemento i vettori.

    int result_vector[block_size];

    if (color == 0)
    {
        printf("\nProc %d sta effettuando una RIDUZIONE con SOMMA, vettore locale: ", rank);
        print_vector(my_vector, block_size);
        fflush(stdout);

        MPI_Allreduce(my_vector, result_vector, block_size, MPI_INT, MPI_SUM, new_comm);
    }
    else
    {
        printf("\nProc %d sta effettuando una RIDUZIONE con MOLTIPLICAZIONE, vettore locale: ", rank);
        print_vector(my_vector, block_size);
        fflush(stdout);

        MPI_Allreduce(my_vector, result_vector, block_size, MPI_INT, MPI_PROD, new_comm);
    }

    // si poteva anche fare con solo reduce e poi solo new_rank == 0 stampava vabbe
    if (new_rank == 0)
    {
        printf("\n>>> [GRUPPO %s] Risultato finale stampato dal processo %d:\n",
               color == 0 ? "SOMMA" : "MOLTIPLICAZIONE", rank);
        print_vector(result_vector, block_size);
        fflush(stdout);
    }

    printf("\n");
    MPI_Type_free(&blocks_type);
    MPI_Finalize();
    return 0;
}
