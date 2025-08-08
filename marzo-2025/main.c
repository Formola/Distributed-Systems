/*
Realizzare un programma mpi C con nproc=nxn che effettui:

- realizzare una topologia bidimensionale nxn.

- i processi della prima colonna leggano da file diversi
  un vettore di DIM interi e lo inviano a tutti i processi della propria riga.

- i processi effettuano uno shift facendo scorrere
  verso l'alto sulla colonna il proprio vettore.

- a questo punto, viene calcolato il vettore dei massimi per ogni riga.
  sta scritto male!.

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

void print_vector(int *v, int dim)
{

    for (int i = 0; i < dim; i++)
    {
        printf("%d ", v[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 3;

    if (size == (int)sqrt(size) * (int)sqrt(size))
    {
        n = (int)sqrt(size);
    }
    else
    {
        perror("Wrong size.\n");
        return 1;
    }

    MPI_Comm topology;
    int dims[2] = {n, n};
    int periods[2] = {1, 1};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &topology);

    int coords[2];
    MPI_Cart_coords(topology, rank, 2, coords);

    MPI_Comm row_comm;
    MPI_Comm_split(topology, coords[0], rank, &row_comm);
    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);

    printf("Process %d is at position (%d, %d) with rank %d and row_rank %d\n", rank, coords[0], coords[1], rank, row_rank);
    fflush(stdout);

    // processi sulla prima colonna leggono da file

    char *filenames[3] = {"input1.txt", "input2.txt", "input3.txt"};

    int dim;
    int *V = malloc(dim * sizeof(int));

    if (coords[1] == 0)
    {
        printf("Process %d is reading from file\n", rank);

        FILE *file;
        const char *filename = filenames[coords[0]];

        file = fopen(filename, "r");

        if (file == NULL)
        {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read the dimension of the vector
        fscanf(file, "%d", &dim);

        // read vector

        for (int i = 0; i < dim; i++)
        {
            fscanf(file, "%d", &V[i]);
        }

        fclose(file);
    }

    // Tutti nella riga ricevono la dimensione
    MPI_Bcast(&dim, 1, MPI_INT, 0, row_comm);

    // tutti ricevono il vettore
    MPI_Bcast(V, dim, MPI_INT, 0, row_comm);

    printf("Process %d has vector: ", rank);
    print_vector(V, dim);

    // i processi effettuano uno shift verso l'alto

    int source_rank, dest_rank;
    MPI_Cart_shift(topology, 0, -1, &source_rank, &dest_rank);
    printf("[Rank %d, coords (%d,%d)] Shift su colonna: ricevo da %d, invio a %d\n", rank, coords[0], coords[1], source_rank, dest_rank);

    // i processi inviano il proprio vettore al processo sopra e lo ricevono da quello sotto
    int *rcv_v = malloc(dim * sizeof(int));
    MPI_Status status;

    MPI_Sendrecv(V, dim, MPI_INT, dest_rank, 0, rcv_v, dim, MPI_INT, source_rank, 0, topology, &status);
    printf("[Rank %d, coords (%d,%d)] Dopo lo shift il vettore ricevuto è: ", rank, coords[0], coords[1]);
    print_vector(rcv_v, dim);

    // rcv_v deve sovrascirivere V

    for (int i = 0; i < dim; i++)
    {
        V[i] = rcv_v[i];
    }
    free(rcv_v);

    // calcolo del vettore dei massimi per ogni riga
    // int *max_vector = malloc(dim * sizeof(int));

    // MPI_Allreduce(V, max_vector, dim, MPI_INT, MPI_MAX, row_comm);

    // printf("[Rank %d, coords (%d,%d)] Vettore dei massimi per la riga: ", rank, coords[0], coords[1]);
    // print_vector(max_vector, dim);

    // free(max_vector);

    int max = -9999;

    if (row_rank == 0)
    {
        for (int i = 0; i < dim; i++)
        {
            if (V[i] > max)
            {
                max = V[i];
            }
        }
        printf("Max for row %d is %d\n", coords[0], max);
    }

    // send max to all processes in the row
    MPI_Bcast(&max, 1, MPI_INT, 0, row_comm);

    // costruire ìl vettore con allgather
    int *max_vector = malloc(n * sizeof(int));
    MPI_Allgather(&max, 1, MPI_INT, max_vector, 1, MPI_INT, row_comm);

    printf("[Rank %d, coords (%d,%d)] Vettore dei massimi per la riga: ", rank, coords[0], coords[1]);
    print_vector(max_vector, n);
    free(max_vector);

    printf("\n");
    MPI_Finalize();
    return 0;
}