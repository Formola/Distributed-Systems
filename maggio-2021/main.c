/*
Realizzare in mpi C un'applicazione con numero di processi nproc (nxn) in cui:

- realizzare una topologia bidimensionale

- i processi della diagonale principale leggono da file un vettore di interi
  di dimensione DIM

- i processi della diag fanno bcast del vettore ai proc sulla propria riga

- ogni processo manda il proprio vettore al processo
  che sta sopra di lui sulla stessa colonna

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define DIM 4

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
    int dim = DIM;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = (int)sqrt(size);
    if (n * n != size)
    {
        if (rank == 0)
        {
            printf("Errore: il numero di processi deve essere un quadrato perfetto\n");
        }
        MPI_Finalize();
        return -1;
    }

    MPI_Comm topology;
    int ndims = 2;
    int dims[2] = {n, n};
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &topology);

    int coords[2];
    MPI_Cart_coords(topology, rank, ndims, coords);

    printf("Rank %d ha coordinate (%d, %d)\n", rank, coords[0], coords[1]);

    // supponiamo 3x3, quindi dobbiamo leggere tre file

    int V[dim];

    if (coords[0] == coords[1])
    {

        FILE *fp;
        char filename[20];
        sprintf(filename, "vector_%d.txt", coords[0]);
        fp = fopen(filename, "r");
        if (fp == NULL)
        {
            perror("Errore apertura file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < dim; i++)
        {
            fscanf(fp, "%d", &V[i]);
        }
        fclose(fp);
    }

    MPI_Comm row_comm;
    MPI_Cart_sub(topology, (int[]){0,1}, &row_comm);

    // Nella riga i, il root è il processo (i,i): proiettato sul comm di riga ha coordinata 1D = i.
    int root_col_coord = coords[0]; // colonna = indice di riga i
    int root_row_rank;
    // Il row_comm è 1D: calcola il rank dato coords[1D] = root_col_coord
    MPI_Cart_rank(row_comm, &root_col_coord, &root_row_rank);

    MPI_Bcast(V, dim, MPI_INT, root_row_rank, row_comm);
    printf("Processo %d ha ricevuto il vettore: ", rank);
    print_vector(V, dim);
    fflush(stdout);

    // ogni processo manda il proprio vettore a quello che sta sopra di lui in colonna
    int source, dest;
    MPI_Cart_shift(topology, 0, -1, &source, &dest); // spostamento di -1 sulla dimensione 0 (colonne)

    int recv_V[DIM];

    MPI_Sendrecv(&V, dim, MPI_INT, dest, 0, &recv_V, dim, MPI_INT, source, 0, topology, MPI_STATUS_IGNORE);

    printf("Processo %d ha ricevuto il vettore dal processo %d: ", rank, source);
    print_vector(recv_V, dim);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
