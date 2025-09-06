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
    perror("Number of processes must be a perfect square.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  MPI_Comm topology;
  int ndims = 2;
  int dims[2] = {n, n};
  int periods[2] = {1, 1};
  int coords[2];

  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &topology);
  MPI_Cart_coords(topology, rank, ndims, coords);

  printf("Rank %d has coordinates (%d,%d)\n", rank, coords[0], coords[1]);
  fflush(stdout);

  const char *filename[] = {
      "vector_0.txt",
      "vector_1.txt",
      "vector_2.txt"};

  int V[dim];

  if (coords[0] == coords[1])
  {

    // processi sulla diagonale principale
    FILE *fp = fopen(filename[coords[0]], "r");
    if (fp == NULL)
    {
      perror("Error opening file.\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < dim; i++)
    {
      fscanf(fp, "%d", &V[i]);
    }

    fclose(fp);

    printf("Rank %d (coords %d,%d) read vector: ", rank, coords[0], coords[1]);
    print_vector(V, dim);
    fflush(stdout);
  }

  // i processi sulla diag che hanno letto devono fare bcast del vettore

  MPI_Comm row_comm;
  int remain_dims[2] = {0, 1};
  MPI_Cart_sub(topology, remain_dims, &row_comm);

  // o uso direttamente coords[0] in bcast oppure mi calcolo il rank corrispondente nella row
  // i root sono (0,0), (1,1), (2,2) quindi nella loro riga sono quelli con rank 0,1,2 = coords[0].
  int row_root_coord = coords[0];
  int row_root_rank;
  MPI_Cart_rank(row_comm, &row_root_coord, &row_root_rank);

  MPI_Bcast(V, dim, MPI_INT, row_root_rank, row_comm);

  if (coords[0] != coords[1])
  {

    printf("\nRank %d (coords %d,%d) after bcast has vector: ", rank, coords[0], coords[1]);
    print_vector(V, dim);
    fflush(stdout);
  }

  // ogni processo manda il proprio vettore al processo
  // che sta sopra di lui sulla stessa colonna

  int source, dest;

  MPI_Cart_shift(topology, 0, -1, &source, &dest);

  int recv_V[dim];

  MPI_Sendrecv(V, dim, MPI_INT, dest, 0, recv_V, dim, MPI_INT, source, 0, topology, MPI_STATUS_IGNORE);

  printf("\nRank %d (coords %d,%d) received vector from rank %d (coords %d,%d): ", rank, coords[0], coords[1], source, (source / n), (source % n));
  print_vector(recv_V, dim);
  fflush(stdout);

  printf("\n");
  MPI_Finalize();
  return 0;
}
