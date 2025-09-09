/*
Realizzare in mpi C un'applicazione con numero di processi nproc in cui:

- il processo di rango 0 legge da file un vettore di interi V(dim)
  e ne distribuisce in round-robin agli altri, compreso se stesso,
  k (dim/nproc) elementi in due blocchi non consecutivi di k/2 elementi
  considerando l'ipotesi di k pari.

- ogni processo, memorizza gli elementi ricevuti in una matrice
  in cui ogni blocco ricevuto diventa una colonna di una
  matrice A[k/2][2]

- viene creata una topologia a matrice con 2 righe e nproc/2 colonne
  Se nproc è dispari, va eliminato dal gruppo il processo col
  valore minimo in A[0][0].

- usando la topologia, i processi della prima riga inviano
  il loro valore di A[0][0] ai corrispondenti processi della seconda
  riga che lo dovranno sostituire nella propria matrice.

*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define FILENAME "vector.txt"

#define DIM 20
// se vuoi testare se funziona in caso siano gia in numero pari metti dim=24 e 4 proc
// (caso in cui non si fa excl)

void print_vector(int *v, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("%d ", v[i]);
    }
    printf("\n");
}

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
    printf("\n");
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = DIM;

    int V[dim];

    if (rank == 0)
    {
        // lettura del vettore da file
        FILE *fp = fopen(FILENAME, "r");
        if (fp == NULL)
        {
            printf("Errore nell'apertura del file %s\n", FILENAME);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < dim; i++)
        {
            fscanf(fp, "%d", &V[i]);
        }
        fclose(fp);

        printf("Vettore letto da file:\n");
        print_vector(V, dim);
    }

    if (dim % (2 * size) != 0)
    {
        if (rank == 0)
        {
            printf("Il dimensione del vettore deve essere multiplo di 2*nproc\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int k = dim / size;                   // numero di elementi per processo
    int blocks_per_proc = 2;              // numero di blocchi per processo
    int block_size = k / blocks_per_proc; // dimensione di ogni blocco

    MPI_Datatype block_type;
    MPI_Type_vector(block_size, 1, 1, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);

    if (rank == 0)
    {

        for (int p = 0; p < size; p++)
        {

            for (int b = 0; b < blocks_per_proc; b++)
            {

                MPI_Send(&V[p * block_size + b * block_size * size], 1, block_type, p, 0, MPI_COMM_WORLD);
            }
        }
    }

    // tutti, compreso p0, devono riceve i blocchi ma ogni blocco deve diventare una colonna
    int A[block_size][blocks_per_proc]; // matrice locale

    MPI_Datatype col_type;
    MPI_Type_vector(block_size, 1, blocks_per_proc, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    for (int b = 0; b < blocks_per_proc; b++)
    {
        MPI_Recv(&A[0][b], 1, col_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("Processo %d, matrice A ricevuta:\n", rank);
    print_matrix(&A[0][0], block_size, blocks_per_proc);
    fflush(stdout);

    // creazione della topologia a matrice 2 x (size/2)
    // se size è dispari, escludo il processo con il valore minimo in A[0][0]

    typedef struct
    {
        int value;
        int rank;
    } minloc_t;

    minloc_t local_min, global_min;

    local_min.value = A[0][0];
    local_min.rank = rank;

    MPI_Allreduce(&local_min, &global_min, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    if (rank == 0){
        printf("Valore minimo in A[0][0] è %d trovato dal processo %d\n", global_min.value, global_min.rank);
        fflush(stdout);
    }

    MPI_Group world_group, new_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    MPI_Comm new_comm;
    if (size %2 != 0){

        int ranks_to_exclude[1] = {global_min.rank};

        if (rank != global_min.rank) {
            MPI_Group_excl(world_group, 1, ranks_to_exclude, &new_group);
        } else {
            new_group = MPI_GROUP_EMPTY; // processo escluso: gruppo vuoto
        }
    } else {
        new_group = world_group; // tutti rimangono
    }

    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    if (new_comm != MPI_COMM_NULL)
    {
        int new_rank, new_size;
        MPI_Comm_rank(new_comm, &new_rank);
        MPI_Comm_size(new_comm, &new_size);

        int ndims = 2;
        int dims[2] = {2, new_size / 2};
        int periods[2] = {1, 1};

        MPI_Comm topology;
        MPI_Cart_create(new_comm, ndims, dims, periods, 0, &topology);

        int coords[2];
        MPI_Cart_coords(topology, new_rank, ndims, coords);
        printf("Processo %d ha coordinate nella topologia: [%d, %d]\n",
               rank, coords[0], coords[1]);
        fflush(stdout);

        // Scambio tra righe usando MPI_Cart_shift
        int src, dest;
        // spostamento lungo la dimensione 0 (righe), 1 passo
        MPI_Cart_shift(topology, 0, 1, &src, &dest);

        if (coords[0] == 0)
        {
            // riga 0 invia A[0][0] alla riga 1 nella stessa colonna
            MPI_Send(&A[0][0], 1, MPI_INT, dest, 0, topology);
            printf("Processo %d (riga 0 - new_rank = %d) ha inviato %d al processo %d (riga 1)\n", rank, new_rank, A[0][0], dest);
            fflush(stdout);
        }
        else if (coords[0] == 1)
        {
            // riga 1 riceve da riga 0
            int recv_value;
            MPI_Recv(&recv_value, 1, MPI_INT, src, 0, topology, MPI_STATUS_IGNORE);
            A[0][0] = recv_value;
            printf("Processo %d (riga 1 - new_rank = %d) ha ricevuto %d da processo %d e aggiornato A[0][0]\n", rank, new_rank, recv_value, src);
            print_matrix(&A[0][0], block_size, blocks_per_proc);
            fflush(stdout);
        }

        MPI_Comm_free(&topology);
    }

    MPI_Type_free(&col_type);
    MPI_Type_free(&block_type);
    printf("\n");
    MPI_Finalize();
    return 0;
}
