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
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = DIM;

    int k = dim / size;
    int num_blocchi_per_proc = 2; // fissato dalla traccia
    int block_size = k/2;

    if ( k % 2 != 0){
        perror("k is not even. change dim or size!");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int v[dim];
    int A[k/2][2]; // ognuno deve memorizzare i blocchi ricevuti come colonna. ogni blocco è di k/2 elementi e ognuno riceve 2 blocchi.

    if (rank == 0){
        FILE *fp = fopen(FILENAME, "r");

        if (fp == NULL){
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < dim; i++){
            fscanf(fp, "%d", &v[i]);
        }
        fclose(fp);

        printf("\nVector V:\n");
        print_vector(v, dim);
        fflush(stdout);
    }


    MPI_Datatype block_type;
    MPI_Type_vector(block_size, 1, 1, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);

    if (rank == 0){

        // send 2 blocks to all;

        for (int p = 0; p < size; p++) {

            for (int b = 0; b < num_blocchi_per_proc; b++) {
                MPI_Send(&v[(p*block_size) + b * size * block_size], 1, block_type, p, 0, MPI_COMM_WORLD);
            }
        }

    }

    MPI_Datatype col_type;
    MPI_Type_vector(block_size, 1, num_blocchi_per_proc, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    // ognuno deve ricevere due blocchi in A[k/2][2];
    for (int b = 0; b < num_blocchi_per_proc; b++) {

        MPI_Recv(&A[0][b], 1, col_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("\nProc %d ha ricevuto:\n", rank);
    print_matrix(&A[0][0], k/2, 2);
    fflush(stdout);


    // si crea una topologia a 2 righe e nproc/2 colonne.
    // se nproc è dispari si elimina dal gruppo il processo col min in A[0][0].

    int rank_to_escl = -1;

    if (size % 2 != 0) {

        typedef struct {
            int value;
            int rank;
        } minloc_t;

        minloc_t local_min, global_min;
        local_min.value = A[0][0];
        local_min.rank = rank;

        MPI_Allreduce(&local_min, &global_min, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        rank_to_escl = global_min.rank;

    }

    MPI_Group world_group, new_group;
    MPI_Comm new_comm = MPI_COMM_WORLD;

    if (rank_to_escl != -1){
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Group_excl(world_group, 1, &rank_to_escl, &new_group);
        MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
    }

    int new_rank;
    int new_size;

    int ndims = 2;
    int dims[2] = {2, size/2};
    int periods[2] = {1,1};
    int coords[2];
    MPI_Comm topology;
    
    if (new_comm != MPI_COMM_NULL){
        
        MPI_Comm_rank(new_comm, &new_rank);
        MPI_Comm_size(new_comm, &new_size);

        MPI_Cart_create(new_comm, ndims, dims, periods, 0, &topology);
        MPI_Cart_coords(topology, new_rank, ndims, coords);

        printf("Proc %d (new rank = %d) coordinates in the new topology: [%d, %d]\n", rank, new_rank, coords[0], coords[1]);


        // utilizzando tale topologia
        // i processi della prima riga
        // inviano il proprio A[0][0]
        // ai corrispondenti processi
        // della seconda riga che lo dovranno sostituire.

        int source, dest;
        MPI_Cart_shift(topology, 0, 1, &source, &dest);

        
        if (coords[0] == 0) {

            // send
            // int dest;
            // int dest_coords[2] = {1, coords[1]};
            // MPI_Cart_rank(topology, dest_coords, &dest);
            MPI_Send(&A[0][0], 1, MPI_INT, dest, 0, topology);

            printf("Proc %d ha inviato %d a proc %d\n", rank, A[0][0], dest);
            fflush(stdout);

        } else {

            // recv
            // int source;
            // int source_coords[2] = {0, coords[1]};
            // MPI_Cart_rank(topology, source_coords, &source);
            MPI_Recv(&A[0][0], 1, MPI_INT, source, 0, topology, MPI_STATUS_IGNORE);

            printf("Proc %d ha ricevuto %d da proc %d\n", rank, A[0][0], source);
            fflush(stdout);
        }
    }
    
    printf("\n");
    MPI_Type_free(&col_type);
    MPI_Type_free(&block_type);
    MPI_Finalize();
    return 0;
}

