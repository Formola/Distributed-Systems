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

typedef struct
{
    int value;
    int rank;
} minloc_t;

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

    int dim;
    int *V = NULL;

    if (rank == 0)
    {
        FILE *fp = fopen(FILENAME, "r");
        if (fp == NULL)
        {
            perror("Errore nell'apertura del file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // leggi dim da prima riga file
        fscanf(fp, "%d", &dim);
        if (dim % size != 0)
        {
            fprintf(stderr, "dim deve essere multiplo di nproc\n");
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // leggi vector
        V = malloc(dim * sizeof(int));
        for (int i = 0; i < dim; i++)
        {
            fscanf(fp, "%d", &V[i]);
        }

        printf("Process %d ha letto il vettore:\n", rank);
        print_vector(V, dim);

        fclose(fp);
    }

    // mandiamo dim a tutti
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int k = dim / size;
    // check sia pari
    if (k % 2 != 0)
    {
        if (rank == 0)
            fprintf(stderr, "k deve essere pari\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // processo 0 deve distribuire k elementi in due blocchi non consecutivi di k/2 elementi ciascuno

    int *A = malloc((k / 2) * 2 * sizeof(int));

    MPI_Datatype block;
    MPI_Type_vector(1, k / 2, dim / 2, MPI_INT, &block);
    MPI_Type_commit(&block);

    // invio 2 blocchi separati ma senza datatype di send
    // altrimenti in recv diventa difficile memorizzare i blocchi
    // ricevuti in colonna
    MPI_Datatype col_type;
    MPI_Type_vector(k / 2, 1, 2, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    if (rank == 0)
    {
        for (int p = 0; p < size; p++)
        {
            MPI_Send(&V[p * (k / 2)], 1, block, p, 0, MPI_COMM_WORLD);           // primo blocco
            MPI_Send(&V[p * (k / 2) + dim / 2], 1, block, p, 1, MPI_COMM_WORLD); // secondo blocco
        }
    }

    // ricezione
    MPI_Recv(&A[0], 1, col_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&A[1], 1, col_type, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Processo %d ha ricevuto la matrice A:\n", rank);
    print_matrix(A, k / 2, 2);

    // Trova il valore minimo in A[0][0]
    minloc_t local_min = {A[0], rank};
    minloc_t global_min;

    MPI_Allreduce(&local_min, &global_min, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Il valore minimo in A[0][0] è %d, trovato dal processo %d\n", global_min.value, global_min.rank);
    }

    // se size è dispari, il proc con rank in global_min va eliminato dal gruppo

    // Creiamo il gruppo
    MPI_Group world_group, new_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    MPI_Comm new_comm;
    if (size % 2 != 0)
    {
        int ranks_to_exclude[1] = {global_min.rank};

        // Solo i processi NON esclusi creano il gruppo
        if (rank != global_min.rank)
        {
            MPI_Group_excl(world_group, 1, ranks_to_exclude, &new_group);
        }
        else
        {
            new_group = MPI_GROUP_EMPTY; // Processo escluso: gruppo vuoto
        }
    }
    else
    {
        new_group = world_group; // tutti rimangono
    }

    // Tutti i processi chiamano MPI_Comm_create
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    // Solo i processi con new_comm != MPI_COMM_NULL continuano
    if (new_comm != MPI_COMM_NULL)
    {
        int new_rank, new_size;
        MPI_Comm_rank(new_comm, &new_rank);
        MPI_Comm_size(new_comm, &new_size);

        int ndims = 2;
        int dims[2] = {2, new_size / 2};
        int periods[2] = {0, 0};
        MPI_Comm topology;
        MPI_Cart_create(new_comm, ndims, dims, periods, 0, &topology);

        int coords[2];
        MPI_Cart_coords(topology, new_rank, ndims, coords);
        printf("Processo %d ha coordinate nella topologia: [%d, %d]\n",
               new_rank, coords[0], coords[1]);

        // Scambio tra righe usando MPI_Cart_shift
        int src, dest;
        // spostamento lungo la dimensione 0 (righe), 1 passo
        MPI_Cart_shift(topology, 0, 1, &src, &dest);

        if (coords[0] == 0)
        {
            // riga 0 invia A[0][0] alla riga 1 nella stessa colonna
            MPI_Send(&A[0], 1, MPI_INT, dest, 0, topology);
        }
        else if (coords[0] == 1)
        {
            // riga 1 riceve da riga 0
            int recv_value;
            MPI_Recv(&recv_value, 1, MPI_INT, src, 0, topology, MPI_STATUS_IGNORE);
            A[0] = recv_value;
            printf("Processo %d ha ricevuto %d e aggiornato A[0][0]\n", new_rank, recv_value);
            print_matrix(A, k / 2, 2);
        }
    }

    free(A);
    MPI_Type_free(&col_type);
    MPI_Type_free(&block);
    MPI_Finalize();
    return 0;
}

/*
SPIEGAZIONE DEL DATATYPE SEND:

MPI_Type_vector(2, k/2, dim/2, MPI_INT, &send_type);

- count = 2: Vogliamo 2 blocchi
- blocklength = k/2: Ogni blocco ha k/2 elementi
- stride = dim/2: La distanza tra l'inizio del primo e secondo blocco

ESEMPIO con dim=12, nproc=3, k=4:
V = [1,2,3,4,5,6,7,8,9,10,11,12]

Per processo 0 (start_idx = 0):
- Blocco 1: elementi 1,2 (indici 0,1)
- Blocco 2: elementi 7,8 (indici 6,7)
- stride = dim/2 = 6, quindi 0 + 6 = 6 ✓

Per processo 1 (start_idx = 2):
- Blocco 1: elementi 3,4 (indici 2,3)
- Blocco 2: elementi 9,10 (indici 8,9)
- stride = 6, quindi 2 + 6 = 8 ✓

DATATYPE RECV:
MPI_Type_vector(2, k/2, k/2, MPI_INT, &recv_type);

- count = 2: Riceviamo 2 blocchi
- blocklength = k/2: Ogni blocco ha k/2 elementi
- stride = k/2: Nel buffer locale, i blocchi sono consecutivi

Questo posiziona il secondo blocco subito dopo il primo nel buffer locale.
*/