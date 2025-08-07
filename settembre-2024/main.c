/*
Realizzare in MPI C un programma in cui:

- il processo di rango 0 legge da file un vettore
  di interi A[DIM] e ne distribuisce a tutti i
  processi compreso se stesso blocchi di k elementi consecutivi
  in modalità round-robin. DIM si suppone multiplo di k*nproc.

- il singolo processo, ordina in senso crescente il proprio vettore V.

- il processo 0 raccoglie in un vettore i 5 valori più grandi
  diversi tra gli elementi dell'intero vettore attraverso
  ripetute operazioni di calcolo collettive.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define FILE_NAME "input.txt"

int *read_vector_from_file(const char *filename, int *dim)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        return NULL;
    }
    fscanf(file, "%d", dim);
    int *vector = malloc(*dim * sizeof(int));
    for (int i = 0; i < *dim; i++)
    {
        fscanf(file, "%d", &vector[i]);
    }
    fclose(file);
    return vector;
}

void print_vector(int *vector, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        printf("%d ", vector[i]);
    }
    printf("\n");
}

void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void sort_vector(int *vector, int dim)
{
    for (int i = 0; i < dim - 1; i++)
    {
        for (int j = 0; j < dim - i - 1; j++)
        {
            if (vector[j] > vector[j + 1])
            {
                swap(&vector[j], &vector[j + 1]);
            }
        }
    }
}

// Cerca il massimo locale escludendo elementi segnati da excluded_flags
// e valori contenuti in common_values_to_exclude di dimensione exclude_count
int find_max_element_in_array(int *array, int dim, int *excluded_flags, int *excluded_values)
{
    int max_val = -1; 

    for (int i = 0; i < dim; i++)
    {
        if (excluded_flags[i]) // == 1 significa che il valore è da escludere
            continue;

        int skip = 0;
        for (int j = 0; j < 5; j++) // 5 è il numero fisso di valori da escludere
        {
            if (array[i] == excluded_values[j])
            {
                skip = 1;
                break;
            }
        }
        if (skip) // == 1 significa che il valore è da escludere
            continue;

        if (array[i] > max_val)
        {
            max_val = array[i];
        }
    }

    return max_val;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dim;
    int *A = NULL;

    int k = 4; // elementi per processo

    if (rank == 0)
    {
        A = read_vector_from_file(FILE_NAME, &dim);
        if (dim % (k * size) != 0)
        {
            fprintf(stderr, "Dimensione vettore non compatibile\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *my_vector = malloc(k * sizeof(int));

    MPI_Scatter(A, k, MPI_INT, my_vector, k, MPI_INT, 0, MPI_COMM_WORLD);
        
    printf("Process %d received vector: ", rank);
    print_vector(my_vector, k);


    sort_vector(my_vector, k);
    printf("Process %d sorted vector: ", rank);
    print_vector(my_vector, k);
    

    int *top_5_values = malloc(5 * sizeof(int));
    if (rank == 0) {
        // Inizializza top_5_values con valori impossibili
        for (int i = 0; i < 5; i++)
        {
            top_5_values[i] = -1;
        }
    }

    int excluded_flags[k];
    for (int i = 0; i < k; i++)
    {
        excluded_flags[i] = 0;
    }


    int saved = 0;

    for (int iter = 0; iter < 5; iter++)
    {
        int local_max = find_max_element_in_array(my_vector, k, excluded_flags, top_5_values);
        int global_max;

        // MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        // MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0)
        {
            int already_present = 0;
            for (int i = 0; i < saved; i++)
            {
                if (top_5_values[i] == global_max)
                {
                    already_present = 1;
                    break;
                }
            }

            if (!already_present)
            {
                top_5_values[saved] = global_max;
                saved++;
                printf("Saved new value: %d\n", global_max);
            }
            else
            {
                printf("Duplicate value skipped: %d\n", global_max);
                // Se è duplicato, decrementa iter per ripetere il passo e prendere un altro massimo
                iter--;
            }
        }

        // Broadcast per aggiornare top_5_values a tutti, dato che solo il processo 0 lo aggiorna.
        // Tutti i processi devono avere la stessa visione dei top 5 valori, perche devono usarlo per escludere i valori nei prossimi passi.
        MPI_Bcast(top_5_values, 5, MPI_INT, 0, MPI_COMM_WORLD);

        // Escludi localmente il valore trovato (anche se duplicato per evitare loop)
        for (int j = 0; j < k; j++)
        {
            if (my_vector[j] == global_max)
            {
                excluded_flags[j] = 1;
            }
        }
    }

    if (rank == 0)
    {
        printf("Top 5 values: ");
        for (int i = 0; i < 5; i++)
        {
            printf("%d ", top_5_values[i]);
        }
        free(A);
        free(top_5_values);
    }

    printf("\n");
    free(my_vector);
    MPI_Finalize();
    return 0;
}
