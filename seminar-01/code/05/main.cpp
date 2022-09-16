#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int number;

    int array[10];

    MPI_Status status;
    if (world_rank == 0) {
        for (int i = 0; i < 10; ++i) {
            array[i] = i;
        }

        MPI_Send (array, 5, MPI_INT, 1, 12, MPI_COMM_WORLD);
        MPI_Send (array, 5, MPI_INT, 1, 12, MPI_COMM_WORLD);
        MPI_Send (array, 5, MPI_INT, 1, 12, MPI_COMM_WORLD);

    } else if (world_rank == 1) {
        MPI_Recv (array, 5, MPI_INT, MPI_ANY_SOURCE, 12, MPI_COMM_WORLD, &status);
        std::cout << "Process 1 received 5 elements from unknown process" << std::endl;

        MPI_Recv (array, 5, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        std::cout << "Process 1 received 5 elements from process 0 with unknown tag" << std::endl;

        MPI_Recv (array, 5, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        std::cout << "Process 1 received 5 elements from unknown process with unknown tag. They are" << std::endl;

        for (int i = 0; i < 5; ++i) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    }
    MPI_Finalize();
	return 0;
}

