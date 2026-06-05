#include <iostream>
#include <fmt/core.h>
#include <mpi.h>
#include <vector>

#define MATRIX_DIM 25

void imprimir_matriz(const std::vector<double>& A, int rows, int cols){
    fmt::print("Matriz A local:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fmt::print("{:.2f} ", A[i*cols + j]);
        }
        fmt::print("\n");
    }
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int nprocs;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        std::vector<double> A(MATRIX_DIM * MATRIX_DIM);
        std::vector<double> b(MATRIX_DIM);
        std::vector<double> x(MATRIX_DIM);

        //inicializar la matriz A y el vector b

        for (int i =0; i<MATRIX_DIM; i++){
            for(int j=0; j<MATRIX_DIM; j++){
                int index = i*MATRIX_DIM + j;
                A[index] = i;
            }
        }

        for(int i =0; i<MATRIX_DIM; i++){
            b[i] = 1;
        }

        // nmero de filas para cada RANK (proceso)
        int rows_per_rank = std::ceil(MATRIX_DIM * 1.0 / nprocs);
        int padding = rows_per_rank * nprocs - MATRIX_DIM;

        fmt::print("MATRIX_DIM: {}, nprocs: {}, rows_per_rank: {}, padding: {}\n",
                   MATRIX_DIM, nprocs, rows_per_rank, padding);

        //enviar dimensiones y datos
        for (int i = 1; i < nprocs; i++)
        {
            int filas = rows_per_rank;
            if (i == nprocs - 1)
            {
                filas = rows_per_rank - padding;
            }
            //enviar dimension
            std::vector<int> data = {MATRIX_DIM, filas};

            MPI_Send(
                data.data(),
                2, // data.size()
                MPI_INT,
                i,
                0,
                MPI_COMM_WORLD
            );

            const double* buffer = A.data();
            MPI_Send(
                &buffer[i*rows_per_rank*MATRIX_DIM],
                filas*MATRIX_DIM, // data.size()
                MPI_DOUBLE,
                i,
                0,
                MPI_COMM_WORLD
            );
        }
        fmt::print("RANK: {}, {} x {} \n", rank, rows_per_rank, MATRIX_DIM);
    }
    else
    {
        std::vector<int> data_rec(2);
        MPI_Recv(
            data_rec.data(),
            2, // data.size()
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        int matrix_dim = data_rec[0];
        int rows = data_rec[1];

        fmt::print("RANK: {}, {} x {} \n", rank, rows, matrix_dim);
        
        std::vector<double> A_local(rows*matrix_dim);

        MPI_Recv(
            A_local.data(),
            rows*matrix_dim, // data.size()
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
            
            if(rank=2) {
                imprimir_matriz(A_local, rows, matrix_dim);
            }

    }
    MPI_Finalize();

    return 0;
}