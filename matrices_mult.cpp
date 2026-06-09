#include <iostream>
#include <vector>
#include <cmath>
#include <fmt/core.h>
#include <mpi.h>

#define MATRIX_DIM 8

void imprimir_vector(const std::vector<double> &v, int size)
{
    for (int i = 0; i < size; i++)
    {
        fmt::print("{:.2f} ", v[i]);
    }
    fmt::print("\n");
}

void multiplicar_matriz_vector(
    const std::vector<double> &A,
    const std::vector<double> &b,
    std::vector<double> &x,
    int rows,
    int cols)
{
    for (int i = 0; i < rows; i++)
    {
        double suma = 0.0;

        for (int j = 0; j < cols; j++)
        {
            suma += A[i * cols + j] * b[j];
        }

        x[i] = suma;
    }
}
void imprimir_matriz(const std::vector<double> &A, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fmt::print("{:5.0f} ", A[i * cols + j]);
        }
        fmt::print("\n");
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Cantidad de filas por proceso
    int rows_per_rank =
        static_cast<int>(std::ceil(MATRIX_DIM * 1.0 / nprocs));

    // Filas totales después del padding
    int padded_rows = rows_per_rank * nprocs;

    // Filas agregadas
    int padding = padded_rows - MATRIX_DIM;

    if (rank == 0)
    {
        fmt::print(
            "MATRIX_DIM: {}, nprocs: {}, rows_per_rank: {}, padded_rows: {}, padding: {}\n",
            MATRIX_DIM,
            nprocs,
            rows_per_rank,
            padded_rows,
            padding);

        // Matriz con padding
        std::vector<double> A(
            padded_rows * MATRIX_DIM,
            0.0);

        std::vector<double> b(MATRIX_DIM);

        // Resultado completo (incluye padding)
        std::vector<double> x(padded_rows);

        // Inicializar matriz real
        for (int i = 0; i < MATRIX_DIM; i++)
        {
            for (int j = 0; j < MATRIX_DIM; j++)
            {
                A[i * MATRIX_DIM + j] = i;
            }
        }
        fmt::print("\nMatriz con padding:\n");
        imprimir_matriz(A, padded_rows, MATRIX_DIM);

        // Vector b
        for (int i = 0; i < MATRIX_DIM; i++)
        {
            b[i] = 1.0;
        }

        for (int proc = 1; proc < nprocs; proc++)
        {
            std::vector<int> metadata =
                {
                    MATRIX_DIM,
                    rows_per_rank};

            MPI_Send(
                metadata.data(),
                2,
                MPI_INT,
                proc,
                0,
                MPI_COMM_WORLD);

            MPI_Send(
                &A[proc * rows_per_rank * MATRIX_DIM],
                rows_per_rank * MATRIX_DIM,
                MPI_DOUBLE,
                proc,
                0,
                MPI_COMM_WORLD);

            MPI_Send(
                b.data(),
                MATRIX_DIM,
                MPI_DOUBLE,
                proc,
                0,
                MPI_COMM_WORLD);
        }

        std::vector<double> A_local(
            A.begin(),
            A.begin() + rows_per_rank * MATRIX_DIM);

        std::vector<double> x_local(rows_per_rank);

        multiplicar_matriz_vector(
            A_local,
            b,
            x_local,
            rows_per_rank,
            MATRIX_DIM);

        for (int i = 0; i < rows_per_rank; i++)
        {
            x[i] = x_local[i];
        }

        for (int proc = 1; proc < nprocs; proc++)
        {
            MPI_Recv(
                &x[proc * rows_per_rank],
                rows_per_rank,
                MPI_DOUBLE,
                proc,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        }

        fmt::print("\nResultado:\n");

        imprimir_vector(x, MATRIX_DIM);
    }
    else
    {

        std::vector<int> metadata(2);

        MPI_Recv(
            metadata.data(),
            2,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        int matrix_dim = metadata[0];
        int rows = metadata[1];

        std::vector<double> A_local(
            rows * matrix_dim);

        MPI_Recv(
            A_local.data(),
            rows * matrix_dim,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        std::vector<double> b_local(matrix_dim);

        MPI_Recv(
            b_local.data(),
            matrix_dim,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        std::vector<double> x_local(rows);

        multiplicar_matriz_vector(
            A_local,
            b_local,
            x_local,
            rows,
            matrix_dim);

        MPI_Send(
            x_local.data(),
            rows,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD);
    }
    MPI_Finalize();

    return 0;
}