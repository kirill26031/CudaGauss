#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "book.h"

#ifndef __MATRIX
#define __MATRIX
#include "Matrix.h"
#endif
#include "SimpleGauss.h"

void parallelGauss(Matrix& matrix);

__global__ void locateMaxElement(double* data, int height, int width, bool* used_columns, bool* used_rows,
    double* max_values, int* max_rows, int* max_columns) {
    int row_id = blockDim.x * blockIdx.x + threadIdx.x;
    int column_id = blockDim.y * blockIdx.y + threadIdx.y;
    int b_in_g = gridDim.x * blockIdx.y + blockIdx.x;
    int t_in_b = blockDim.x * threadIdx.y + threadIdx.x;

    extern __shared__ char* max_per_block[4096];
    double* max_values_per_block = (double*)max_per_block;
    int* max_columns_per_block = (int*)(max_per_block + blockDim.x * blockDim.y * 8);
    int* max_rows_per_block = (int*)(max_per_block + blockDim.x * blockDim.y * 12);

    double max_value = fabs(data[column_id * width + row_id]);
    int max_row = row_id;
    int max_column = column_id;
    double temp;
    while (row_id < width && column_id < height) {
        temp = fabs(data[column_id * width + row_id]);
        if (temp > max_value) {
            max_value = temp;
            max_row = row_id;
            max_column = column_id;
        }
        row_id += blockDim.x;
        column_id += blockDim.y;
    }
    max_values_per_block[t_in_b] = max_value;
    max_rows_per_block[t_in_b] = max_row;
    max_columns_per_block[t_in_b] = max_column;
    __syncthreads();
    // Reduction
    int i = gridDim.x * gridDim.y / 2;
    while (i != 0) {
        if (t_in_b < i) {
            if (max_values_per_block[t_in_b] < max_values_per_block[t_in_b + i]) {
                max_values_per_block[t_in_b] = max_values_per_block[t_in_b + i];
                max_rows_per_block[t_in_b] = max_rows_per_block[t_in_b + i];
                max_columns_per_block[t_in_b] = max_columns_per_block[t_in_b + i];
            }
        }
        __syncthreads();
        i /= 2;
    }
    if (t_in_b == 0) {
        max_rows[b_in_g] = max_rows_per_block[0];
        max_columns[b_in_g] = max_columns_per_block[0];
        max_values[b_in_g] = /*max_values_per_block[0];*/ 
            data[max_columns[b_in_g] * width + max_rows[b_in_g]]; // return without abs
    }
}

void parallelGauss(Matrix& matrix) {
    //matrix.getWidth() / 16, matrix.getHeight() / 16
    dim3 blocks(64, 64);
    dim3 threads(16, 16);
    const int blocks_amount = blocks.x * blocks.y;
    const int threads_amount = threads.x * threads.y;

    double pivot_value;
    int pivot_row;
    int pivot_column;
    int iteration = 0;
    
    bool* used_columns = new bool[matrix.getWidth()]();
    bool* used_rows = new bool[matrix.getHeight()]();
    double* max_values = new double[blocks_amount];
    int* max_rows = new int[blocks_amount];
    int* max_columns = new int[blocks_amount];

    double* dev_matrix = nullptr;
    bool* dev_used_columns = nullptr;
    bool* dev_used_rows = nullptr;
    double* dev_max_values = nullptr;
    int* dev_max_rows = nullptr;
    int* dev_max_columns = nullptr;

    HANDLE_ERROR(cudaSetDevice(0));

    HANDLE_ERROR(cudaMalloc((void**)&dev_matrix, matrix.getHeight() * matrix.getWidth() * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_used_columns, matrix.getWidth() * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_used_rows, matrix.getHeight() * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_max_values, blocks_amount * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_max_rows, blocks_amount * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_max_columns, blocks_amount * sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(dev_matrix, matrix[0], matrix.getHeight() * matrix.getWidth() * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_used_columns, used_columns, matrix.getWidth() * sizeof(bool), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_used_rows, used_rows, matrix.getHeight() * sizeof(bool), cudaMemcpyHostToDevice));

    const int max_values_rows_columns_array_size = threads_amount * (sizeof(double) + 2 * sizeof(int));

    while (iteration < std::min(matrix.getHeight(), matrix.getWidth()) - 1) {
        locateMaxElement<<<blocks, threads>>>(dev_matrix, matrix.getHeight(), matrix.getWidth(), dev_used_columns, dev_used_rows, dev_max_values, dev_max_rows, dev_max_columns);
        HANDLE_ERROR(cudaMemcpy(max_values, dev_max_values, blocks_amount * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(max_rows, dev_max_rows, blocks_amount * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(max_columns, dev_max_columns, blocks_amount * sizeof(int), cudaMemcpyDeviceToHost));

        //Find largest element among blocks
        double max_elem = std::abs(max_values[0]);
        int max_block = 0;
        for (int b_i = 1; b_i < blocks_amount; ++b_i) {
            if (std::abs(max_values[b_i]) > max_elem) {
                max_elem = std::abs(max_values[b_i]);
                max_block = b_i;
            }
        }
        pivot_value = max_values[max_block];
        pivot_row = max_rows[max_block];
        pivot_column = max_columns[max_block];
        used_columns[pivot_column] = true;
        used_rows[pivot_row] = true;
        iteration++;
        if (pivot_value == 0) {
            break;
        }

        std::cout << "\n" << pivot_value;

        HANDLE_ERROR(cudaMemcpy(dev_used_columns + pivot_column, used_columns + pivot_column, sizeof(bool), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_used_rows + pivot_row, used_rows + pivot_row, sizeof(bool), cudaMemcpyHostToDevice));
    }
    cudaDeviceReset();
}


int main()
{

    Matrix* storedMatrix = readFromFile("matrices/1024x1024.txt");
    //std::cout << *storedMatrix;
    //std::cout << "\n\n";
    //SimpleGauss sg(*storedMatrix);
    //sg.byMaxLeadColumn();
    //std::cout << *storedMatrix;
    parallelGauss(*storedMatrix);

    return 0;
}