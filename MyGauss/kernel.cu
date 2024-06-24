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
#include <string> 

void parallelGauss(Matrix& matrix);

__global__ void locateMaxElement(double* data, int height, int width, bool* used_columns, bool* used_rows,
    double* max_values, int* max_rows, int* max_columns) {
    int row_id = blockDim.y * blockIdx.y + threadIdx.y;
    int column_id = blockDim.x * blockIdx.x + threadIdx.x;
    int b_in_g = gridDim.x * blockIdx.y + blockIdx.x;
    int t_in_b = blockDim.x * threadIdx.y + threadIdx.x;

    extern __shared__ char* max_per_block[4096];
    double* max_values_per_block = (double*)max_per_block;
    int* max_columns_per_block = (int*)(max_per_block + blockDim.x * blockDim.y * 8);
    int* max_rows_per_block = (int*)(max_per_block + blockDim.x * blockDim.y * 12);

    double max_value = 0;
    int max_row = -1;
    int max_column = -1;
    double temp;
    for (int i = row_id; i < height; i += blockDim.y * gridDim.y) {
        for (int j = column_id; j < width; j += blockDim.x * gridDim.x) {
            if (!used_columns[j] && !used_rows[i]) {
                temp = fabs(data[i * width + j]);
                if (temp > max_value) {
                    max_value = temp;
                    max_row = i;
                    max_column = j;
                }
            }
        }
    }
    max_values_per_block[t_in_b] = max_value;
    max_rows_per_block[t_in_b] = max_row;
    max_columns_per_block[t_in_b] = max_column;
    __syncthreads();
    // Reduction
    int i = blockDim.x * blockDim.y / 2;
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
        max_values[b_in_g] = //max_values_per_block[0];
            (max_rows[b_in_g] == -1 || max_columns[b_in_g] == -1) ? 0 :
            data[max_rows[b_in_g] * width + max_columns[b_in_g]]; // return without abs
    }
}

__global__ void storePivotSeparately(double* data, int height, int width, int pivot_row, int pivot_column,
    double* pivot_row_data, double* pivot_column_data) {
    int row_id = blockDim.y * blockIdx.y + threadIdx.y;
    int column_id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = row_id; i < height; i += blockDim.y * gridDim.y) {
        for (int j = column_id; j < width; j += blockDim.x * gridDim.x) {
            if (i == pivot_row) {
                pivot_row_data[j] = data[i * width + j];
            }
            if (j == pivot_column) {
                pivot_column_data[i] = data[i * width + j];
            }
        }
    }
}

__global__ void changeCoeffs(double* data, int height, int width, int pivot_row, int pivot_column,
    double* pivot_row_data, double* pivot_column_data, bool* used_rows, bool* used_columns, double pivot_value) {
    int row_id = blockDim.y * blockIdx.y + threadIdx.y;
    int column_id = blockDim.x * blockIdx.x + threadIdx.x;
    int b_in_g = gridDim.x * blockIdx.y + blockIdx.x;
    int t_in_b = blockDim.x * threadIdx.y + threadIdx.x;

    extern __shared__ double pivot_column_and_row[2048];
    double* shared_pivot_column = pivot_column_and_row;
    double* shared_pivot_row = pivot_column_and_row + height;

    // Copy pivot row, column to shared memory
    for (int i = t_in_b; i < height; i += blockDim.x * blockDim.y) {
        shared_pivot_column[i] = pivot_column_data[i];
    }
    for (int j = t_in_b; j < width; j += blockDim.x * blockDim.y) {
        shared_pivot_row[j] = pivot_row_data[j];
    }
    __syncthreads();

    // Update coefficients
    for (int i = row_id; i < height; i += blockDim.y * gridDim.y) {
        for (int j = column_id; j < width; j += blockDim.x * gridDim.x) {
            if (!used_columns[j] && !used_rows[i]) {
                double f = shared_pivot_column[i] / pivot_value;
                //data[i * width + j] += 1; // for testing purposes
                data[i * width + j] -= f * shared_pivot_row[j];
            }
            if (j == pivot_column && i != pivot_row && !used_rows[i]) {
                data[i * width + j] = 0;
            }
        }
    }
}

void parallelGauss(Matrix& matrix) {
    const int blocksAmountInXAxis = std::min(std::max(16, matrix.getWidth() / 16), 512);
    const int blocksAmountInYAxis = std::min(std::max(16, matrix.getHeight() / 16), 512);
    //std::cout << matrix;

    dim3 blocks(blocksAmountInXAxis, blocksAmountInYAxis);
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
    double* dev_pivot_row_data = nullptr;
    double* dev_pivot_column_data = nullptr;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    HANDLE_ERROR(cudaSetDevice(0));

    HANDLE_ERROR(cudaMalloc((void**)&dev_matrix, matrix.getHeight() * matrix.getWidth() * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_used_columns, matrix.getWidth() * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_used_rows, matrix.getHeight() * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_pivot_column_data, matrix.getHeight() * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_pivot_row_data, matrix.getWidth() * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_max_values, blocks_amount * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_max_rows, blocks_amount * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_max_columns, blocks_amount * sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(dev_matrix, matrix[0], matrix.getHeight() * matrix.getWidth() * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_used_columns, used_columns, matrix.getWidth() * sizeof(bool), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_used_rows, used_rows, matrix.getHeight() * sizeof(bool), cudaMemcpyHostToDevice));

    const int max_values_rows_columns_array_size = threads_amount * (sizeof(double) + 2 * sizeof(int));

    while (iteration < std::min(matrix.getHeight(), matrix.getWidth()) - 1) {
        locateMaxElement<<<blocks, threads>>>(dev_matrix, matrix.getHeight(), matrix.getWidth(), dev_used_columns, dev_used_rows, dev_max_values, dev_max_rows, dev_max_columns);
        cudaError_t error = cudaGetLastError();
        HANDLE_ERROR(cudaMemcpy(max_values, dev_max_values, blocks_amount * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(max_rows, dev_max_rows, blocks_amount * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(max_columns, dev_max_columns, blocks_amount * sizeof(int), cudaMemcpyDeviceToHost));

        //Find largest element among blocks
        double max_elem = 0;
        int max_block = 0;
        for (int b_i = 0; b_i < blocks_amount; ++b_i) {
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
        if (pivot_value == 0 ) {
            break;
        }

        //std::cout << "\nPivot:" << pivot_value << ", " << pivot_column << " , " << pivot_row << "\n";

        storePivotSeparately<<<blocks, threads>>>(dev_matrix, matrix.getHeight(), matrix.getWidth(), pivot_row, pivot_column,
            dev_pivot_row_data, dev_pivot_column_data);

        HANDLE_ERROR(cudaMemcpy(dev_used_columns, used_columns, matrix.getWidth() * sizeof(bool), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_used_rows, used_rows, matrix.getHeight() * sizeof(bool), cudaMemcpyHostToDevice));
        const int pivot_column_row_array_size = sizeof(double) * (matrix.getHeight() + matrix.getWidth());
        changeCoeffs<<<blocks, threads>>>(dev_matrix, matrix.getHeight(), matrix.getWidth(), pivot_row, pivot_column,
            dev_pivot_row_data, dev_pivot_column_data, dev_used_rows, dev_used_columns, pivot_value);
        cudaDeviceSynchronize();

        /*Matrix temp(matrix.getHeight(), matrix.getWidth());
        HANDLE_ERROR(cudaMemcpy(temp[0], dev_matrix, matrix.getHeight() * matrix.getWidth() * sizeof(double), cudaMemcpyDeviceToHost));

        std::ofstream output;
        output.open(std::string("matrices/1024x1024-temp-") + std::to_string(iteration) + std::string(".txt"));
        output << temp;
        output.close();*/
    }
    HANDLE_ERROR(cudaMemcpy(matrix[0], dev_matrix, matrix.getHeight() * matrix.getWidth() * sizeof(double), cudaMemcpyDeviceToHost));

    // cuda free
    HANDLE_ERROR(cudaFree(dev_matrix));
    HANDLE_ERROR(cudaFree(dev_used_columns));
    HANDLE_ERROR(cudaFree(dev_used_rows));
    HANDLE_ERROR(cudaFree(dev_pivot_column_data));
    HANDLE_ERROR(cudaFree(dev_pivot_row_data));
    HANDLE_ERROR(cudaFree(dev_max_values));
    HANDLE_ERROR(cudaFree(dev_max_rows));
    HANDLE_ERROR(cudaFree(dev_max_columns));

    cudaEventRecord(stop);
    float elapsed_time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "\nElapsed time: " << elapsed_time;

    cudaDeviceReset();
}


int main()
{
    std::ofstream output;
    //Matrix huge(std::pow(2, 13), std::pow(2, 13), true);d 
    //output.open("matrices/7kx7k.txt");
    //output << huge;
    //output.close();

    Matrix matrix = *readFromFile("matrices/7kx7k.txt");

    //SimpleGauss sg(*storedMatrix);
    //sg.byMaxLeadColumn();     

    //Matrix zeros(1024, 1024);
    //parallelGauss(zeros);

    parallelGauss(matrix);
    
    output.open("matrices/7kx7k-parallel.txt");
    output << matrix;
    output.close();

    return 0;
}