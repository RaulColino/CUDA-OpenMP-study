#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void showMatrix(const int* m, const unsigned matrixWidth, const unsigned matrixHeight) {
    printf("\n");
    for (unsigned i = 0; i < matrixHeight; i++) {
        for (unsigned j = 0; j < matrixWidth; j++) {
            printf("%d ", m[j + i * matrixWidth]);
        }
        printf("\n");
    }
    printf("\n");
}


__global__ void matrixSum1D(const int* a, const int* b, int* c, const int n_array_elements)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos < n_array_elements) {
        c[pos] = a[pos] + b[pos];
    }
}

__global__ void matrixSum2D(int* a, int* b, int* c, const int TAM_X, const int TAM_Y) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < TAM_X) && (y < TAM_Y)) {
        int pos = y * TAM_X + x;
        c[pos] = a[pos] + b[pos];
    }
}


void matsum1D(int matrixWidth, int matrixHeight, int trials) {
    printf("\nmatsum1D\n");

    //CUDA Events to measure time
    cudaEvent_t startKernel, stopKernel; //kernel execution
    cudaEvent_t startDataTransf, stopDataTransf; //kernel execution+data transfers

    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&startDataTransf);
    cudaEventCreate(&stopDataTransf);
    
    //Size of matrix
    const int TAM_X = matrixWidth;
    const int TAM_Y = matrixHeight;
    
    //We fix the number of threads to 256
    const int THREADS_PER_BLOCK = 256;
    
    //We calculate the number of blocks (of 256 threads for the current size of the matrix)
    const int BLOCKS_PER_GRID = ceil((float)(TAM_X * TAM_Y) / THREADS_PER_BLOCK);
    printf("BLOCKS_PER_GRID: %d\n", BLOCKS_PER_GRID);
    
    //Tamaño del vector que almacena la matriz en bytes
    const size_t matrix_space = TAM_X*TAM_Y*sizeof(int);
    
    int* h_a, 
    int* h_b, 
    int* h_c;
    int* dev_a, 
    int* dev_b, 
    int* dev_c;
    
    //Init matrices
    h_a = (int*)malloc(matrix_space);
    h_b = (int*)malloc(matrix_space);
    h_c = (int*)malloc(matrix_space);
    
    for (int i = 0; i < (TAM_X*TAM_Y); i++) {
        h_a[i] = (int)rand() % 10;
        h_b[i] = (int)rand() % 10;
    }
    
    //Allocate memory in GPU
    cudaMalloc(&dev_a, matrix_space);
    cudaMalloc(&dev_b, matrix_space);
    cudaMalloc(&dev_c, matrix_space);
    
    //Transfer data to GPU memory
    cudaEventRecord(startDataTransf);
    cudaMemcpy(dev_a, h_a, matrix_space, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_b, matrix_space, cudaMemcpyHostToDevice);

    
    //Run kernel
    cudaEventRecord(startKernel);
    for (int i = 0; i < trials; i++)
    {
        //as vector
        matrixSum1D <<< BLOCKS_PER_GRID, THREADS_PER_BLOCK >>> (dev_a, dev_b, dev_c, TAM_X*TAM_Y);
    }
    cudaEventRecord(stopKernel);
    

    //copy back result to host
    cudaMemcpy(h_c, dev_c, matrix_space, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopDataTransf);


    //Print elapsed time
    cudaEventSynchronize(stopKernel);
    float elapsedTimeMillisecondsK = 0;
    cudaEventElapsedTime(&elapsedTimeMillisecondsK, startKernel, stopKernel);
    printf("Execution time of kernel on GPU: (average of %d iter) %lf milliseconds\n", trials, elapsedTimeMillisecondsK / trials);

    cudaEventSynchronize(stopDataTransf);
    float elapsedTimeMillisecondsDT = 0;
    cudaEventElapsedTime(&elapsedTimeMillisecondsDT, startDataTransf, stopDataTransf);
    printf("Elapsed time of data transfers between CPU and GPU: %lf milliseconds\n", elapsedTimeMillisecondsDT - (elapsedTimeMillisecondsK / trials));

    //Print result
    //showMatrix(h_a, TAM_X, TAM_Y);
    //printf("+\n");
    //showMatrix(h_b, TAM_X, TAM_Y);
    //printf("-----------");
    //showMatrix(h_c, TAM_X, TAM_Y);

        
    //Free memory resources
    free(h_a); free(h_b); free(h_c);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
}


void matsum2D(int matrixWidth, int matrixHeight, int trials) {
    printf("\nmatsum2D\n");

    //CUDA Events to measure time
    cudaEvent_t startKernel, stopKernel; //kernel execution
    cudaEvent_t startDataTransf, stopDataTransf; //kernel execution+data transfers

    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&startDataTransf);
    cudaEventCreate(&stopDataTransf);

    //Size of matrix
    const int TAM_X = matrixWidth;
    const int TAM_Y = matrixHeight;

    //We fix the number of threads to 256
    const int THREADS_PER_BLOCK = 256;

    //We calculate the number of blocks (of 256 threads for the current size of the matrix)
    const int BLOCKS_PER_GRID = ceil((float)(TAM_X * TAM_Y) / THREADS_PER_BLOCK);
    printf("BLOCKS_PER_GRID: %d\n", BLOCKS_PER_GRID);

    //Tamaño del vector que almacena la matriz en bytes
    const size_t matrix_space = TAM_X * TAM_Y * sizeof(int);

    int* h_a; 
    int* h_b;
    int* h_c;
    int* dev_a;
    int* dev_b,
    int* dev_c;

    //Init matrices
    h_a = (int*)malloc(matrix_space);
    h_b = (int*)malloc(matrix_space);
    h_c = (int*)malloc(matrix_space);

    for (int i = 0; i < (TAM_X * TAM_Y); i++) {
        h_a[i] = (int)rand() % 10;
        h_b[i] = (int)rand() % 10;
    }

    //Allocate memory in GPU
    cudaMalloc(&dev_a, matrix_space);
    cudaMalloc(&dev_b, matrix_space);
    cudaMalloc(&dev_c, matrix_space);

    //Transfer data to GPU memory
    cudaEventRecord(startDataTransf);
    cudaMemcpy(dev_a, h_a, matrix_space, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_b, matrix_space, cudaMemcpyHostToDevice);


    // matrixSum2D data
    dim3 threadsPerBlock2D(16, 16); //FIXED SIZE 256 THREADS! = 16x16x1 threads = 256 < 1024
    dim3 numBlocks2D(ceil((float)TAM_X / threadsPerBlock2D.x), ceil((float)TAM_Y / threadsPerBlock2D.y));

    //Run kernel
    cudaEventRecord(startKernel);
    for (int i = 0; i < trials; i++)
    {
        //as matrix
        matrixSum2D <<< numBlocks2D, threadsPerBlock2D >>> (dev_a, dev_b, dev_c, TAM_X, TAM_Y);
    }
    cudaEventRecord(stopKernel);


    //copy back result to host
    cudaMemcpy(h_c, dev_c, matrix_space, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopDataTransf);


    //Print elapsed time
    cudaEventSynchronize(stopKernel);
    float elapsedTimeMillisecondsK = 0;
    cudaEventElapsedTime(&elapsedTimeMillisecondsK, startKernel, stopKernel);
    printf("Execution time of kernel on GPU: (average of %d iter) %lf milliseconds\n", trials, elapsedTimeMillisecondsK / trials);

    cudaEventSynchronize(stopDataTransf);
    float elapsedTimeMillisecondsDT = 0;
    cudaEventElapsedTime(&elapsedTimeMillisecondsDT, startDataTransf, stopDataTransf);
    printf("Elapsed time of data transfers between CPU and GPU: %lf milliseconds\n", elapsedTimeMillisecondsDT - (elapsedTimeMillisecondsK / trials));

    //Print result
    //showMatrix(h_a, TAM_X, TAM_Y);
    //printf("+\n");
    //showMatrix(h_b, TAM_X, TAM_Y);
    //printf("-----------");
    //showMatrix(h_c, TAM_X, TAM_Y);


    //Free memory resources
    free(h_a); free(h_b); free(h_c);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
}