#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void initVectorFloatsTwodigits(float* v, int vec_size) {

    for (int i = 0; i < vec_size; i++) {
        //v[i] = rand() % 100;
        v[i] = (float)(rand() % 100) / 10;
        //printf(" %f ", v[i]);
    }
    //printf("\n\n");
}

void initVectorFloatsZeros(float* v, int vec_size) {

    for (int i = 0; i < vec_size; i++) {
        v[i] = 0.0;
    }
    //printf("\n\n");
}


void printMatrix(float* m, int matrixWidth, int matrixHeight) {
    //printf("\n");
    for (int i = 0; i < matrixHeight; i++) {
        for (int j = 0; j < matrixWidth; j++) {
            //printf("%f ", m[j + i * matrixWidth]);
        }
        //printf("\n");
    }
    printf("\n");
}

void printVector(float* v, int n_elements) {
    //printf("\n");
    for (int i = 0; i < n_elements; i++) {
        //printf(" % f ", v[i]);
    }
    //printf("\n\n");
}





//euclidean dist between two points
__device__ float euclDistBetweenTwoPoints(float point1, float point2) {

    int ax = floor(point1);
    int ay = (int)(point1 * 10) % 10;

    int bx = floor(point2);
    int by = (int)(point2 * 10) % 10;

    float euclideanDist = (float)sqrt((float)((ax - bx) * (ax - bx) + (ay - by) * (ay - by)));

    //printf("euclideanDist between %f and %f: %f\n", point1, point2, euclideanDist);

    return euclideanDist;
}

//Kernel
__global__ void findHeuclideanDistBetweenTwoVectors(float* vec, float* matrix, float* sharedMatrix, float* out, int TAM_VEC, int N_VECTORS) {

    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("ThreadIdx %d from block %d - pos: %d\n", threadIdx.x, blockIdx.x, pos);

    if (pos >= TAM_VEC * N_VECTORS) return;
    //printf("Accessed pos: %d\n", pos);

    //sharedMatrix[pos] = euclDistBetweenTwoPoints(vec[pos%TAM_VEC], matrix[pos]);
    //printf("Euclidean distance between vec[%d] and matrix[%d] = sharedMatrix[%d] = %f\n", pos % TAM_VEC, pos, pos, euclDistBetweenTwoPoints(vec[pos % TAM_VEC], matrix[pos]));
    
    //Reduce performed per matrix row (in global memory). Use atomicAdd() to avoid race conditions.
    float dist = euclDistBetweenTwoPoints(vec[pos % TAM_VEC], matrix[pos]);
    atomicAdd(&out[pos / TAM_VEC], dist); 
    //printf("Euclidean distance between vec[%d] and matrix[%d] = %f\n", pos % TAM_VEC, pos, dist);
}


void docSearch(int n_vector_elements, int n_vectors, int trials) {
	
    cudaEvent_t startKernel, stopKernel, startDataTransf, stopDataTransf;
	cudaEventCreate(&startKernel);
	cudaEventCreate(&stopKernel);
    cudaEventCreate(&startDataTransf);
    cudaEventCreate(&stopDataTransf);

    ///Initial data
    const int TAM_VEC = n_vector_elements;     //width  : 128, 256, 512
    const int N_VECTORS = n_vectors;  //height : 1024, 4096, 16384, 65536

    //We fix the number of threads to 256
    const int THREADS_PER_BLOCK = 256;

    //We calculate the number of blocks (of 256 threads for the current size of the matrix)
    const int BLOCKS_PER_GRID = ceil((float)(TAM_VEC * N_VECTORS) / THREADS_PER_BLOCK);
    printf("BLOCKS_PER_GRID: %d\n", BLOCKS_PER_GRID);


    ///Allocate Host memory for our data
    float* host_vec; //vector to compare
    float* host_data; //array of vectors (matrix of size TAM_VEC x N_VECTORS)
    float* host_out;  //Output: array of euclidean distances of each array in host_data

    host_vec = (float*)malloc(TAM_VEC * sizeof(float));
    host_data = (float*)malloc(TAM_VEC * N_VECTORS * sizeof(float));
    host_out = (float*)malloc(N_VECTORS * sizeof(float));
    
    initVectorFloatsTwodigits(host_vec, TAM_VEC);
    initVectorFloatsTwodigits(host_data, TAM_VEC * N_VECTORS);
    initVectorFloatsZeros(host_out,  N_VECTORS);

    ///Allocate GPU memory for our data
    float* d_vec;
    float* d_data;
    float* d_shared_data;
    float* d_out;
    cudaMalloc((float**)&d_vec, TAM_VEC * sizeof(float));
    cudaMalloc((float**)&d_data, TAM_VEC * N_VECTORS * sizeof(float));
    cudaMalloc((float**)&d_shared_data, TAM_VEC * N_VECTORS * sizeof(float));
    cudaMalloc((float**)&d_out, N_VECTORS * sizeof(float));

    ///Transfer host data to the GPU memory
    cudaEventRecord(startDataTransf);
    cudaMemcpy(d_vec, host_vec, TAM_VEC * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, host_data, TAM_VEC * N_VECTORS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, host_out, N_VECTORS * sizeof(float), cudaMemcpyHostToDevice);


    ///Launch the kernel
    cudaEventRecord(startKernel);
    for (size_t i = 0; i < trials; i++)
    {
        findHeuclideanDistBetweenTwoVectors <<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>> (d_vec, d_data, d_shared_data, d_out, TAM_VEC, N_VECTORS);

    }
    cudaEventRecord(stopKernel);
    
    
    ///Copy back the result to host
    cudaMemcpy(host_out, d_out, N_VECTORS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopDataTransf);


    ///Print elapsed time
    cudaEventSynchronize(stopKernel);
    float elapsedTimeMillisecondsK = 0;
    cudaEventElapsedTime(&elapsedTimeMillisecondsK, startKernel, stopKernel);
    printf("Execution time of kernel on GPU: (average of %d iter) %lf milliseconds\n", trials, elapsedTimeMillisecondsK/trials);

    cudaEventSynchronize(stopDataTransf);
    float elapsedTimeMillisecondsDT = 0;
    cudaEventElapsedTime(&elapsedTimeMillisecondsDT, startDataTransf, stopDataTransf);
    printf("Elapsed time of data transfers between CPU and GPU: %lf milliseconds\n", elapsedTimeMillisecondsDT - (elapsedTimeMillisecondsK / trials));
    

    ///Print results
    printf("Vector to search: \n");
    printVector(host_vec, TAM_VEC);

    printf("Data: \n");
    printMatrix(host_data, TAM_VEC, N_VECTORS);

    printf("Vector of euclidean distances: \n");
    printVector(host_out, N_VECTORS);

    float minDist = host_out[0];
    int minVecPos = 0;
    for (int i = 0; i < N_VECTORS; i++)
    {
        if (host_out[i] < minDist) minDist = host_out[i];
        minVecPos = i;
    }
    printf("Vector from data with min euclidean distance: vector %d from data with distance %f\n", minVecPos, minDist);

    cudaFree(d_vec);
    cudaFree(d_data);
    cudaFree(d_out);
}