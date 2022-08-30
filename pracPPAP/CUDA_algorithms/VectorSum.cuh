#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


__global__ void vectorSum(const int* a, const int* b, int* c, const int n_elements)
{
    const int THREADS_PER_BLOCK = blockDim.x;
    int pos = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    if (pos < n_elements) {
        c[pos] = a[pos] + b[pos];
    }
}

void vecsum(int n_elements_vector, int trials) {

    //We fix the number of threads to 256
    const int THREADS_PER_BLOCK = 256;
    const int ARRAY_N_ELEMENTS = n_elements_vector;
    //Number of blocks depends on input vector size 
    const int NUM_BLOCKS = ceil((float)ARRAY_N_ELEMENTS / THREADS_PER_BLOCK);
    const int ARRAY_SPACE = ARRAY_N_ELEMENTS * sizeof(int);

    //CUDA Events to measure time
    cudaEvent_t startKernel, stopKernel; //kernel execution
    cudaEvent_t startDataTransf, stopDataTransf; //kernel execution + data transfers
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&startDataTransf);
    cudaEventCreate(&stopDataTransf);

    //Allocate and initialize input arrays with random values on the host.
    //Why not allocate on the stack and allocate dynamically (on the heap): https://forums.developer.nvidia.com/t/why-am-i-getting-stack-overflow/25970/6
    int* host_a1 = (int*)malloc(ARRAY_SPACE);
    int* host_a2 = (int*)malloc(ARRAY_SPACE);
    int* host_a_out = (int*)malloc(ARRAY_SPACE);
    
    for (int i = 0; i < ARRAY_N_ELEMENTS; i++) {
        host_a1[i] = (int)rand() % 10;
        host_a2[i] = (int)rand() % 10;
    }

    //Declare GPU memory pointers
    int* device_a1;
    int* device_a2;
    int* device_a_out;

    //Allocate memory in GPU
    cudaMalloc((int**) & device_a1, ARRAY_SPACE);
    cudaMalloc((int**) &device_a2, ARRAY_SPACE);
    cudaMalloc((int**) &device_a_out, ARRAY_SPACE);

    //Transfer data to GPU memory
    cudaEventRecord(startDataTransf);
    cudaMemcpy(device_a1, host_a1, ARRAY_SPACE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_a2, host_a2, ARRAY_SPACE, cudaMemcpyHostToDevice);


    //Launch the kernel
    cudaEventRecord(startKernel);
    for (int i = 0; i < trials; i++) {
        vectorSum <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> (device_a1, device_a2, device_a_out, ARRAY_N_ELEMENTS);
    }
    cudaEventRecord(stopKernel);

    //Copy back the result to host
    cudaMemcpy(host_a_out, device_a_out, ARRAY_SPACE, cudaMemcpyDeviceToHost);
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
    for (int i = 0; i < ARRAY_N_ELEMENTS; i++) {
        printf("out[%d] = %d + %d = %d\n", i, host_a1[i], host_a2[i], host_a_out[i]);
    }
    printf("\n");
 
    //free resources
	free(host_a1); free(host_a2); free(host_a_out);
	cudaFree(device_a1); cudaFree(device_a2); cudaFree(device_a_out);

}