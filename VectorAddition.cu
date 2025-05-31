#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cmath> 

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void addFloat(float* a, float* b, float* c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void vectorAddCPUFloat(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    int n = 10000000; 
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n <= 0) {
        std::cerr << "Error: Vector size N must be positive." << std::endl;
        return 1;
    }
    std::cout << "Running Vector Addition (float) for N = " << n << std::endl;
    size_t bytes = n * sizeof(float);

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_result;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c_cpu = (float*)malloc(bytes);        
    h_c_gpu_result = (float*)malloc(bytes); 
    if (!h_a || !h_b || !h_c_cpu || !h_c_gpu_result) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu_result);
        return 1;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    std::cout << "Initializing vectors..." << std::endl;
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i % 100);          
        h_b[i] = static_cast<float>((n - i) % 100); 
        h_c_cpu[i] = 0.0f;
        h_c_gpu_result[i] = 0.0f;
    }
    std::cout << "Initialization complete." << std::endl;

    std::cout << "\nStarting CPU computation..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    vectorAddCPUFloat(h_a, h_b, h_c_cpu, n);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU computation finished." << std::endl;

    std::cout << "\nStarting GPU computation..." << std::endl;

    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    std::cout << "Copying data H->D..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching kernel..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start_gpu, 0));

    addFloat<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    CUDA_CHECK(cudaEventRecord(stop_gpu, 0));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventSynchronize(stop_gpu));
    std::cout << "Kernel finished. Copying data D->H..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_c_gpu_result, d_c, bytes, cudaMemcpyDeviceToHost));
    std::cout << "Copy D->H complete." << std::endl;

    float gpu_duration_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_duration_ms, start_gpu, stop_gpu));
    std::cout << "GPU computation finished." << std::endl;


    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));

    std::cout << "\nVerifying results..." << std::endl;
    bool success = true;
    float epsilon = 1e-5;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_c_cpu[i] - h_c_gpu_result[i]) > epsilon) {
            if (success) {
                printf("Verification FAILED!\n");
            }
            if (i < 10) {
                 printf("Error at index %d: CPU=%.5f, GPU=%.5f\n", i, h_c_cpu[i], h_c_gpu_result[i]);
            }
            success = false;
        }
    }

    if (success) {
        printf("Verification successful.\n");
    }

    std::cout << "\n--- Performance ---" << std::endl;
    std::cout << "Vector size (N)  : " << n << std::endl;
    std::cout << "CPU Time         : " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "GPU Kernel Time  : " << gpu_duration_ms << " ms" << std::endl;
    if (gpu_duration_ms > 0) {
         double speedup = cpu_duration.count() / gpu_duration_ms;
         std::cout << "Speedup (CPU/GPU): " << speedup << "x" << std::endl;
    } else {
         std::cout << "GPU execution too fast to measure reliably or an error occurred." << std::endl;
    }

    std::cout << "\nFreeing memory..." << std::endl;
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_result);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::cout << "Memory freed." << std::endl;

    return 0;
}