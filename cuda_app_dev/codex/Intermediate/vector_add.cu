#include <iostream>
#include <cmath>

const int N = 1 << 20;

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    std::cout << "Starting CUDA Vector Addition with N = " << N << " elements." << std::endl;
    size_t dataSize = N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(dataSize);
    h_B = (float*)malloc(dataSize);
    h_C = (float*)malloc(dataSize);
    
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    cudaError_t err = cudaMalloc((void**)&d_A, dataSize);
    err = cudaMalloc((void**)&d_B, dataSize);
    err = cudaMalloc((void**)&d_C, dataSize);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaMemcpy(d_A, h_A, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, dataSize, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize; 

    std::cout << "Launching kernel with " << numBlocks << " blocks and " << blockSize << " threads/block." << std::endl;
    
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    cudaMemcpy(h_C, d_C, dataSize, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        if (std::abs(h_C[i] - 3.0f) > 1e-5) {
            errors++;
        }
    }
    
    if (errors == 0) {
        std::cout << "\nSUCCESS! Vector addition result verified (1.0 + 2.0 = 3.0)." << std::endl;
    } else {
        std::cout << "\nFAILURE! " << errors << " errors found in verification." << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}