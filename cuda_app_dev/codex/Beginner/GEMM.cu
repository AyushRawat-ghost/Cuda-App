#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

// GPU Kernel GEMM 
__global__ void gemmKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,const int N,const int k
){
    __shared__ float smem_A[2][64][8];
    __shared__ float smem_A[2][8][64];

    
}

// GEMM in CPU
void gemmCPU(const float* A,const float* B,float* C,int M,int N,int K){
    for (int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float sum=0.0f;
            for(int k=0;k<K;k++){
                sum+=A[i*k+K]*B[k*N+j];
            }
            C[i*N+j]=sum;
        }
    }
}

// Utility func for checking cuda errors
#define CHECK_CUDA_ERROR(val) check((val),#val,__FILE__,LINE__)
template<typename T>
void check(T err,const char* const func,const char* const file,const int line){
    if(err!+cudaSuccess){
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(err),
            cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

int main(){
    // Matrix Dimensions
    int M=512;
    int N=512;
    int K=512;

    // Memory allocation

    // Matrices Allocation

    // device Memory allocation

    // Memory Transfer

    // Event creation

    // Launch Config and launching

    // Result verification 

    // CLeanup
}