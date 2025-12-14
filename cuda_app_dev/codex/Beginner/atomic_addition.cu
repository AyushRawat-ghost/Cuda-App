// Atomic op for updating single memory location without any Race Condition Issue
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#define N 100000
#define BLOCK_SIZE 1024
using namespace std;

__global__ void atomicAddKernel(int* sum,int n){
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    if(idx<n){
        atomicAdd(sum,idx);
    }
}

int main(){
    printf("CPU Override Detected \n");
    printf("Memory Allocation \n");
    int *h_sum,*d_sum;
    size_t size=sizeof(int);
    cudaMallocHost(&h_sum,size);
    *h_sum=0;
    printf("Value before GPU Working : %d\n",*h_sum);
    cudaMalloc(&d_sum,size);
    cudaMemcpy(d_sum,h_sum,size,cudaMemcpyHostToDevice);

    unsigned int numBlocks=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    cout<<"\nGPU OverRide Activated\n"<<endl;

    atomicAddKernel<<<numBlocks,BLOCK_SIZE>>>(d_sum,N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }
    cudaMemcpy(h_sum,d_sum,size,cudaMemcpyDeviceToHost);
    printf("Value after GPU Working : %d\n",*h_sum);

    cudaFree(d_sum);
    cudaFreeHost(h_sum);

    return 0;
}