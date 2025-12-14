#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include<iostream>
using namespace std;


#define N (1ULL << 20)
#define BLOCK_SIZE 256

// GPU Kernel
__global__ void vectorAddGPU(int* a,int *b,int *result){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N){
        result[idx]=a[idx]+b[idx];
    }
}

// Vector Initializer
void vectorInit(int *v){
    for (int i=0;i<N;i++){
        v[i]=rand()%100;
    }
}

// Vector Verifier
void checkResult(int *a,int *b,int *c){
    for(int i=0;i<N;i++){
        assert(c[i]==a[i]+b[i]);
    }
}

int main(){
    int id;

    cudaGetDevice(&id);
    int *a,*b,*c;
    size_t size =N*sizeof(int);

    cudaMallocManaged(&a,size);
    cudaMallocManaged(&b,size);
    cudaMallocManaged(&c,size);
    
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    vectorInit(a);
    vectorInit(b);
    int grid_size=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    
    // cudaMemPrefetchAsync(a,size,id);
    // cudaMemPrefetchAsync(b,size,id);
    cudaEventRecord(start);
    vectorAddGPU<<<grid_size,BLOCK_SIZE>>>(a,b,c);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        cout << "Critical Kernel Failure: " << cudaGetErrorString(err) << endl;
        cudaFree(a); cudaFree(b); cudaFree(c);
        cudaEventDestroy(start); cudaEventDestroy(end);
        return 1;
    }

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cout << "GPU Execution Time: " << ms << " ms" << endl;
    
    cudaMemPrefetchAsync(c,size,cudaCpuDeviceId);
    checkResult(a,b,c);
    printf("Completed Succesfully\n");
    cout<<"Result : "<<c[0]<<endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}