#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include<assert.h>

// #include <device_launch_parameter.h>

// GPU Kernel
__global__ void vectorAddGPU(int *a,int *b,int *c,int n){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n){
        c[idx]=a[idx]+b[idx];
    }
}

// Vector initializer
void init_vector(int*a,int*b,int n){
    for(int i=0;i<n;i++){
        a[i]=rand()%100;
        b[i]=rand()%100;
    }
}

// Answer Check
void check_answer(int*a,int*b,int*c,int n){
    for(int i=0;i<n;i++){
        assert(c[i]==a[i]+b[i]);
    }
    printf("\nVerification of all %d elements passed.\n", n);
}
int main(){
    int id=cudaGetDevice(&id);
    printf("Cuda ID %d",id);
    int n = 1 << 20;
    int block_size=256;
    int grid_size=(n+block_size-1)/block_size;
    size_t size =n*sizeof(int);
    int *a,*b,*c;

    cudaMallocManaged(&a,size);
    cudaMallocManaged(&b,size);
    cudaMallocManaged(&c,size);
    vectorAddGPU<<<grid_size,block_size>>>(a,b,c,n);
    cudaDeviceSynchronize();
    check_answer(a,b,c,n);
    printf("\ncompleted successfully , result of sum is %d\n", c[n-1]);

    return 0;
}