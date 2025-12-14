#include <iostream>
#include <cuda_runtime.h>
#include<stdlib.h>
#include <device_launch_parameters.h>
using namespace std;
#define BLOCK_SIZE 32
#define HEIGHT 1024
#define WIDTH 1024

// GPU Kernel for Transposing
__global__ void transposeKernel(float* input,float *output){
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];
    int x=blockIdx.x*BLOCK_SIZE+threadIdx.x;
    int y=blockIdx.y*BLOCK_SIZE+threadIdx.y;

    if(x<WIDTH && y<HEIGHT){
        tile[threadIdx.y][threadIdx.x] = input[y * WIDTH + x];
    }
    __syncthreads();

    int transposedX=blockIdx.y*BLOCK_SIZE+threadIdx.x;
    int transposedY=blockIdx.x*BLOCK_SIZE+threadIdx.y;

    if(transposedX<HEIGHT && transposedY <WIDTH){
        output[transposedY * HEIGHT + transposedX] = tile[threadIdx.x][threadIdx.y];
    }
}

// Transpose Pre Preparation
void transpose(float *input,float *output,float &time){
    
    float *d_input,*d_output;
    cudaMalloc((void**)&d_input,WIDTH*HEIGHT*sizeof(float));
    cudaMalloc((void**)&d_output,WIDTH*HEIGHT*sizeof(float));
    cudaMemcpy(d_input,input,WIDTH*HEIGHT*sizeof(float),cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE);
    dim3 gridDim((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    transposeKernel<<<gridDim,blockDim>>>(d_input,d_output);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);

    cudaMemcpy(output,d_output,WIDTH*HEIGHT*sizeof(float),cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
}

int main(){
    const int allocate_size=WIDTH*HEIGHT*sizeof(float);
    float* h_input=(float*)malloc(allocate_size);
    float* h_output=(float*)malloc(allocate_size);
    for(int i=0;i<WIDTH*HEIGHT;i++){
        h_input[i]=static_cast<float>(i);
    }
    float time=0.0f;
    transpose(h_input,h_output,time);
    cout<<"GPU Transpose Time : "<<time<<endl;
    cout<<"Verification : "<<h_output[1]<<endl;
    free(h_input);
    free(h_output);
    return 0;
}