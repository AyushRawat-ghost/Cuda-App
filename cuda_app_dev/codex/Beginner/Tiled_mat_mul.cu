#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <time.h>
#include <cuda_runtime.h>
using namespace std;

const int N=1<<10;
const int SHMEM_SIZE =2048;

// GPU Kernel
__global__ void matrixMul(const int* a,const int *b,int* c){
    int row=threadIdx.y+blockDim.y*blockIdx.y;
    int col=threadIdx.x+blockDim.x*blockIdx.x;

    __shared__ int s_a[SHMEM_SIZE/2];
    __shared__ int s_b[SHMEM_SIZE/2];
    int tmp=0;

    for(int i=0;i<N;i+=blockDim.x){
        s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[(i + threadIdx.y) * N + col];
        __syncthreads();
        for (int j = 0; j < blockDim.x; j++) {
            tmp +=
                s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        c[row * N + col] = tmp;
    }
    }

// Result Verification
void verify_result(vector<int>&a,vector<int>&b,vector<int>&c){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            long long tmp=0;
            for(int k=0;k<N;k++){
                tmp+=(long long)a[i*N+k]*b[k*N+j];
            }
            assert((int)tmp == c[i*N+j]);
        }
    }
    printf("Verification Successful\n");
}

int main(){
    srand(time(NULL));
    const int THREADS=32;
    const int BLOCKS = N/THREADS;
    size_t size=N*N*sizeof(int);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    // Host Vectors initializer
    vector<int> h_a(N*N);
    vector<int> h_b(N*N);
    vector<int> h_c(N*N);

    // Vector Filling
    generate(
        h_a.begin(),h_a.end(),[]() 
        { 
            return rand() % 100; 
        }
    );
    generate(
        h_b.begin(),h_b.end(),[]() 
        { 
            return rand() % 100; 
        }
    );

    // Device Pointers
    int *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);
    
    // Data Copying
    cudaMemcpy(d_a,h_a.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.data(),size,cudaMemcpyHostToDevice);
    
    dim3 threads(THREADS,THREADS);
    dim3 blocks(BLOCKS,BLOCKS);

    cudaEventRecord(start);
    matrixMul<<<blocks,threads>>>(d_a,d_b,d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){
        cout<<"Error Raised";
    }
    cudaMemcpy(h_c.data(),d_c,size,cudaMemcpyDeviceToHost);
    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    cout<<"GPU Total Time  : "<<ms<<" ms"<<endl;
    
    verify_result(h_a,h_b,h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}