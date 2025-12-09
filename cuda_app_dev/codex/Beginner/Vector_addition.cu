#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <stdlib.h>
#include <cmath>
using namespace std;

#define N 10000000
#define BLOCK_SIZE 256

void vector_add_cpu(float *v1, float *v2, float *result, int size){
    for(int i=0;i<size;i++){
        result[i]=v1[i]+v2[i];
    }
}

__global__ void vector_add_gpu(float *v1, float *v2, float *result, int size){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<size){
        result[idx]=v1[idx]+v2[idx];
    }
}

void init_vector(float *vec, int size){
    for (int i=0;i<size;i++){
        vec[i]=(float)rand()/RAND_MAX;
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}

int main(){
    cout << "--- CUDA Vector Addition Benchmark ---" << endl;
    
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double start_time, end_time, cpu_time=0.0, cpu_time_avg;
    
    float *h_v1,*h_v2,*h_result_cpu,*h_result_gpu;
    float *d_v1,*d_v2,*d_result;
    size_t size=(size_t)N*sizeof(float);
    
    cudaMallocHost(&h_v1,size);
    cudaMallocHost(&h_v2,size);
    cudaMallocHost(&h_result_cpu,size);
    cudaMallocHost(&h_result_gpu,size);

    cudaMalloc(&d_v1,size);
    cudaMalloc(&d_v2,size);
    cudaMalloc(&d_result,size);
    
    init_vector(h_v1,N);
    init_vector(h_v2,N);
    
    cudaMemcpy(d_v1,h_v1,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2,h_v2,size,cudaMemcpyHostToDevice);

    unsigned int numBlocks=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    const int RUNS=20;

    // --- 1. CPU BENCHMARK ---
    printf("\n[1] Benchmarking CPU...\n");
    for(int i=0;i<RUNS;i++){
        start_time=get_time();
        vector_add_cpu(h_v1,h_v2,h_result_cpu,N);
        end_time=get_time();
        cpu_time+=end_time-start_time;
    }
    cpu_time_avg=cpu_time/RUNS;
    cout<<"CPU Total Time (20 runs) : "<<cpu_time*1000.0<<" ms"<<endl;
    cout<<"CPU Avg Time : "<<cpu_time_avg*1000.0<<" ms"<<endl;

    // --- 2. GPU BENCHMARK ---
    printf("\n[2] Benchmarking GPU...\n");
    cudaEventRecord(start);
    for(int i=0;i<RUNS;i++){
        vector_add_gpu<<<numBlocks,BLOCK_SIZE>>>(d_v1,d_v2,d_result,N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    
    cout<<"GPU Total Time (20 runs) : "<<ms<<" ms"<<endl;
    cout<<"GPU Avg Time : "<<ms/RUNS<<" ms"<<endl;
    
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){
        cout<<"\nRUNTIME ERROR: "<<cudaGetErrorString(err)<<endl;
    }
    
    cudaMemcpy(h_result_gpu, d_result, size, cudaMemcpyDeviceToHost);

    // --- 3. VERIFICATION ---
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_result_gpu[i] - h_result_cpu[i]) > 1e-5) {
            errors++;
        }
    }
    cout << "\nVerification Result:" << endl;
    if (errors == 0) {
        cout << "SUCCESS! GPU and CPU results match." << endl;
    } else {
        cout << "FAILURE! " << errors << " total errors found." << endl;
    }

    // --- 4. CLEANUP ---
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_result);
    cudaFreeHost(h_v1);
    cudaFreeHost(h_v2);
    cudaFreeHost(h_result_cpu);
    cudaFreeHost(h_result_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}