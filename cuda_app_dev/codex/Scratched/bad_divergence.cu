#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
using namespace std;

__global__ void bad_divergence(float* data,int n){
    int global_pos=blockIdx.x*blockDim.x+threadIdx.x;
    if(global_pos<n){
        if(global_pos%2==0){
            data[global_pos]=sqrtf(data[global_pos]);
        }else{
            data[global_pos]=cbrtf(data[global_pos]);
        }
    }
}
int main(){
    cout<<"CPU Intialized"<<endl;
    int N=1 << 20;
    size_t size=N*sizeof(float);

    // Host Mem Allocation
    float* h_data =(float*)malloc(size);
    for (int i=0;i<N;i++){
        h_data[i]=(float)i;
    }

    // Device Memeory
    float* d_data;
    cudaMalloc(&d_data,size);

    // Data Copy
    cudaMemcpy(d_data,h_data,size,cudaMemcpyHostToDevice);

    // Gpu Workload
    int blockSize=256;
    int numBlocks=(N+blockSize-1)/blockSize;

    cout<<"GPU override Begin"<<endl;
    bad_divergence<<<numBlocks,blockSize>>>(d_data,N);
    cudaDeviceSynchronize();

    cudaError_t err=cudaGetLastError();
    if (err!=cudaSuccess){
        cout<<"Kernel Launch Failed"<<endl;
    }

    // Mem Copy
    cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost);

    cout << "Index 0 (Even - sqrt(0)): " << h_data[0] << " (Expected 0)" << endl; 
    cout << "Index 1 (Odd - cbrt(1)):  " << h_data[1]  << endl;
    cout << "Index 4 (Even - sqrt(4)): " << h_data[4]  << endl;
    cout << "Index 8 (Even - sqrt(8)): " << h_data[8]  << endl;
    cout << "Index 27 (Odd - cbrt(27)): " << h_data[27] <<  endl;
    cudaFree(d_data);
    free(h_data);
    return 0;
}