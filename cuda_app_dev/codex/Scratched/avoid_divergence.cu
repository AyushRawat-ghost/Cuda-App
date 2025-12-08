#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
using namespace std;

__global__ void avoid_divergence(float* data,int N){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int half=N/2;
    if(tid<half){
        int targetIdx=tid*2;
        data[targetIdx]=cbrtf(data[targetIdx]);
    }else if(tid<N){
        int targetIdx=((tid-half)*2)+1;
        data[targetIdx]=sqrtf(data[targetIdx]);
    }else{
        printf("Out of bond detected");
        // cout<<"Out of bond detected"<<endl;
    }
}

int main(){
    cout <<"CPU Intialized"<<endl;
    int N=1<<20;
    size_t size =N*sizeof(float);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Host mem and data Allocation
    float* h_data=(float*)malloc(size);
    for (int i=0;i<N;i++){
        h_data[i]=(float)i;
    }

    // Device Mem Allocation
    float* d_data;
    cudaMalloc(&d_data,size);

    // Memory Copying
    cudaMemcpy(d_data,h_data,size,cudaMemcpyHostToDevice);
    int blockSize=256;
    int numBlocks = (N / blockSize) + (N % blockSize != 0);

    // func call
    cout<<"GPU Override begins"<<endl;
    
    cudaEventRecord(start);
    avoid_divergence<<<numBlocks,blockSize>>>(d_data,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){
        cout<<"Runtime Error Detected (Kernel Failed to Launch)";
    }

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    cout<<"Time for Execution : "<<ms<<" ms"<<endl<<endl;

    // Memory and data recall
    cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost);
    cout << "Index 0 (Even-cbrt): " << h_data[0] << " (Exp: 0)" << endl;
    cout << "Index 1 (Odd-sqrt):  " << h_data[1] << " (Exp: 1)" << endl;
    cout << "Index 4 (Even-cbrt): " << h_data[4] << " (Exp: 1.58)" << endl;
    cout << "Index 9 (Odd-sqrt):  " << h_data[9] << " (Exp: 3)" << endl;

    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}