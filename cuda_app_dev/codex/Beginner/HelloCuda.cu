#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
using namespace std;

__constant__ char d_message[25];
__global__ void welcome(char* msg,int N){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    // printf("Device Message",msg);
    if(idx<N){
        msg[idx]=d_message[idx];
    }
}

int main(){
    cout<<"Event Creation"<<endl;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cout<<"Memeory Allocation and transfers"<<endl;
    char *d_msg,*h_msg;
    const char message[]="Alpha Prime Activated";
    const int length = strlen(message)+1;

    if (length > sizeof(d_message)) {
        cout << "Error: Message exceeds 20-byte __constant__ size!" << endl;
        return 1;
    }
    // h_msg=(char*)malloc(length*sizeof(char));
    cudaMallocHost(&h_msg,length*sizeof(char));
    cudaMalloc(&d_msg,length*sizeof(char));
    cudaMemcpyToSymbol(d_message,message,length);

    int blockSize=256;
    int numBlocks=(length+blockSize-1)/blockSize;

    cout<<"GPU Override Begins"<<endl;
    cudaEventRecord(start);
    welcome<<<numBlocks,blockSize>>>(d_msg,length);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){
        cout<<"Runtime Error Detected"<<cudaGetErrorString(err)<<endl;
    }

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    cout<<"Time for Execution : "<<ms<<" ms"<<endl;
    cudaMemcpy(h_msg,d_msg,length,cudaMemcpyDeviceToHost);
    
    cout<<"Host Message : "<<h_msg<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_msg);
    cudaFree(d_msg);
    return 0;

}