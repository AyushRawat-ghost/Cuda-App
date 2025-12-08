#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void helloFromGPU(){
    printf("Hello from GPU thread: %d\n", threadIdx.x);
}

int main(){
    cout<<"CPU Drive begins"<<endl;
    cout<<"GPU Override Intiatied"<<endl;
    helloFromGPU<<<2,10>>>();
    cudaDeviceSynchronize();
    cudaError_t err=cudaGetLastError();
    if (err!=cudaSuccess){
        cout<<"Kernel Failed Reached"<<cudaGetErrorString(err)<<endl;
    }
    cout<<"Cpu Drive Ended"<<endl;
    return 0;
}