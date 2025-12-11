#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
// #define numElements 1000000
#define CHECK_CUDA_ERROR(val) check((val),#val,__FILE__,__LINE__)

// Func to check error
template <typename T>
void check(T err,const char* const func,const char* const file,const int line){
    if (err!=cudaSuccess){
        fprintf(stderr,"CudaError Detected at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
            exit(EXIT_FAILURE);
    }
}

// cuda kernel for complex transformation
__global__ void vectorComplexCompute(const float* A,float* B,float* C,int numElements){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<numElements){
        float temp=A[i]*B[i];
        for(int j=0;j<100;j++){
                temp=sqrt(temp)+exp(A[i])-log(B[i]+1);        
            }
            C[i]=temp;
        }

}

// Cpu Working
void vectorComplexComputeCPU(const float* A,float* B,float* C,int numElements){
    for(int i=0;i<numElements;i++){
        float temp=A[i]*B[i];
        for(int j=0;j<100;j++){
            temp=sqrt(temp)+exp(A[i])-log(B[i]+1);        
        }
        C[i]=temp;
    }
}

int main(){
    const int numElements =10000000;
    size_t size=sizeof(float)*numElements;
    int halfElements=numElements/2;

    printf("CPU Memory Allocation Started\n");
    float* h_A,*h_B,*h_C,*h_result;
    cudaMallocHost(&h_A,size);
    cudaMallocHost(&h_B,size);
    cudaMallocHost(&h_C,size);
    cudaMallocHost(&h_result,size);

    for(int i=0;i<numElements;i++){
        h_A[i]=rand()/(float)RAND_MAX+1.0f;
        h_B[i]=rand()/(float)RAND_MAX+1.0f;
    }
    printf("CPU Bencmarking\n");
    clock_t start_cpu=clock();
    vectorComplexComputeCPU(h_A,h_B,h_result,numElements);
    clock_t end_cpu=clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;


    printf("GPU Memory Allocation Started\n");
    float* d_A,*d_B,*d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A,size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B,size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C,size));
    
    cudaStream_t stream1,stream2;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);


    printf("GPU CPU Memory Transfer Started\n");
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A,h_A,halfElements*sizeof(float),cudaMemcpyHostToDevice,stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B,h_B,halfElements*sizeof(float),cudaMemcpyHostToDevice,stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A+halfElements,h_A+halfElements,halfElements*sizeof(float),cudaMemcpyHostToDevice,stream2));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B+halfElements,h_B+halfElements,halfElements*sizeof(float),cudaMemcpyHostToDevice,stream2));
    

    printf("GPU Working\n");
    int threadPerBlock=256;
    int blocksPerGrid=(halfElements+threadPerBlock-1)/threadPerBlock;

    vectorComplexCompute <<<blocksPerGrid, threadPerBlock, 0, stream1 >>> (d_A, d_B, d_C, halfElements);
    vectorComplexCompute <<<blocksPerGrid, threadPerBlock, 0, stream2 >>> (d_A + halfElements, d_B + halfElements, d_C + halfElements, halfElements);
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, halfElements * sizeof(float), cudaMemcpyDeviceToHost, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C + halfElements, d_C + halfElements, halfElements * sizeof(float), cudaMemcpyDeviceToHost, stream2));
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    CHECK_CUDA_ERROR(cudaEventRecord(stop,0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float gpu_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time, start, stop));

    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_result[i] - h_C[i]) > 1e-4) {
             fprintf(stderr, "Verification FAILED at element %d! CPU=%.5f, GPU=%.5f\n", i, h_result[i], h_C[i]);
             exit(EXIT_FAILURE);
        }
    }
    printf("\nVerification PASSED\n");

    printf("\n=== Performance Comparison ===\n");
    printf("CPU Time: %.3f ms\n", cpu_time);
    printf("GPU Time (Streams): %.3f ms\n", gpu_time);
    printf("Speedup (CPU / GPU): %.2fx\n", cpu_time / gpu_time);
    
    printf("GPU Memory Deallocation Started\n");
    CHECK_CUDA_ERROR(cudaFreeHost(h_A));
    CHECK_CUDA_ERROR(cudaFreeHost(h_B));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C));
    CHECK_CUDA_ERROR(cudaFreeHost(h_result));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));

    return 0;
}