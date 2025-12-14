# 📝 GPU Programming Concepts (CUDA)

This README summarizes the core architecture, programming model, and optimization techniques learned in GPU computing using the NVIDIA CUDA framework.

## 1\. 🏗️ Hardware and Execution Model

The GPU uses a hierarchical structure to organize computation.

### 1.1. Thread Hierarchy

The parallel computation is structured in three layers:

  * **Thread:** The fundamental unit of execution. Executes the kernel code. Identified by `threadIdx`.
  * **Thread Block:** A group of threads (up to 1024) that execute together on a single **Streaming Multiprocessor (SM)**. Threads within a block can communicate using **Shared Memory** and synchronize using `__syncthreads()`. Identified by `blockIdx`.
  * **Grid:** The entire launch configuration, composed of multiple thread blocks.

### 1.2. Indexing and Global ID Calculation

Every thread calculates its unique position across the entire grid using the following built-in variables:

  * `blockDim`: The dimensions (size) of a thread block (e.g., 32x32).
  * `blockIdx`: The index of the current block in the grid (starting from 0).
  * `threadIdx`: The index of the current thread in the block (starting from 0).

**Unique Global Index Calculation:**
Global Row = (blockIdx.y * blockDim.y) + threadIdx.y
Global Column = (blockIdx.x * {blockDim.x) + threadIdx.x

### 1.3. Kernels and Execution Flow

  * **Kernel:** A function executed by the GPU (device). It is declared with `__global__` and must return **`void`** because thousands of threads run asynchronously.
  * **Launch Syntax:** The kernel is launched from the CPU (host) using triple chevrons:
    ```c++
    kernel_name<<<Grid_Size, Block_Size>>>(arguments);
    ```

-----

## 2\. 💾 CUDA Memory Model

CUDA defines several types of memory, each with different scopes, speeds, and access methods.

| Memory Type | Location | Scope | Usage | Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Global** | Device (VRAM) | All threads, All Blocks | Input/Output data | Slowest (High Latency) |
| **Shared** | Device (SM) | Threads in the same Block | Inter-thread communication | Fastest (Low Latency) |
| **Local** | Device (VRAM) | Single Thread | Large arrays, register spill | Slow |
| **Registers** | Device (SM) | Single Thread | Scalar variables | Fastest |

### Key Optimizations

  * **Shared Memory Tiling:** Moving frequently accessed data from slow Global Memory into fast Shared Memory for reuse, drastically improving performance.
  * **Coalescing:** Accessing Global Memory in a sequential, aligned pattern to minimize the number of memory transactions.

-----

## 3\. 🌐 Unified Memory (UM)

Unified Memory simplifies data management by creating a single pool of memory accessible by both the CPU and GPU, eliminating explicit `cudaMemcpy` calls.

| Function | Purpose |
| :--- | :--- |
| `cudaMallocManaged(&ptr, size)` | Allocates memory accessible by both Host and Device. |
| **Demand Paging** | The system automatically moves data pages between Host and Device as needed (on CPU access or GPU kernel launch). |

### Optimized Unified Memory with Prefetching

To overcome the latency of demand paging, **Asynchronous Prefetching** is used to proactively move data:

| Function | Purpose | Example |
| :--- | :--- | :--- |
| `cudaMemPrefetchAsync` | Hints to the runtime to asynchronously move data to the specified location before it's needed. | `cudaMemPrefetchAsync(a, size, device_id);` |
| `cudaCpuDeviceId` | A special constant used as the destination ID when prefetching data back to the CPU/Host. | `cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);` |

-----

## 4\. ⚙️ Synchronization and Atomicity

### 4.1. Synchronization

  * `__syncthreads()`: Synchronizes all threads **within the same block**. Ensures all threads have finished reading/writing Shared Memory before proceeding.
  * `cudaDeviceSynchronize()`: Synchronizes the entire **Host** with the entire **Device**. Blocks the CPU until all GPU commands have completed.
  * `cudaEventSynchronize(event)`: Blocks the CPU until a specific GPU event has been recorded.

### 4.2. Atomic Operations

  * **Race Condition:** When multiple threads try to simultaneously read, modify, and write a single shared memory location, resulting in lost updates.
  * **Atomic Addition (`atomicAdd`):** An operation that guarantees the entire read-modify-write cycle on a shared variable is **indivisible**, ensuring correctness for parallel operations like counting or summation (reduction).

-----

## 5\. 🛠️ Optimization Technique: Tiling Example (Matrix Transpose)

The optimized matrix transpose kernel used **Tiling and Padding** to maximize performance:

1.  **Padding:** The Shared Memory tile is declared with an extra column (`BLOCK\_SIZE + 1`) to shift addresses and prevent **Bank Conflicts** during the column-wise read/write access.
2.  **Load:** Threads read the input matrix $A$ in a **coalesced** row-major pattern and store it into the Shared Tile.
3.  **Store:** Threads read the Shared Tile in a **transposed** pattern and write the output matrix $A^T$ in a **coalesced** pattern. The shared memory acts as a high-speed cache, mitigating the uncoalesced nature of the global memory access.

-----



<!-- Building the file -->
docker-compose up -d --build

<!-- container Attachment -->
docker exec -it cuda_cpp_dev_env /bin/bash

<!-- GPU Access  -->
docker run -it --rm --gpus all cuda_cpp_dev_env 

<!-- Compilation and running -->
nvcc <file_name.cu> -o <file_name>
./<file_name>

<!-- Version  -->
nvcc --version

<!-- Docker cmd-->
docker-compose down
docker-compose up -d

<!-- pytorch verification  -->
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device name: {torch.cuda.get_device_name(0)}') "