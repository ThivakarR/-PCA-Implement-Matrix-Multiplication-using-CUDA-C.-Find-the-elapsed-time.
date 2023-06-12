
## Aim:
To perform Matrix addition with unified memory and check its performance with nvprof.
## Procedure:

### Step 1 :

Include the required files and library.
### Step 2 :

Introduce a function named "initialData","sumMatrixOnHost","checkResult" to return the initialize the data , perform matrix summation on the host and then check the result.
### Step 3 :

Create a grid 2D block 2D global function to perform matrix on the gpu.
### Step 4 :

Declare the main function. In the main function set up the device & data size of matrix , perform memory allocation on host memory & initialize the data at host side then add matrix at host side for result checks followed by invoking kernel at host side. Then warm-up kernel,check the kernel error, and check device for results.Finally free the device global memory and reset device.
### Step 5 :

Execute the program and run the terminal . Check the performance using nvprof.
## Program :

Developed By : R.Thivakar
Register no. : 212222240109


#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates the use of CUDA managed memory to implement matrix
 * addition. In this example, arbitrary pointers can be dereferenced on the host
 * and device. CUDA will automatically manage the transfer of data to and from
 * the GPU as needed by the application. There is no need for the programmer to
 * use cudaMemcpy, cudaHostGetDevicePointer, or any other CUDA API involved with
 * explicitly transferring data. In addition, because CUDA managed memory is not
 * forced to reside in a single place it can be transferred to the optimal
 * memory space and not require round-trips over the PCIe bus every time a
 * cross-device reference is performed (as is required with zero copy and UVA).
 */

void initialData(float *ip, const int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
    {
        printf("Arrays do not match.\n\n");
    }
}

// grid 2D block 2D
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx,
                             int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting ", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx, ny;
    int ishift = 12;

    if  (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged((void **)&A, nBytes));
    CHECK(cudaMallocManaged((void **)&B, nBytes));
    CHECK(cudaMallocManaged((void **)&gpuRef,  nBytes);  );
    CHECK(cudaMallocManaged((void **)&hostRef, nBytes););

    // initialize data at host side
    double iStart = seconds();
    initialData(A, nxy);
    initialData(B, nxy);
    double iElaps = seconds() - iStart;
    printf("initialization: \t %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrix on host:\t %f sec\n", iElaps);

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warm-up kernel, with unified memory all pages will migrate from host to
    // device
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);

    // after warm-up, time with unified memory
    iStart = seconds();

    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", iElaps,
            grid.x, grid.y, block.x, block.y);

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}

### Removing the memsets

Developed By : G.Chethan Kumar
Register no. : 212222240022


#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates the use of CUDA managed memory to implement matrix
 * addition. In this example, arbitrary pointers can be dereferenced on the host
 * and device. CUDA will automatically manage the transfer of data to and from
 * the GPU as needed by the application. There is no need for the programmer to
 * use cudaMemcpy, cudaHostGetDevicePointer, or any other CUDA API involved with
 * explicitly transferring data. In addition, because CUDA managed memory is not
 * forced to reside in a single place it can be transferred to the optimal
 * memory space and not require round-trips over the PCIe bus every time a
 * cross-device reference is performed (as is required with zero copy and UVA).
 */

void initialData(float *ip, const int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
    {
        printf("Arrays do not match.\n\n");
    }
}

// grid 2D block 2D
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx,
                             int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting ", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx, ny;
    int ishift = 12;

    if  (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged((void **)&A, nBytes));
    CHECK(cudaMallocManaged((void **)&B, nBytes));
    CHECK(cudaMallocManaged((void **)&gpuRef,  nBytes);  );
    CHECK(cudaMallocManaged((void **)&hostRef, nBytes););

    // initialize data at host side
    double iStart = seconds();
    initialData(A, nxy);
    initialData(B, nxy);
    double iElaps = seconds() - iStart;
    printf("initialization: \t %f sec\n", iElaps);

    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrix on host:\t %f sec\n", iElaps);

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warm-up kernel, with unified memory all pages will migrate from host to
    // device
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);

    // after warm-up, time with unified memory
    iStart = seconds();

    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", iElaps,
            grid.x, grid.y, block.x, block.y);

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}

## Output:

root@MidPC:/home/student/Desktop# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
root@MidPC:/home/student/Desktop# nvcc test.cu
root@MidPC:/home/student/Desktop# ./a.out
./a.out Starting using Device 0: NVIDIA GeForce GTX 1660 SUPER
Matrix size: nx 4096 ny 4096
initialization: 	 0.385131 sec
sumMatrix on host:	 0.060113 sec
sumMatrix on gpu :	 0.048901 sec <<<(128,128), (32,32)>>> 
root@MidPC:/home/student/Desktop# nvprof ./a.out
==9917== NVPROF is profiling process 9917, command: ./a.out
./a.out Starting using Device 0: NVIDIA GeForce GTX 1660 SUPER
Matrix size: nx 4096 ny 4096
initialization: 	 0.402009 sec
sumMatrix on host:	 0.059709 sec
sumMatrix on gpu :	 0.066506 sec <<<(128,128), (32,32)>>> 
==9917== Profiling application: ./a.out
==9917== Profiling result:
No kernels were profiled.
No API activities were profiled.
==9917== Warning: Some profiling data are not recorded. Make sure cudaProfilerStop() or cuProfilerStop() is called before application exit to flush profile data.
======== Error: Application received signal 139

![pcaig1](https://github.com/Gchethankumar/PCA-Matrix-Addition-With-Unified-Memory/assets/118348224/7b60baa6-81a7-4258-add2-fd75c8939930)


### Removing the memsets

root@MidPC:/home/student/Desktop# nvcc test.cu
root@MidPC:/home/student/Desktop# ./a.out
./a.out Starting using Device 0: NVIDIA GeForce GTX 1660 SUPER
Matrix size: nx 4096 ny 4096
initialization: 	 0.385390 sec
sumMatrix on host:	 0.068792 sec
sumMatrix on gpu :	 0.039151 sec <<<(128,128), (32,32)>>> 
root@MidPC:/home/student/Desktop# nvprof ./a.out
==10297== NVPROF is profiling process 10297, command: ./a.out
./a.out Starting using Device 0: NVIDIA GeForce GTX 1660 SUPER
Matrix size: nx 4096 ny 4096
initialization: 	 0.418289 sec
sumMatrix on host:	 0.065890 sec
sumMatrix on gpu :	 0.042262 sec <<<(128,128), (32,32)>>> 
==10297== Profiling application: ./a.out
==10297== Profiling result:
No kernels were profiled.
No API activities were profiled.
==10297== Warning: Some profiling data are not recorded. Make sure cudaProfilerStop() or cuProfilerStop() is called before application exit to flush profile data.
======== Error: Application received signal 139


![pcaig2](https://github.com/Gchethankumar/PCA-Matrix-Addition-With-Unified-Memory/assets/118348224/890e1a04-2832-4869-9028-6cf685cb9100)

## Result:

Thus Matrix addition with unified memory is done and its performance with nvprof is checked.
[2:08 PM, 6/12/2023] saturn kumar clg frnd: # -PCA-Implement-Matrix-Multiplication-using-CUDA-C.-Find-the-elapsed-time.
Implement Matrix Multiplication using GPU.

## Aim:
To implement Matrix Multiplication using GPU.

## Procedure:
### Step 1 :

Include the required files and library.
### Step 2 :

Declare the block size and the size of elements .
### Step 3 :

Introduce Kernel function to perform matrix multiplication.In the kernal function,decalre the row column size and initialize the sum to be 0,then using for loop calculate the sum.
### Step 4 :

Intoduce a Main function, in the main method declare the required variables and Initialize the matrices 'a' and 'b'.Allocate memory on the device and then copy the input matrices from host to device memory and set the grid and block sizes . Launch the kernel,Copy the result matrix from device to host memory ,Print the result matrix and the elapsed time followed by freeing the device memory.
### Step 5 :

Save the program and execute it .
## Program:

Developed By: G.Chethan kumar
Register no.: 212222240022


#include <stdio.h>
#include <sys/time.h>

#define SIZE 4
#define BLOCK_SIZE 2

// Kernel function to perform matrix multiplication
__global__ void matrixMultiply(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < size; ++k)
    {
        sum += a[row * size + k] * b[k * size + col];
    }

    c[row * size + col] = sum;
}

int main()
{
    int a[SIZE][SIZE], b[SIZE][SIZE], c[SIZE][SIZE];
    int *dev_a, *dev_b, *dev_c;
    int size = SIZE * SIZE * sizeof(int);

    // Initialize matrices 'a' and 'b'
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    // Copy input matrices from host to device memory
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 dimGrid(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Launch kernel
    matrixMultiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, SIZE);

    // Copy result matrix from device to host memory
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    // Stop timer
    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    // Print the result matrix
    printf("Result Matrix:\n");
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    // Print the elapsed time
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

## Output:

root@MidPC:/home/student/Desktop# nvcc first.cu
root@MidPC:/home/student/Desktop# ./a.out
Result Matrix:
14 8 2 -4 
20 10 0 -10 
26 12 -2 -16 
32 14 -4 -22 
Elapsed Time: 0.000023 seconds
root@MidPC:/home/student/Desktop# nvprof ./a.out
==18221== NVPROF is profiling process 18221, command: ./a.out
Result Matrix:
14 8 2 -4 
20 10 0 -10 
26 12 -2 -16 
32 14 -4 -22 
Elapsed Time: 0.000037 seconds
==18221== Profiling application: ./a.out
==18221== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   39.90%  2.5280us         1  2.5280us  2.5280us  2.5280us  matrixMultiply(int*, int*, int*, int)
                   38.89%  2.4640us         2  1.2320us     928ns  1.5360us  [CUDA memcpy HtoD]
                   21.21%  1.3440us         1  1.3440us  1.3440us  1.3440us  [CUDA memcpy DtoH]
      API calls:   99.38%  126.78ms         3  42.262ms  2.2600us  126.78ms  cudaMalloc
                    0.28%  356.84us         1  356.84us  356.84us  356.84us  cuDeviceTotalMem
                    0.20%  252.08us        97  2.5980us     210ns  107.52us  cuDeviceGetAttribute
                    0.07%  87.360us         3  29.120us  2.5700us  79.360us  cudaFree
                    0.03%  36.470us         1  36.470us  36.470us  36.470us  cuDeviceGetName
                    0.02%  29.180us         3  9.7260us  5.9900us  12.080us  cudaMemcpy
                    0.02%  23.180us         1  23.180us  23.180us  23.180us  cudaLaunchKernel
                    0.00%  4.5900us         1  4.5900us  4.5900us  4.5900us  cuDeviceGetPCIBusId
                    0.00%  2.4000us         3     800ns     250ns  1.8100us  cuDeviceGetCount
                    0.00%     930ns         2     465ns     210ns     720ns  cuDeviceGet
                    0.00%     310ns         1     310ns     310ns     310ns  cuDeviceGetUuid

![pcaimgg1](https://github.com/Gchethankumar/-PCA-Implement-Matrix-Multiplication-using-CUDA-C.-Find-the-elapsed-time./assets/118348224/ab51d0ed-f7c2-4682-b0e8-9b32af52d1cf)

## Result:
The implementation of Matrix Multiplication using GPU is done successfully.
