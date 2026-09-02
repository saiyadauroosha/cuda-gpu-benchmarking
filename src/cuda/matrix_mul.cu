#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;


// --------------------------------------------------
// CPU matrix multiplication
// --------------------------------------------------
void matrixMultiplyCPU(const vector<int>& A,
                       const vector<int>& B,
                       vector<int>& C,
                       int N)
{
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {

            int sum = 0;

            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] *
                       B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}


// --------------------------------------------------
// CUDA kernel
// Each GPU thread computes one output matrix element
// --------------------------------------------------
__global__ void matrixMultiplyGPU(const int* A,
                                  const int* B,
                                  int* C,
                                  int N)
{
    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {

        int sum = 0;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] *
                   B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


int main(int argc, char* argv[])
{
    // --------------------------------------------------
    // Configuration
    // --------------------------------------------------
    int N = 256;

    if (argc > 1) {
        N = stoi(argv[1]);
    }


    int repetitions = 10;

    size_t bytes = N * N * sizeof(int);


    // --------------------------------------------------
    // CPU memory
    // --------------------------------------------------
    vector<int> A(N * N, 1);
    vector<int> B(N * N, 2);

    vector<int> cpuResult(N * N);
    vector<int> gpuResult(N * N);


    // ==================================================
    // CPU BENCHMARK
    // ==================================================

    double cpuTotal = 0.0;

    for (int r = 0; r < repetitions; r++) {

        auto cpuStart =
            chrono::steady_clock::now();

        matrixMultiplyCPU(
            A,
            B,
            cpuResult,
            N
        );

        auto cpuEnd =
            chrono::steady_clock::now();

        double cpuMilliseconds =
            chrono::duration<double, milli>(
                cpuEnd - cpuStart
            ).count();

        cpuTotal += cpuMilliseconds;
    }

    double averageCPU =
        cpuTotal / repetitions;


    // ==================================================
    // GPU MEMORY
    // ==================================================

    int* d_A;
    int* d_B;
    int* d_C;

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);


    // --------------------------------------------------
    // CUDA launch configuration
    // --------------------------------------------------

    // 16 x 16 = 256 threads per block
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1)
            / threadsPerBlock.x,

        (N + threadsPerBlock.y - 1)
            / threadsPerBlock.y
    );


    // ==================================================
    // GPU WARM-UP
    // ==================================================

    // Copy inputs once for warm-up
    cudaMemcpy(
        d_A,
        A.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        B.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    // Warm-up kernel
    matrixMultiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaDeviceSynchronize();


    // ==================================================
    // GPU BENCHMARK
    // ==================================================

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double gpuKernelTotal = 0.0;
    double gpuEndToEndTotal = 0.0;


    for (int r = 0; r < repetitions; r++) {

        // ----------------------------------------------
        // End-to-end timing starts
        // ----------------------------------------------
        auto totalStart =
            chrono::steady_clock::now();


        // CPU -> GPU
        cudaMemcpy(
            d_A,
            A.data(),
            bytes,
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            d_B,
            B.data(),
            bytes,
            cudaMemcpyHostToDevice
        );


        // ----------------------------------------------
        // Kernel-only timing starts
        // ----------------------------------------------
        cudaEventRecord(start);

        matrixMultiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(
            d_A,
            d_B,
            d_C,
            N
        );

        cudaEventRecord(stop);

        // Wait until kernel finishes
        cudaEventSynchronize(stop);


        // Get kernel execution time
        float kernelMilliseconds = 0.0f;

        cudaEventElapsedTime(
            &kernelMilliseconds,
            start,
            stop
        );


        // ----------------------------------------------
        // GPU -> CPU
        // ----------------------------------------------
        cudaMemcpy(
            gpuResult.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        );


        // ----------------------------------------------
        // End-to-end timing stops
        // ----------------------------------------------
        auto totalEnd =
            chrono::steady_clock::now();

        double totalMilliseconds =
            chrono::duration<double, milli>(
                totalEnd - totalStart
            ).count();


        // Add this run to totals
        gpuKernelTotal += kernelMilliseconds;
        gpuEndToEndTotal += totalMilliseconds;
    }


    // Calculate GPU averages
    double averageGPUKernel =
        gpuKernelTotal / repetitions;

    double averageGPUEndToEnd =
        gpuEndToEndTotal / repetitions;


    // ==================================================
    // RESULTS
    // ==================================================

    cout << "Matrix size: "
         << N << " x " << N << endl;

    cout << "Repetitions: "
         << repetitions << endl;

    cout << endl;

    cout << "Average CPU time: "
         << averageCPU
         << " ms" << endl;

    cout << "Average GPU kernel time: "
         << averageGPUKernel
         << " ms" << endl;

    cout << "Average GPU end-to-end time: "
         << averageGPUEndToEnd
         << " ms" << endl;

    cout << endl;

    cout << "First CPU result: "
         << cpuResult[0] << endl;

    cout << "First GPU result: "
         << gpuResult[0] << endl;

    cout << "Last GPU result: "
         << gpuResult[N * N - 1] << endl;


    // ==================================================
    // CLEANUP
    // ==================================================

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}