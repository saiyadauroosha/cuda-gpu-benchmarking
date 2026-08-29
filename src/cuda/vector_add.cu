#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;


// --------------------------------------------------
// GPU kernel
// Each GPU thread adds one pair of vector elements.
// --------------------------------------------------
__global__ void vectorAddGPU(const int* a,
                             const int* b,
                             int* c,
                             int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        c[i] = a[i] + b[i];
    }
}


// --------------------------------------------------
// CPU implementation
// --------------------------------------------------
void vectorAddCPU(const vector<int>& a,
                  const vector<int>& b,
                  vector<int>& c)
{
    for (int i = 0; i < a.size(); i++) {
        c[i] = a[i] + b[i];
    }
}


int main(int argc, char* argv[])
{
    // --------------------------------------------------
    // Configuration
    // --------------------------------------------------
    int N = 1000000;
    int repetitions = 10;

    // Allow vector size from command line
    // Example: ./vector_add 5000000
    if (argc > 1) {
        N = stoi(argv[1]);
    }

    size_t bytes = N * sizeof(int);


    // --------------------------------------------------
    // CPU memory
    // --------------------------------------------------
    vector<int> a(N, 1);
    vector<int> b(N, 2);

    vector<int> cpuResult(N);
    vector<int> gpuResult(N);


    // ==================================================
    // CPU BENCHMARK
    // ==================================================

    double cpuTotal = 0.0;

    for (int r = 0; r < repetitions; r++) {

        auto cpuStart = chrono::steady_clock::now();

        vectorAddCPU(a, b, cpuResult);

        auto cpuEnd = chrono::steady_clock::now();

        double cpuTime =
            chrono::duration<double, milli>(
                cpuEnd - cpuStart
            ).count();

        cpuTotal += cpuTime;
    }

    double averageCPU =
        cpuTotal / repetitions;


    // ==================================================
    // GPU MEMORY ALLOCATION
    // ==================================================

    int* d_a;
    int* d_b;
    int* d_c;

    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);


    // --------------------------------------------------
    // CUDA launch configuration
    // --------------------------------------------------

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;


    // ==================================================
    // GPU WARM-UP
    // ==================================================

    // Copy data once for warm-up
    cudaMemcpy(
        d_a,
        a.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_b,
        b.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    // Warm-up kernel
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(
        d_a,
        d_b,
        d_c,
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
            d_a,
            a.data(),
            bytes,
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            d_b,
            b.data(),
            bytes,
            cudaMemcpyHostToDevice
        );


        // ----------------------------------------------
        // Kernel-only timing
        // ----------------------------------------------

        cudaEventRecord(start);

        vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(
            d_a,
            d_b,
            d_c,
            N
        );

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float kernelTime = 0.0f;

        cudaEventElapsedTime(
            &kernelTime,
            start,
            stop
        );


        // GPU -> CPU
        cudaMemcpy(
            gpuResult.data(),
            d_c,
            bytes,
            cudaMemcpyDeviceToHost
        );


        // ----------------------------------------------
        // End-to-end timing stops
        // ----------------------------------------------

        auto totalEnd =
            chrono::steady_clock::now();

        double totalTime =
            chrono::duration<double, milli>(
                totalEnd - totalStart
            ).count();


        // Add this run's values to totals
        gpuKernelTotal += kernelTime;
        gpuEndToEndTotal += totalTime;
    }


    // Calculate GPU averages

    double averageGPUKernel =
        gpuKernelTotal / repetitions;

    double averageGPUEndToEnd =
        gpuEndToEndTotal / repetitions;


    // ==================================================
    // RESULTS
    // ==================================================

    cout << "Vector size: "
         << N << endl;

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
         << gpuResult[N - 1] << endl;


    // ==================================================
    // CLEANUP
    // ==================================================

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}