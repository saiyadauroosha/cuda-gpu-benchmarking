#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

void matrixMultiplyCPU(const vector<int>& A,
                       const vector<int>& B,
                       vector<int>& C,
                       int N)
{
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {

            int sum = 0;

            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}
int main()
{
    int N = 256;

    vector<int> A(N * N, 1);
    vector<int> B(N * N, 2);
    vector<int> C(N * N);

    auto start = chrono::steady_clock::now();

    matrixMultiplyCPU(A, B, C, N);

    auto end = chrono::steady_clock::now();

    double milliseconds =
        chrono::duration<double, milli>(end - start).count();

    cout << "CPU matrix multiplication time: "
         << milliseconds << " ms" << endl;

    cout << "First result: " << C[0] << endl;
    cout << "Last result: " << C[N * N - 1] << endl;

    return 0;
}