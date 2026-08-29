#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

vector<int> vectorAddCPU(const vector<int>& a, const vector<int>& b)
{
    vector<int> result(a.size());

    for (int i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }

    return result;
}

int main()
{
    int N = 1000000;

    vector<int> a(N, 1);
    vector<int> b(N, 2);

    auto start = chrono::high_resolution_clock::now();

    vector<int> result = vectorAddCPU(a, b);

    auto end = chrono::high_resolution_clock::now();

    auto duration =
        chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "First result: " << result[0] << endl;
    cout << "Last result: " << result[N - 1] << endl;

    cout << "CPU time: "
         << duration.count()
         << " microseconds"
         << endl;

    return 0;
}