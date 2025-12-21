#include "../include/DiscretizeData.hpp"
#include "../src/AMG.hpp"
#include <Eigen/Sparse>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

// we don't discretize the matrix manually each time, because it's too expensive
// we are just going to use this example for our convergence test
void assemble_poisson_example(
    int nx, int ny,
    double kappa_x,
    double kappa_y,
    Eigen::SparseMatrix<double>& A)
{
    const int N = nx * ny;
    A.resize(N, N);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(5 * N);

    auto idx = [nx](int i, int j) { return j * nx + i; };

    const double hx = 1.0 / (nx + 1);
    const double hy = 1.0 / (ny + 1);

    const double cx = kappa_x / (hx * hx);
    const double cy = kappa_y / (hy * hy);

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int k = idx(i, j);

            trips.emplace_back(k, k, 2.0 * (cx + cy));

            if (i > 0)
                trips.emplace_back(k, idx(i - 1, j), -cx);
            if (i < nx - 1)
                trips.emplace_back(k, idx(i + 1, j), -cx);
            if (j > 0)
                trips.emplace_back(k, idx(i, j - 1), -cy);
            if (j < ny - 1)
                trips.emplace_back(k, idx(i, j + 1), -cy);
        }
    }

    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();
}

int main()
{
    // sizes to test
    std::vector<int> sizes = { 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 600, 800, 1000 };

    // CSV header
    std::cout << "Size,Iterations,Time(s)" << std::endl;
    // Also save to file
    std::ofstream out("convergence.csv");
    out << "N,Iterations,Time(s)" << std::endl;

    for (int n : sizes)
    {
        SpMat A;
        assemble_poisson_example(n, n, 1.0, 1.0, A);

        Eigen::VectorXd b;
        b = Eigen::VectorXd::Ones(n * n);

        Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());

        // AMG params
        int maxLevels = 3;
        int minCoarseSize = 10;
        int maxIterations = 200;
        int preSmooth = 3;
        int postSmooth = 3;
        double tol = 1e-8;

        AMG amg(A, maxLevels, minCoarseSize);

        auto start = std::chrono::high_resolution_clock::now();
        int iters = amg.solve(x, b, maxIterations, preSmooth, postSmooth, tol);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();

        // Report total unknowns vs iterations
        std::cout << n << " x " << n << "," << iters << "," << elapsed << std::endl;
        // Also write to CSV file
        out << n << "," << iters << "," << elapsed << std::endl;
    }

    out.close();
    return 0;
}
