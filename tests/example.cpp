#include "AMG.hpp"
#include "DiscretizeData.hpp"
#include "utilities.hpp"
#include <Eigen/Sparse>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

int main()
{
    using Trip = Eigen::Triplet<double>;

    // ========== 1D Poisson Problem ==========
    // EXAMPLE WITH ONE VARIABLE
    print_header("1D Poisson Problem");

    int n = 50;
    std::cout << "Problem size: " << n << " unknowns\n";

    std::vector<Trip> trips;
    trips.reserve(3 * n);
    for (int i = 0; i < n; ++i)
    {
        trips.emplace_back(i, i, 2.0);
        if (i > 0)
            trips.emplace_back(i, i - 1, -1.0);
        if (i + 1 < n)
            trips.emplace_back(i, i + 1, -1.0);
    }
    SpMat A(n, n);
    A.setFromTriplets(trips.begin(), trips.end());

    std::cout << "Matrix: " << A.rows() << "x" << A.cols()
              << ", nnz = " << A.nonZeros() << "\n";

    save_sparse_matrix(A, "matrix_1d_poisson.mtx");

    AMG amg(A, /*maxLevels*/ 10, /*minCoarseSize*/ 10, /*theta*/ 0.25);
    std::cout << "AMG hierarchy: " << amg.num_levels() << " max levels\n";

    Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

    save_vector(b, "rhs_1d_poisson.txt");

    int cycles = 5;
    std::cout << "\nSolving with " << cycles << " max iterations...\n";

    auto start = std::chrono::high_resolution_clock::now();
    amg.solve(x, b, cycles);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double residual_norm = (b - A * x).norm();
    double relative_residual = residual_norm / b.norm();

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Results:\n";
    std::cout << "  Solution norm:        " << x.norm() << "\n";
    std::cout << "  Residual norm:        " << residual_norm << "\n";
    std::cout << "  Relative residual:    " << relative_residual << "\n";
    std::cout << "  Solve time:           " << duration.count() / 1000.0 << " ms\n";

    // ========== 2D Variable-Coefficient Poisson Problem ==========
    // same with two variables
    print_header("2D Variable-Coefficient Poisson Problem");

    Discretization::RectDomain dom { 0.0, 1.0, 0.0, 1.0 };
    const int nx = 50;
    const int ny = 50;
    auto grid = Discretization::make_grid(nx, ny, dom);

    std::cout << "Grid: " << nx << "x" << ny << " ("
              << grid.nx * grid.ny << " interior points)\n";

    // Variable coefficient a(x,y)
    auto a = [](double x, double y) {
        return 1.0 + 0.5 * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
    };

    SpMat A2D;
    Discretization::assemble_variable_coeff_poisson(grid, dom, a, A2D);

    std::cout << "Matrix: " << A2D.rows() << "x" << A2D.cols()
              << ", nnz = " << A2D.nonZeros() << "\n";

    save_sparse_matrix(A2D, "matrix_2d_poisson.mtx");

    auto f = [](double /*x*/, double /*y*/) { return 1.0; };
    auto g = [](double /*x*/, double /*y*/) { return 0.0; };

    Eigen::VectorXd b2D;
    Discretization::assemble_rhs(grid, dom, a, f, g, b2D);

    save_vector(b2D, "rhs_2d_poisson.txt");

    AMG amg2(A2D, /*maxLevels*/ 10, /*minCoarseSize*/ 20, /*theta*/ 0.25);
    std::cout << "AMG hierarchy: " << amg2.num_levels() << " max levels\n";

    Eigen::VectorXd x2D = Eigen::VectorXd::Zero(b2D.size());

    int cycles2D = 5;
    std::cout << "\nSolving with " << cycles2D << " max iterations...\n";

    start = std::chrono::high_resolution_clock::now();
    amg2.solve(x2D, b2D, cycles2D);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double residual_norm_2d = (b2D - A2D * x2D).norm();
    double relative_residual_2d = residual_norm_2d / b2D.norm();

    std::cout << "Results:\n";
    std::cout << "  Solution norm:        " << x2D.norm() << "\n";
    std::cout << "  Residual norm:        " << residual_norm_2d << "\n";
    std::cout << "  Relative residual:    " << relative_residual_2d << "\n";
    std::cout << "  Solve time:           " << duration.count() / 1000.0 << " ms\n";

    print_header("Summary");
    std::cout << "All tests completed successfully!\n";
    std::cout << "Matrix files saved in Matrix Market format (.mtx)\n\n";

    return 0;
}
