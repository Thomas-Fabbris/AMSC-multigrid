#include "AMG.hpp"
#include "utilities.hpp"
#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix_file> [rhs_option] [max_levels] [min_coarse_size] [theta]\n";
        std::cerr << "RHS options:\n";
        std::cerr << "  <rhs_file>   : Load RHS from file\n";
        std::cerr << "  1 (default)  : RHS = vector of ones\n";
        std::cerr << "  2            : RHS = A * ones (true solution is ones)\n";
        std::cerr << "Example: " << argv[0] << " matrix.mtx 1 10 20 0.25\n";
        std::cerr << "Example: " << argv[0] << " matrix.mtx rhs.txt 10 20 0.25\n";
        return 1;
    }

    std::string matrix_file = argv[1];
    std::string rhs_option = (argc > 2) ? argv[2] : "1";
    int max_levels = (argc > 3) ? std::stoi(argv[3]) : 10;
    int min_coarse_size = (argc > 4) ? std::stoi(argv[4]) : 20;
    double theta = (argc > 5) ? std::stod(argv[5]) : 0.25;

    // Load matrix
    SpMat A;
    if (!load_sparse_matrix(matrix_file, A))
        return 1;

    // Load or generate RHS
    Eigen::VectorXd b;
    if (rhs_option == "1")
    {
        // RHS = vector of ones
        b = Eigen::VectorXd::Ones(A.rows());
        std::cout << "Generated RHS: vector of ones\n";
    }
    else if (rhs_option == "2")
    {
        // RHS = A * ones (true solution is ones)
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(A.rows());
        b = A * ones;
        std::cout << "Generated RHS: A * ones (true solution is ones)\n";
    }
    else
    {
        // Load RHS from file
        if (!load_vector(rhs_option, b))
            return 1;
    }

    if (A.rows() != b.size())
    {
        std::cerr << "Error: Matrix size (" << A.rows() << ") does not match RHS size (" << b.size() << ")\n";
        return 1;
    }

    // Build AMG hierarchy and solve
    std::cout << "\nBuilding AMG hierarchy...\n";
    AMG amg(A, max_levels, min_coarse_size, theta);
    std::cout << "AMG hierarchy: " << amg.num_levels() << " max levels\n";

    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());

    int cycles = 5;
    std::cout << "\nSolving with " << cycles << " max iterations...\n";

    auto start = std::chrono::high_resolution_clock::now();
    amg.solve(x, b, cycles);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double residual_norm = (b - A * x).norm();
    double relative_residual = residual_norm / b.norm();

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "\nResults:\n";
    std::cout << "  Solution norm:        " << x.norm() << "\n";
    std::cout << "  Residual norm:        " << residual_norm << "\n";
    std::cout << "  Relative residual:    " << relative_residual << "\n";
    std::cout << "  Solve time:           " << duration.count() / 1000.0 << " ms\n";

    return 0;
}
