#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <Eigen/Sparse>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file utilities.hpp
 * @brief I/O and printing helpers.
 *
 * Provides routines to load/save sparse matrices in Matrix Market format,
 * load/save dense vectors, and `print_header`
 * helper for formatted console output.
 */

using SpMat = Eigen::SparseMatrix<double>; ///< Alias for sparse matrix type

/**
 * @brief Save a sparse matrix in Matrix Market format.
 *
 * @param A Sparse matrix to save.
 * @param filename Output filename.
 */
inline void save_sparse_matrix(const SpMat& A, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << A.rows() << " " << A.cols() << " " << A.nonZeros() << "\n";

    for (int k = 0; k < A.outerSize(); ++k)
    {
        for (SpMat::InnerIterator it(A, k); it; ++it)
        {
            file << (it.row() + 1) << " " << (it.col() + 1) << " "
                 << std::setprecision(16) << it.value() << "\n";
        }
    }

    file.close();
    std::cout << "  → Saved to: " << filename << std::endl;
}

/**
 * @brief Save a dense vector to a text file (one value per line).
 *
 * Values are written using scientific notation with high precision.
 *
 * @param v Vector to save.
 * @param filename Output filename.
 */
inline void save_vector(const Eigen::VectorXd& v, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    file << std::scientific << std::setprecision(16);
    for (int i = 0; i < v.size(); ++i)
    {
        file << v(i) << "\n";
    }

    file.close();
    std::cout << "  → Saved to: " << filename << std::endl;
}

/**
 * @brief Load a sparse matrix from a Matrix Market file.
 *
 * The routine reads and ignores comment lines starting with '%', then
 * expects a line with `rows cols nnz` followed by `nnz` triplets
 * (1-based indices) formatted as `row col value`.
 *
 * @param filename Input filename in Matrix Market coordinate format.
 * @param A (output) Sparse matrix to populate.
 * @return true on success, false on failure to open the file.
 */
inline bool load_sparse_matrix(const std::string& filename, SpMat& A)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    int rows = 0, cols = 0, nnz = 0;

    while (std::getline(file, line))
    {
        if (line[0] != '%')
        {
            std::istringstream iss(line);
            iss >> rows >> cols >> nnz;
            break;
        }
    }

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(nnz);

    int row, col;
    double val;
    while (file >> row >> col >> val)
    {
        trips.emplace_back(row - 1, col - 1, val);
    }

    A.resize(rows, cols);
    A.setFromTriplets(trips.begin(), trips.end());
    file.close();

    std::cout << "Loaded matrix: " << rows << "x" << cols
              << ", nnz = " << nnz << std::endl;
    return true;
}

/**
 * @brief Load a dense vector from a text file (one value per line).
 *
 * @param filename Input filename.
 * @param v (output) Vector to populate.
 * @return true on success, false on failure to open the file.
 */
inline bool load_vector(const std::string& filename, Eigen::VectorXd& v)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::vector<double> data;
    double val;
    while (file >> val)
    {
        data.push_back(val);
    }

    v.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        v(i) = data[i];
    }

    file.close();
    std::cout << "Loaded vector of size " << v.size() << std::endl;
    return true;
}

/**
 * @brief Print a formatted ASCII header to stdout.
 * @param title Header title text printed centered.
 */
inline void print_header(const std::string& title)
{
    std::cout << "\n"
              << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

#endif // UTILITIES_HPP
