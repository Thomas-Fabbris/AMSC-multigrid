#ifndef DISCRETIZE_DATA_HPP
#define DISCRETIZE_DATA_HPP

#include <functional>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

/**
 * @brief Discretization helpers for finite-difference Poisson problems.
 *
 * Provides structs to describe a rectangular domain and a
 * finite-difference grid, along with helper routines to assemble the
 * variable-coefficient Poisson matrix and the corresponding right-hand side
 * vector for Dirichlet boundary conditions.
 */
namespace Discretization
{

/**
 * @brief Axis-aligned rectangular domain.
 *
 */
struct RectDomain
{
    double x0 { 0.0 }, x1 { 1.0 }; ///< x-range [x0,x1]
    double y0 { 0.0 }, y1 { 1.0 }; ///< y-range [y0,y1]
};

/**
 * @brief Description of a uniform 2D finite-difference grid.
 *
 * `nx` and `ny` represent the number of interior (unknown) points in the
 * x- and y-directions respectively. The spacing `h` is the physical mesh
 * size between neighboring grid points (assumed uniform in both directions).
 */
struct FDGrid
{
    int nx { 0 };     ///< number of interior points in x-direction
    int ny { 0 };     ///< number of interior points in y-direction
    double h { 0.0 }; ///< grid spacing
};

/**
 * @brief Create a uniform finite-difference grid over `dom`.
 *
 * The spacing `h` is computed as (x1 - x0) / (nx + 1) which corresponds to
 * `nx` interior points and Dirichlet boundaries at the domain edges.
 *
 * @param nx Number of interior nodes in x-direction.
 * @param ny Number of interior nodes in y-direction.
 * @param dom Rectangular domain describing physical extent.
 * @return FDGrid Configured grid with `nx`, `ny`, and computed `h`.
 */
inline FDGrid make_grid(int nx, int ny, const RectDomain& dom)
{
    FDGrid g;
    g.nx = nx;
    g.ny = ny;
    g.h = (dom.x1 - dom.x0) / (nx + 1);
    return g;
}

/**
 * @brief Assemble the sparse matrix for a variable-coefficient Poisson problem.
 *
 * The discretization uses a 5-point finite-difference stencil on a uniform
 * grid with Dirichlet boundary conditions. The coefficient function `a(x,y)`
 * is evaluated at nodal locations and used as a heuristic per-direction
 * coefficient. The assembled matrix `A` has size `N x N` where `N = nx * ny`.
 *
 * @param grid Grid description returned by `make_grid`.
 * @param dom Physical rectangular domain corresponding to `grid`.
 * @param a Diffusion coefficient function.
 * @param A (output) Sparse matrix to be resized and filled.
 */
inline void assemble_variable_coeff_poisson(
    const FDGrid& grid,
    const RectDomain& dom,
    const std::function<double(double, double)>& a,
    Eigen::SparseMatrix<double>& A)
{
    const int nx = grid.nx;
    const int ny = grid.ny;
    const double h = grid.h;
    const double x0 = dom.x0;
    const double y0 = dom.y0;

    const int N = nx * ny;
    A.resize(N, N);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(5 * N);

    auto idx = [nx](int i, int j) { return j * nx + i; };

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            const int k = idx(i, j);

            const double x = x0 + (i + 1) * h;
            const double y = y0 + (j + 1) * h;

            // For simplicity we sample a(x,y) at the node and reuse it for
            // all directional conductances. A more accurate discretization
            // would evaluate edge-centered coefficients.
            const double axy = a(x, y);
            const double ax_p = axy;
            const double ax_m = axy;
            const double ay_p = axy;
            const double ay_m = axy;

            const double cx_p = ax_p / (h * h);
            const double cx_m = ax_m / (h * h);
            const double cy_p = ay_p / (h * h);
            const double cy_m = ay_m / (h * h);

            double diag = cx_p + cx_m + cy_p + cy_m;
            trips.emplace_back(k, k, diag);

            if (i + 1 < nx)
            {
                trips.emplace_back(k, idx(i + 1, j), -cx_p);
            }
            if (i - 1 >= 0)
            {
                trips.emplace_back(k, idx(i - 1, j), -cx_m);
            }
            if (j + 1 < ny)
            {
                trips.emplace_back(k, idx(i, j + 1), -cy_p);
            }
            if (j - 1 >= 0)
            {
                trips.emplace_back(k, idx(i, j - 1), -cy_m);
            }
        }
    }

    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();
}

/**
 * @brief Assemble the right-hand side vector `b` for Dirichlet BCs.
 *
 * The forcing term `f(x,y)` is evaluated at interior nodes. Dirichlet
 * boundary values are provided by `g(x,y)` and incorporated into the RHS
 * via standard finite-difference boundary adjustments. The function `a(x,y)`
 * is used with the same nodal sampling heuristic as in the matrix assembly.
 *
 * @param grid Grid description returned by `make_grid`.
 * @param dom Physical rectangular domain corresponding to `grid`.
 * @param a Diffusion coefficient function.
 * @param f Right-hand side function.
 * @param g Dirichlet boundary function defined on the domain boundary.
 * @param b (output) Vector containing assembled RHS.
 *
 * @note The implementation assumes Dirichlet values are known on the full
 * domain boundary..
 */
inline void assemble_rhs(
    const FDGrid& grid,
    const RectDomain& dom,
    const std::function<double(double, double)>& a,
    const std::function<double(double, double)>& f,
    const std::function<double(double, double)>& g,
    Eigen::VectorXd& b)
{
    const int nx = grid.nx;
    const int ny = grid.ny;
    const double h = grid.h;
    const double x0 = dom.x0;
    const double y0 = dom.y0;

    const int N = nx * ny;
    b.resize(N);

    auto idx = [nx](int i, int j) { return j * nx + i; };

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            const int k = idx(i, j);
            const double x = x0 + (i + 1) * h;
            const double y = y0 + (j + 1) * h;

            double rhs = f(x, y);
            const double axy = a(x, y);

            // Boundary neighbors: incorporate Dirichlet boundary values
            // Right boundary (i+1 is boundary when i == nx-1)
            if (i + 1 == nx)
            {
                rhs += (axy / (h * h)) * g(x + h, y);
            }
            // Left boundary (i-1 is boundary when i == 0)
            if (i == 0)
            {
                rhs += (axy / (h * h)) * g(x - h, y);
            }
            // Top boundary (j+1 is boundary when j == ny-1)
            if (j + 1 == ny)
            {
                rhs += (axy / (h * h)) * g(x, y + h);
            }
            // Bottom boundary (j-1 is boundary when j == 0)
            if (j == 0)
            {
                rhs += (axy / (h * h)) * g(x, y - h);
            }

            b[k] = rhs;
        }
    }
}
} // namespace Discretization
#endif // DISCRETIZE_DATA_HPP
