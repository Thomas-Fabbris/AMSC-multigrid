/**
 * @file jacobi.hpp
 * @brief Jacobi iterative solvers.
 *
 * This header provides Jacobi iterative solvers that work
 * with Eigen-like matrix/vector types and a preconditioner exposing a
 * `solve()` method.
 */

namespace LinearAlgebra
{

/**
 * @brief Standard Jacobi iterative solver.
 *
 * This templated routine performs classical Jacobi relaxation using a
 * user-supplied preconditioner `M` that must provide a `solve(Vector)`
 * method. The function updates `x` in-place and returns 0 on convergence
 * within the provided tolerance or 1 when the maximum iteration count is
 * reached without meeting the tolerance.
 *
 * @tparam Matrix Matrix-like type with `Scalar` typedef and `operator*`.
 * @tparam Vector Vector-like type with `Scalar` typedef, `.norm()` and size access.
 * @tparam Preconditioner Type providing `solve(const Vector&) -> Vector`.
 *
 * @param A System matrix.
 * @param x On input the initial guess; on output the computed approximation.
 * @param b Right-hand side vector.
 * @param M Preconditioner providing a `solve` method.
 * @param max_iter On input the maximum allowed iterations; on output the
 *                 number of iterations actually performed (or 0 if already converged).
 * @param tol On input the relative residual tolerance; on output the achieved relative residual.
 * @return int 0 if converged within `tol`, 1 otherwise.
 */
template <class Matrix, class Vector, class Preconditioner>
int Jacobi(const Matrix& A, Vector& x, const Vector& b, const Preconditioner& M,
           int& max_iter, typename Vector::Scalar& tol)
{
    using Real = typename Matrix::Scalar;
    Real resid;
    Real normb = b.norm();
    Vector r = b - A * x;

    if (normb == 0.0)
        normb = 1;
    if ((resid = r.norm() / normb) <= tol)
    {
        tol = resid;
        max_iter = 0;
        return 0;
    }

    for (int i = 1; i <= max_iter; i++)
    {
        x = M.solve(r) + x;
        r = b - A * x;
        if ((resid = r.norm() / normb) <= tol)
        {
            tol = resid;
            max_iter = i;
            return 0;
        }
    }

    tol = resid;
    return 1;
}

/**
 * @brief Parallel (row-wise) Jacobi variant.
 *
 * This variant attempts to parallelize the residual update across rows using
 * OpenMP. It retains the same interface and return semantics as `Jacobi`.
 *
 * @tparam Matrix Matrix-like type supporting row access and `rows()`.
 * @tparam Vector Vector-like type with `.norm()` and operator[] access.
 * @tparam Preconditioner Preconditioner type providing `solve(const Vector&)`.
 * @param A System matrix.
 * @param x Solution vector, updated in-place.
 * @param b Right-hand side vector.
 * @param M Preconditioner.
 * @param max_iter Max iterations (input) / iterations performed (output).
 * @param tol Residual tolerance (input) / achieved residual (output).
 * @return int 0 if converged, 1 if maximum iterations reached without convergence.
 */
template <class Matrix, class Vector, class Preconditioner>
int JacobiParallel(const Matrix& A, Vector& x, const Vector& b, const Preconditioner& M,
                   int& max_iter, typename Vector::Scalar& tol)
{
    using Real = typename Matrix::Scalar;
    Real resid;
    Real normb = b.norm();
    Vector r = b - A * x;

    if (normb == 0.0)
        normb = 1;
    if ((resid = r.norm() / normb) <= tol)
    {
        tol = resid;
        max_iter = 0;
        return 0;
    }

    for (int i = 1; i <= max_iter; i++)
    {
        x = M.solve(r) + x;

// r = b - A * x;
#pragma omp parallel for
        for (int i = 0; i < A.rows(); ++i)
        {
            r[i] = b[i] - A.row(i) * x;
        }

        if ((resid = r.norm() / normb) <= tol)
        {
            tol = resid;
            max_iter = i;
            return 0;
        }
    }

    tol = resid;
    return 1;
}
} // namespace LinearAlgebra
