/**
 * @file AMG.hpp
 * @brief Algebraic Multigrid (AMG) hierarchy skeleton using Eigen.
 *
 * This header defines the `AMG` class which builds a multilevel hierarchy
 * for solving sparse linear systems using algebraic multigrid techniques.
 * The implementation provides a Jacobi smoother (via `jacobi.hpp`), a
 * simple C/F coarsening strategy, construction of interpolation
 * and restriction operators, and storage for level matrices and connectivity.
 *
 */

#ifndef AMG_HPP
#define AMG_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <limits>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

#include "jacobi.hpp"

using SpMat = Eigen::SparseMatrix<double>;
using SpVec = Eigen::VectorXd;
using DiagPrec = Eigen::DiagonalPreconditioner<double>;
using IdentityPrec = Eigen::IdentityPreconditioner;

/**
 * @brief Algebraic Multigrid (AMG) solver helper.
 *
 * The `AMG` class stores a hierarchy of levels used for multigrid V-cycles.
 * Each level contains its system matrix, interpolation/restriction operators,
 * an indicator vector of active (fine) nodes, and a graph of strong
 * connections used during coarsening and interpolation construction.
 */
class AMG
{
  public:
    struct Level
    {
        /**
         * @brief System matrix at this level.
         *
         * The matrix represents the discretized operator on the current
         * multigrid level.
         */
        SpMat A;

        /**
         * @brief Interpolation operator to the finer level.
         *
         * Maps coarse-level corrections up to the finer grid.
         */
        SpMat P;

        /**
         * @brief Restriction operator to the coarser level.
         */
        SpMat R;

        /**
         * @brief Active-node flags for the level.
         *
         * A vector of integers used as boolean indicators marking
         * which fine-level unknowns are active.
         */
        Eigen::VectorXi active;

        /**
         * @brief Strong-connections adjacency graph.
         *
         * An integer sparse matrix representing the graph of strong
         * couplings used for coarsening and interpolation decisions.
         */
        Eigen::SparseMatrix<int> strong;
    };

    /**
     * @brief Construct an AMG hierarchy from a finest-level matrix.
     *
     * @param A0 Finest-level system matrix.
     * @param maxLevels Maximum number of multigrid levels to construct.
     * @param minCoarseSize Minimum size of the coarsest level to stop coarsening.
     * @param strengthTheta Threshold for determining strong connections.
     */
    AMG(const SpMat& A0, int maxLevels = 10, int minCoarseSize = 10, double strengthTheta = 0.25)
        : maxLevels_(maxLevels), minCoarseSize_(minCoarseSize), theta_(strengthTheta)
    {
        buildHierarchy(A0);
    }

    /**
     * @brief Perform a single multigrid V-cycle.
     *
     * Applies pre- and post-smoothing and performs coarse-grid correction
     * recursively across the AMG hierarchy starting from the finest level.
     *
     * @param x Initial guess on input; solution estimate on output.
     * @param b Right-hand side vector for the finest level.
     * @param preSmooth Number of pre-smoothing sweeps.
     * @param postSmooth Number of post-smoothing sweeps.
     * @param tol Tolerance for the residual norm to stop the AMG iterations.
     */
    void vcycle(SpVec& x, const SpVec& b, int preSmooth = 2, int postSmooth = 2, double tol = 1e-14) const;

    /**
     * @brief Solve using repeated V-cycles starting from initial guess `x`.
     *
     * Calls `vcycle` multiple times to iteratively reduce the residual.
     *
     * @param x Initial guess on input; solution estimate on output.
     * @param b Right-hand side vector.
     * @param cycles Number of V-cycles to perform.
     * @param preSmooth Number of pre-smoothing sweeps per cycle.
     * @param postSmooth Number of post-smoothing sweeps per cycle.
     * @param tol Tolerance for the residual norm to stop the AMG iterations
     */
    int solve(SpVec& x, const SpVec& b, int cycles = 10, int preSmooth = 2, int postSmooth = 2, double tol = 1e-14) const;

    /**
     * @brief Access the internal levels of the hierarchy.
     * @return Const reference to the vector of levels.
     */
    const std::vector<Level>& levels() const;

    /**
     * @brief Number of levels currently built in the hierarchy.
     * @return Integer count of levels.
     */
    int num_levels() const;

  private:
    std::vector<Level> levels_;
    int maxLevels_;
    int minCoarseSize_;
    double theta_; // strong connection threshold

    /**
     * @brief Build the multigrid hierarchy starting from the finest matrix `A0`.
     *
     * This populates `levels_` with progressively coarser matrices and
     * interpolation/restriction operators until stopping criteria are met.
     *
     * @param A0 Finest-level sparse matrix.
     */
    void buildHierarchy(const SpMat& A0);

    /**
     * @brief Compute the strong-connection graph for matrix `A0`.
     *
     * Entry (i,j) in the returned integer sparse matrix is nonzero when
     * node i is considered to strongly depend on node j. The criterion
     * follows the standard AMG definition.
     *
     * @param A System matrix to analyze.
     * @param theta Relative threshold for determining strong coupling.
     * @return Integer sparse adjacency matrix of strong connections.
     */
    Eigen::SparseMatrix<int> computeStrongConnections(const SpMat& A, double theta) const;

    /**
     * @brief Compute a C/F splitting using a simple greedy heuristic.
     *
     * Uses the strong-connection graph to choose coarse points.
     *
     * @param strong Strong-connections adjacency produced by `computeStrongConnections`.
     * @return Integer vector with entries 1 for coarse nodes and 0 for fine nodes.
     */
    Eigen::VectorXi coarsenCF(const Eigen::SparseMatrix<int>& strong) const;

    /**
     * @brief Build the interpolation (prolongation) operator from `A`.
     *
     * Uses classical direct interpolation based on strong connections and the
     * coarse-point indicator `isCoarse`.
     *
     * @param A System matrix on the current level.
     * @param strong Strong-connections graph for this level.
     * @param isCoarse Indicator vector marking coarse nodes.
     * @return Prolongation operator as a sparse matrix.
     */
    SpMat buildInterpolation(const SpMat& A, const Eigen::SparseMatrix<int>& strong, const Eigen::VectorXi& isCoarse) const;

    /**
     * @brief Build restriction operator.
     * @param P Prolongation matrix to transform into restriction.
     * @return Restriction operator for coarse-grid injection.
     */
    SpMat buildRestriction(const SpMat& P) const;

    /**
     * @brief Apply Jacobi smoothing to approximate solve on a level.
     *
     * Delegates to the implementation provided in `jacobi.hpp`.
     *
     * @param A System matrix.
     * @param x Current solution estimate.
     * @param b Right-hand side vector.
     * @param sweeps Number of Jacobi sweeps to perform.
     * @param tol Tolerance to allow early termination.
     */
    void jacobiSmooth(const SpMat& A, SpVec& x, const SpVec& b, int sweeps, double tol) const;

    /**
     * @brief Internal recursive V-cycle implementation.
     *
     * @param level Current level index (0 = finest).
     * @param x Solution vector for this level.
     * @param b Right-hand side for this level.
     * @param pre Number of pre-smoothing sweeps.
     * @param post Number of post-smoothing sweeps.
     * @param tol Smoother tolerance.
     */
    void vcycleRecursive(int level, SpVec& x, const SpVec& b, int pre, int post, double tol) const;
};

#endif // AMG_HPP
