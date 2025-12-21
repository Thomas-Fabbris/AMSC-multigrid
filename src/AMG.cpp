#include "AMG.hpp"

// Perform one V-cycle starting from initial guess x, right-hand side b
void AMG::vcycle(SpVec& x, const SpVec& b, int preSmooth, int postSmooth, double tol) const
{
    if (levels_.empty())
        return;
    vcycleRecursive(0, x, b, preSmooth, postSmooth, tol);
}

// Solve with repeated V-cycles (stop early when residual falls below tol)
// Returns the number of iterations performed
int AMG::solve(SpVec& x, const SpVec& b, int cycles, int preSmooth, int postSmooth, double tol) const
{
    if (levels_.empty() || cycles <= 0)
        return 0;

    const SpMat& A0 = levels_.front().A;
    int k = 0;
    for (; k < cycles; ++k)
    {
        SpVec r = b - A0 * x;
        if (r.norm() <= tol)
            break;

        vcycle(x, b, preSmooth, postSmooth, tol);
    }
    return k;
}

const std::vector<AMG::Level>& AMG::levels() const
{
    return levels_;
}

void AMG::buildHierarchy(const SpMat& A0)
{
    levels_.clear();
    SpMat Af = A0;

    for (int ell = 0; ell < maxLevels_; ++ell)
    {
        Level L;
        L.A = Af;
        L.strong = computeStrongConnections(Af, theta_);
        L.active = Eigen::VectorXi::Ones(Af.rows());

        // Stop if size too small
        if (Af.rows() <= minCoarseSize_)
        {
            levels_.push_back(std::move(L));
            break;
        }

        // Coarsen: compute C/F splitting and interpolation
        Eigen::VectorXi isCoarse = coarsenCF(L.strong);
        L.P = buildInterpolation(Af, L.strong, isCoarse);
        L.R = buildRestriction(L.P);

        // coarse operator
        SpMat Ac = L.R * Af * L.P;

        levels_.push_back(std::move(L));

        Af = Ac;
        if (Af.rows() == 0 || Af.cols() == 0)
            break;
    }
}

// Strong connections: i strongly depends on j if |a_ij| >= theta * max_{k \neq i} |a_ik|
Eigen::SparseMatrix<int> AMG::computeStrongConnections(const SpMat& A, double theta) const
{
    using Trip = Eigen::Triplet<int>;

#ifdef _OPENMP
    // Prepare thread-local buffers
    std::vector<std::vector<Trip>> threadTrips;

#pragma omp parallel
    {
// Allocate buffers for the actual number of threads in this team
#pragma omp single
        {
            threadTrips.resize(omp_get_num_threads());
        }
        // Implicit barrier ensures resize is complete before any thread proceeds

        const int tid = omp_get_thread_num();
        auto& local = threadTrips[tid];
        local.reserve(std::max<int>(1, A.nonZeros() / threadTrips.size()));

#pragma omp for schedule(static)
        for (int i = 0; i < A.rows(); ++i)
        {
            double maxOff = 0.0; // row-local maximum; keep inside loop (strong threshold is per row)
            for (SpMat::InnerIterator it(A, i); it; ++it)
            {
                if (it.col() != i)
                    maxOff = std::max(maxOff, -it.value());
            }
            const double thresh = theta * maxOff;

            if (maxOff > 0.0)
            {
                for (SpMat::InnerIterator it(A, i); it; ++it)
                {
                    if (it.col() == i)
                        continue;
                    if (-it.value() >= thresh)
                        local.emplace_back(i, it.col(), 1);
                }
            }
        }
    }

    // Merge thread-local triplets
    std::vector<Trip> trips;
    size_t total = 0;
    for (const auto& v : threadTrips)
        total += v.size();
    trips.reserve(total);
    for (auto& v : threadTrips)
        trips.insert(trips.end(), v.begin(), v.end());
#else
    // Serial fallback
    std::vector<Trip> trips;
    trips.reserve(A.nonZeros());
    for (int i = 0; i < A.rows(); ++i)
    {
        double maxOff = 0.0;
        for (SpMat::InnerIterator it(A, i); it; ++it)
        {
            if (it.col() != i)
                maxOff = std::max(maxOff, -it.value());
        }
        const double thresh = theta * maxOff;

        for (SpMat::InnerIterator it(A, i); it; ++it)
        {
            if (it.col() == i)
                continue;
            if (-it.value() >= thresh && maxOff > 0.0)
                trips.emplace_back(i, it.col(), 1);
        }
    }
#endif

    Eigen::SparseMatrix<int> S(A.rows(), A.cols());
    S.setFromTriplets(trips.begin(), trips.end());
    S.makeCompressed();
    return S;
}

// Simple CF splitting: select coarse points via greedy degree-based heuristic on strong graph
Eigen::VectorXi AMG::coarsenCF(const Eigen::SparseMatrix<int>& strong) const
{
    const int n = strong.rows();
    Eigen::VectorXi isCoarse = Eigen::VectorXi::Zero(n);
    std::vector<int> degree(n, 0); // lambda BoomerAMG
    for (int i = 0; i < n; ++i)
        for (Eigen::SparseMatrix<int>::InnerIterator it(strong, i); it; ++it)
            degree[i]++;

    std::vector<char> chosen(n, 0);
    // Greedy: pick nodes with highest degree, mark neighbors as fine
    for (int iter = 0; iter < n; ++iter)
    {
        // get node with best degree
        int best = -1;
        int bestDeg = -1;
        for (int i = 0; i < n; ++i)
        {
            if (chosen[i]) // skip already chosen coarse or fine
                continue;
            if (degree[i] > bestDeg)
            {
                bestDeg = degree[i];
                best = i;
            }
        }
        if (best < 0)
            break;
        // make coarse the best node
        isCoarse[best] = 1;
        chosen[best] = 1;
        // neighbours of a coarse point are fine
        for (Eigen::SparseMatrix<int>::InnerIterator it(strong, best); it; ++it)
            chosen[it.col()] = 1;
    }
    // Ensure at least one coarse point
    if (isCoarse.sum() == 0 && n > 0)
        isCoarse[0] = 1;
    return isCoarse;
}

// Interpolation: classical direct interpolation using strong connections
SpMat AMG::buildInterpolation(const SpMat& A, const Eigen::SparseMatrix<int>& strong, const Eigen::VectorXi& isCoarse) const
{
    const int n = A.rows();
    const int nC = isCoarse.sum();

    // Map coarse indices to columns in P
    Eigen::VectorXi coarseIndex = Eigen::VectorXi::Constant(n, -1);
    int col = 0;
    for (int i = 0; i < n; ++i)
        if (isCoarse[i])
            coarseIndex[i] = col++;

    // Use per-thread triplet buffers to avoid synchronization in the main loop
    std::vector<Eigen::Triplet<double>> trips; // final container

// Heuristic: expected number of entries roughly equals number of nonzeros
// We'll collect in thread-local vectors and merge afterwards
#ifdef _OPENMP
    std::vector<std::vector<Eigen::Triplet<double>>> threadTrips;
#pragma omp parallel
    {
#pragma omp single
        {
            threadTrips.resize(omp_get_num_threads());
        }
        // Implicit barrier

        const int tid = omp_get_thread_num();
        auto& local = threadTrips[tid];
        // Reserve some capacity to reduce reallocations
        local.reserve(std::max<int>(1, A.nonZeros() / threadTrips.size()));

#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i)
        {
            if (isCoarse[i])
            {
                // Inject coarse points
                local.emplace_back(i, coarseIndex[i], 1.0);
                continue;
            }

            // Sum of strong negative off-diagonals
            double sumStrongNeg = 0.0;
            for (SpMat::InnerIterator itA(A, i); itA; ++itA)
            {
                int j = itA.col();
                if (j == i)
                    continue;
                const bool strongConn = strong.coeff(i, j) != 0;
                // Fix: Only sum weights from COARSE strong neighbors to ensure partition of unity
                if (strongConn && isCoarse[j] && itA.value() < 0.0)
                    sumStrongNeg += std::abs(itA.value());
            }

            // If sum of strong negatives is zero: connect to one coarse neighbor
            if (sumStrongNeg <= std::numeric_limits<double>::epsilon())
            {
                for (Eigen::SparseMatrix<int>::InnerIterator itS(strong, i); itS; ++itS)
                {
                    int j = itS.col();
                    if (isCoarse[j])
                    {
                        local.emplace_back(i, coarseIndex[j], 1.0);
                        break;
                    }
                }
                continue;
            }

            // Weights over strong coarse neighbors
            for (SpMat::InnerIterator itA(A, i); itA; ++itA)
            {
                int j = itA.col();
                if (j == i)
                    continue;
                const bool strongConn = strong.coeff(i, j) != 0;
                if (strongConn && isCoarse[j])
                {
                    double w = std::abs(itA.value()) / sumStrongNeg;
                    local.emplace_back(i, coarseIndex[j], w);
                }
            }
        }
    }
    // Merge thread-local triplets
    size_t total = 0;
    for (const auto& v : threadTrips)
        total += v.size();
    trips.reserve(total);
    for (auto& v : threadTrips)
    {
        trips.insert(trips.end(), v.begin(), v.end());
    }
#else
    // Serial if we don't have OpenMP
    trips.reserve(A.nonZeros());
    for (int i = 0; i < n; ++i)
    {
        if (isCoarse[i])
        {
            // Inject coarse points
            trips.emplace_back(i, coarseIndex[i], 1.0);
            continue;
        }

        // Sum of strong negative off-diagonals
        double sumStrongNeg = 0.0;
        for (SpMat::InnerIterator itA(A, i); itA; ++itA)
        {
            int j = itA.col();
            if (j == i)
                continue;
            const bool strongConn = strong.coeff(i, j) != 0;
            // Fix: Only sum weights from COARSE strong neighbors
            if (strongConn && isCoarse[j] && itA.value() < 0.0)
                sumStrongNeg += std::abs(itA.value());
        }

        if (sumStrongNeg <= std::numeric_limits<double>::epsilon())
        {
            for (Eigen::SparseMatrix<int>::InnerIterator itS(strong, i); itS; ++itS)
            {
                int j = itS.col();
                if (isCoarse[j])
                {
                    trips.emplace_back(i, coarseIndex[j], 1.0);
                    break;
                }
            }
            continue;
        }

        for (SpMat::InnerIterator itA(A, i); itA; ++itA)
        {
            int j = itA.col();
            if (j == i)
                continue;
            const bool strongConn = strong.coeff(i, j) != 0;
            if (strongConn && isCoarse[j])
            {
                double w = std::abs(itA.value()) / sumStrongNeg;
                trips.emplace_back(i, coarseIndex[j], w);
            }
        }
    }
#endif

    SpMat P(n, nC);
    P.setFromTriplets(trips.begin(), trips.end());
    P.makeCompressed();
    return P;
}

// Restriction as transpose of interpolation (standard choice)
SpMat AMG::buildRestriction(const SpMat& P) const
{
    return P.transpose();
}

// Jacobi smoother using existing implementation in ⁠ jacobi.hpp ⁠
void AMG::jacobiSmooth(const SpMat& A, SpVec& x, const SpVec& b, int sweeps, double tol) const
{
    DiagPrec M(A);
    int max_iter = sweeps;
    double eps = tol;
    LinearAlgebra::Jacobi(A, x, b, M, max_iter, eps);
}

void AMG::vcycleRecursive(int level, SpVec& x, const SpVec& b, int pre, int post, double tol) const
{
    const Level& L = levels_[level];

    // Pre-smoothing
    jacobiSmooth(L.A, x, b, pre, tol);

    // Compute residual
    SpVec r = b - L.A * x;

    // Coarsest level: solve directly
    if (level == static_cast<int>(levels_.size()) - 1)
    {
        Eigen::SparseLU<SpMat> solver;
        solver.compute(L.A);
        if (solver.info() == Eigen::Success)
        {
            x = solver.solve(b);
        }
        return;
    }

    // Restrict residual
    SpVec rc = L.R * r;

    // Solve on coarse level
    SpVec ec = SpVec::Zero(rc.size());
    vcycleRecursive(level + 1, ec, rc, pre, post, tol);

    // Prolongate and correct
    x += L.P * ec;

    // Post-smoothing
    jacobiSmooth(L.A, x, b, post, tol);
}

int AMG::num_levels() const
{
    return static_cast<int>(levels_.size());
}
