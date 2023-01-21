#ifndef CFD_TIME_INTEGRATION_SCHEMES_HPP
#define CFD_TIME_INTEGRATION_SCHEMES_HPP

#include <Eigen/Core>
#include <type_traits>

#include "cfd/problem_parameters.hpp"

namespace cfd {

class LaxWendroffFluxCalculator;

/**
 * @brief Explicit Euler time integration method (first-order).
 *
 */
class ExplicitEulerTimeIntegrator {
 public:
  ExplicitEulerTimeIntegrator(const ProblemParameters& params)
      : dx_{params.dx},
        n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  /**
   * @brief Update variables using the explicit Euler method.
   *
   * @param U[in,out] Conservation variables
   * @param dt[in] Time step length
   * @param solver[in] Numerical flux solver
   * @param boundary[in] Boundary condition
   */
  template <typename Derived, typename FluxCalculator, typename Boundary>
  void update(Eigen::MatrixBase<Derived>& U, double dt,
              const FluxCalculator& flux,
              const Boundary& boundary) const noexcept {
    using Eigen::all, Eigen::seqN, Eigen::MatrixXd;
    const auto i = n_boundary_cells_;
    const auto n = n_domain_cells_;
    if constexpr (std::is_same_v<FluxCalculator, LaxWendroffFluxCalculator>) {
      const MatrixXd F = flux.compute(U, dt);
      U(seqN(i, n), all) -= (dt / dx_) * (F.bottomRows(n) - F.topRows(n));
    } else {
      const MatrixXd F = flux.compute(U);
      U(seqN(i, n), all) -= (dt / dx_) * (F.bottomRows(n) - F.topRows(n));
    }
    boundary.apply(U);
  }

 private:
  double dx_;             ///> Grid length
  int n_boundary_cells_;  ///> Number of boundary cells
  int n_domain_cells_;    ///> Number of domain cells
};

/**
 * @brief Second-order Runge-Kutta time integartion.
 *
 */
class RungeKutta2ndOrderTimeIntegrator {
 public:
  RungeKutta2ndOrderTimeIntegrator(const ProblemParameters& params)
      : dx_{params.dx},
        n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  /**
   * @brief Update conservation variables.
   *
   * @param U[in,out] Conservation variables
   * @param dt[in] Time step length
   * @param solver[in] Numerical flux solver
   * @param boundary[in] Boundary condition
   */
  template <typename Derived, typename FluxCalculator, typename Boundary>
  void update(Eigen::MatrixBase<Derived>& U, double dt,
              const FluxCalculator& flux,
              const Boundary& boundary) const noexcept {
    static_assert(!std::is_same_v<FluxCalculator, LaxWendroffFluxCalculator>,
                  "LaxWendroffFluxCalculator cannot be used.");

    using Eigen::all, Eigen::seqN, Eigen::MatrixXd;

    const auto i = n_boundary_cells_;
    const auto n = n_domain_cells_;
    const auto ntotal = 2 * n_boundary_cells_ + n_domain_cells_;
    const auto rng = seqN(i, n);

    // First step
    const MatrixXd F1 = flux.compute(U);
    const MatrixXd dU1 = (dt / dx_) * (F1.topRows(n) - F1.bottomRows(n));
    MatrixXd U1 = U(rng, all) + dU1;
    boundary.apply(U1);

    // Second step
    const MatrixXd F2 = flux.compute(U1);
    const MatrixXd dU2 = (dt / dx_) * (F2.topRows(n) - F2.bottomRows(n));
    U(rng, all) += 0.5 * (dU1 + dU2);
    boundary.apply(U);
  }

 private:
  double dx_;             ///> Grid length
  int n_boundary_cells_;  ///> Number of boundary cells
  int n_domain_cells_;    ///> Number of domain cells
};

}  // namespace cfd

#endif  // CFD_TIME_INTEGRATION_SCHEMES_HPP