#ifndef CFD_TIME_INTEGRATION_SCHEMES_HPP
#define CFD_TIME_INTEGRATION_SCHEMES_HPP

#include <Eigen/Core>
#include <type_traits>

#include "cfd/problem_parameters.hpp"

namespace cfd {

class LaxWendroffSolver;

class ExplicitEulerScheme {
 public:
  ExplicitEulerScheme(const ProblemParameters& params)
      : dx_{params.dx},
        n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  /**
   * @brief Update variables using the explicit Euler method.
   *
   * @tparam Derived
   * @param U[in,out] Conservation variables vector
   * @param dt[in] Time step length
   * @param solver[in] Numerical flux solver
   */
  template <typename Derived, typename FluxSolver, typename Boundary>
  void update(Eigen::MatrixBase<Derived>& U, double dt,
              const FluxSolver& solver,
              const Boundary& boundary) const noexcept {
    using Eigen::all, Eigen::seqN, Eigen::MatrixXd;
    const auto i = n_boundary_cells_;
    const auto n = n_domain_cells_;
    if constexpr (std::is_same_v<FluxSolver, LaxWendroffSolver>) {
      const MatrixXd F = solver.calc_flux(U, dt);
      U(seqN(i, n), all) -= (dt / dx_) * (F.bottomRows(n) - F.topRows(n));
    } else {
      const MatrixXd F = solver.calc_flux(U);
      U(seqN(i, n), all) -= (dt / dx_) * (F.bottomRows(n) - F.topRows(n));
    }
    boundary.apply(U);
  }

 private:
  double dx_;             ///> Grid length
  int n_boundary_cells_;  ///> Number of boundary cells
  int n_domain_cells_;    ///> Number of domain cells
};

class RungeKutta2ndOrderTimeIntegration {
 public:
  RungeKutta2ndOrderTimeIntegration(const ProblemParameters& params)
      : dx_{params.dx},
        n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  /**
   * @brief Update conservation variables.
   *
   * @tparam Derived
   * @param U[in,out] Conservation variables vector
   * @param dt[in] Time step length
   * @param solver[in] Numerical flux solver
   */
  template <typename Derived, typename FluxSolver, typename Boundary>
  void update(Eigen::MatrixBase<Derived>& U, double dt,
              const FluxSolver& solver,
              const Boundary& boundary) const noexcept {
    static_assert(!std::is_same_v<FluxSolver, LaxWendroffSolver>,
                  "LaxWendroffSolver cannot be used.");

    using Eigen::all, Eigen::seqN, Eigen::MatrixXd;

    const auto i = n_boundary_cells_;
    const auto n = n_domain_cells_;
    const auto ntotal = 2 * n_boundary_cells_ + n_domain_cells_;
    const auto rng = seqN(i, n);

    // First step
    const MatrixXd F1 = solver.calc_flux(U);
    const MatrixXd Lh1 = (1 / dx_) * (F1.topRows(n) - F1.bottomRows(n));
    const MatrixXd U1 = MatrixXd::Zero(ntotal, 3);
    U1(rng, all) = U(rng, all) + Lh1 * dt;
    boundary.apply(U1);

    // Second step
    const MatrixXd F2 = solver.calc_flux(U1);
    const MatrixXd Lh2 = (1 / dx_) * (F2.topRows(n) - F2.bottomRows(n));
    U(rng, all) += (0.5 / dt) * (Lh1 + Lh2);
    boundary.apply(U);
  }

 private:
  double dx_;             ///> Grid length
  int n_boundary_cells_;  ///> Number of boundary cells
  int n_domain_cells_;    ///> Number of domain cells
};

}  // namespace cfd

#endif  // CFD_TIME_INTEGRATION_SCHEMES_HPP