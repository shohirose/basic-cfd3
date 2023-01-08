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
   * @param F[in] Numerical flux vector
   * @param dt[in] Time step length
   *
   * Conservation variables vector and numerical flux vector are given by
   * @f[
   * \mathbf{U} =
   * \begin{bmatrix}
   * \rho \\ \rho u \\ E^t
   * \end{bmatrix}
   * \quad
   * \mathbf{F} =
   * \begin{bmatrix}
   * \rho u \\
   * \rho u^2 + p \\
   * u ( E^t + p )
   * \end{bmatrix},
   * @f]
   * where @f$ \rho @f$ is density, @f$ u @f$ is velocity, @f$ p @f$ is
   * pressure, and @f$ E^t @f$ is total energy density.
   */
  template <typename Derived, typename FluxSolver>
  void update(Eigen::MatrixBase<Derived>& U, double dt,
              const FluxSolver& solver) const noexcept {
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
  }

 private:
  double dx_;             ///> Grid length
  int n_boundary_cells_;  ///> Number of boundary cells
  int n_domain_cells_;    ///> Number of domain cells
};

}  // namespace cfd

#endif  // CFD_TIME_INTEGRATION_SCHEMES_HPP