#ifndef CFD_TIME_INTEGRATION_SCHEMES_HPP
#define CFD_TIME_INTEGRATION_SCHEMES_HPP

#include <Eigen/Core>
#include <cassert>

#include "cfd/problem_parameters.hpp"

namespace cfd {

class ExplicitEulerScheme {
 public:
  /**
   * @brief Construct a new Explicit Euler Scheme object
   *
   * @param dx Grid length
   * @param gamma Specific heat ratio
   * @param n_boundary_cells Number of boundary cells
   * @param n_domain_cells Number of domain cells
   */
  ExplicitEulerScheme(double dx, double gamma, int n_boundary_cells,
                      int n_domain_cells)
      : dx_{dx},
        gamma_{gamma},
        n_boundary_cells_{n_boundary_cells},
        n_domain_cells_{n_domain_cells} {}

  ExplicitEulerScheme(const ProblemParameters& params)
      : dx_{params.dx},
        gamma_{params.specific_heat_ratio},
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
  template <typename Derived1, typename Derived2>
  void update(Eigen::DenseBase<Derived1>& U,
              const Eigen::DenseBase<Derived2>& F, double dt) const noexcept {
    using Eigen::all;
    using Eigen::seqN;
    const auto i = n_boundary_cells_;
    const auto n = n_domain_cells_;
    assert(U.rows() == 2 * i + n);
    assert(U.cols() == 3);
    assert(F.rows() == n + 1);
    assert(F.cols() == 3);
    U(seqN(i, n), all) -= (dt / dx_) * (F.bottomRows(n) - F.topRows(n));
  }

 private:
  double dx_;             ///> Grid length
  double gamma_;          ///> Specific heat ratio
  int n_boundary_cells_;  ///> Number of boundary cells
  int n_domain_cells_;    ///> Number of domain cells
};

}  // namespace cfd

#endif  // CFD_TIME_INTEGRATION_SCHEMES_HPP