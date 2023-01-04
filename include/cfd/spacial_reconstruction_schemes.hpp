#ifndef CFD_SPACIAL_RECONSTRUCTION_SCHEMES_HPP
#define CFD_SPACIAL_RECONSTRUCTION_SCHEMES_HPP

#include <Eigen/Core>

#include "cfd/problem_parameters.hpp"

namespace cfd {

class FirstOrderSpacialReconstructor {
 public:
  FirstOrderSpacialReconstructor(int n_boundary_cells, int n_domain_cells)
      : n_boundary_cells_{n_boundary_cells},
        n_domain_cells_{n_boundary_cells} {}

  FirstOrderSpacialReconstructor(const ProblemParameters& params)
      : n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  template <typename Derived>
  Eigen::MatrixXd calc_left(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    assert(U.rows() == n_boundary_cells_ * 2 + n_domain_cells_);
    assert(U.cols() == 3);

    using Eigen::all, Eigen::seqN;
    return U(seqN(n_boundary_cells_ - 1, n_domain_cells_ + 1), all);
  }

  template <typename Derived>
  Eigen::MatrixXd calc_right(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    assert(U.rows() == n_boundary_cells_ * 2 + n_domain_cells_);
    assert(U.cols() == 3);

    using Eigen::all, Eigen::seqN;
    return U(seqN(n_boundary_cells_, n_domain_cells_ + 1), all);
  }

 private:
  int n_boundary_cells_;
  int n_domain_cells_;
};

class LaxWendroffSpacialReconstructor {
 public:
  LaxWendroffSpacialReconstructor(double dx, double gamma, int n_boundary_cells,
                                  int n_domain_cells)
      : dx_{dx},
        gamma_{gamma},
        n_boundary_cells_{n_boundary_cells},
        n_domain_cells_{n_boundary_cells} {}

  LaxWendroffSpacialReconstructor(const ProblemParameters& params)
      : dx_{params.dx},
        gamma_{params.specific_heat_ratio},
        n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  template <typename Derived>
  Eigen::MatrixXd calc_flux(const Eigen::MatrixBase<Derived>& U,
                            double dt) const noexcept {
    using Eigen::VectorXd, Eigen::MatrixXd, Eigen::seqN, Eigen::all;
    const VectorXd rho_s =
        calc_density_at_cell_interface(U.col(0), U.col(1), dt);
    const VectorXd p = calc_pressure(U.col(1).array(), U.col(0).array(),
                                     U.col(2).array(), gamma_)
                           .matrix();
    const VectorXd u =
        calc_velocity(U.col(1).array(), U.col(0).array()).matrix();
    const VectorXd rhou_s =
        calc_momentum_density_at_cell_interface(U.col(1), u, p, dt);
    const VectorXd et_s =
        calc_total_energy_density_at_cell_interface(u, U.col(2), p, dt);
    const VectorXd p_s =
        calc_pressure(rhou_s.array(), rho_s.array(), et_s.array(), gamma_)
            .matrix();
    const VectorXd u_s = calc_velocity(rhou_s.array(), rho_s.array()).matrix();
    MatrixXd F(n_domain_cells_ + 1, 3);
    F.col(0) = rhou_s;
    F.col(1) = (rhou_s.array() * u_s.array() + p_s.array()).matrix();
    F.col(2) = (u_s.array() * (et_s.array() + p_s.array())).matrix();
    return F;
  }

 private:
  /**
   * @brief Compute density at cell interfaces
   *
   * @tparam Derived1
   * @tparam Derived2
   * @param rho Density
   * @param rhou Momentum density
   * @param dt Time step length
   * @return Eigen::VectorXd
   */
  template <typename Derived1, typename Derived2>
  Eigen::VectorXd calc_density_at_cell_interface(
      const Eigen::MatrixBase<Derived1>& rho,
      const Eigen::MatrixBase<Derived2>& rhou, double dt) const noexcept {
    using Eigen::seqN;
    const auto i = n_boundary_cells_;
    const auto n = n_domain_cells_ + 1;
    return 0.5 * (rho(seqN(i, n)) + rho(seqN(i - 1, n))) -
           (0.5 * dt / dx_) * (rhou(seqN(i, n)) - rhou(seqN(i - 1, n)));
  }

  /**
   * @brief Compute momentum density at cell interfaces
   *
   * @tparam Derived1
   * @tparam Derived2
   * @tparam Derived3
   * @param rhou Momentum density
   * @param u Velocity
   * @param p Pressure
   * @param dt Time step length
   * @return Eigen::VectorXd
   */
  template <typename Derived1, typename Derived2, typename Derived3>
  Eigen::VectorXd calc_momentum_density_at_cell_interface(
      const Eigen::MatrixBase<Derived1>& rhou,
      const Eigen::MatrixBase<Derived2>& u,
      const Eigen::MatrixBase<Derived3>& p, double dt) const noexcept {
    using Eigen::seqN, Eigen::VectorXd;
    const auto i = n_boundary_cells_;
    const auto n = n_domain_cells_ + 1;
    const VectorXd f = (rhou.array() * u.array() + p.array()).matrix();
    return 0.5 * (rhou(seqN(i, n)) + rhou(seqN(i - 1, n))) -
           (0.5 * dt / dx_) * (f(seqN(i, n)) - f(seqN(i - 1, n)));
  }

  /**
   * @brief Compute total energy density at cell interfaces
   *
   * @tparam Derived1
   * @tparam Derived2
   * @tparam Derived3
   * @param u Velocity
   * @param et Total energy density
   * @param p Pressure
   * @param dt Time step length
   * @return Eigen::VectorXd
   */
  template <typename Derived1, typename Derived2, typename Derived3>
  Eigen::VectorXd calc_total_energy_density_at_cell_interface(
      const Eigen::MatrixBase<Derived1>& u,
      const Eigen::MatrixBase<Derived2>& et,
      const Eigen::MatrixBase<Derived3>& p, double dt) const noexcept {
    using Eigen::seqN, Eigen::VectorXd;
    const auto i = n_boundary_cells_;
    const auto n = n_domain_cells_ + 1;
    const VectorXd f = (u.array() * (et.array() + p.array())).matrix();
    return 0.5 * (et(seqN(i, n)) + et(seqN(i - 1, n))) -
           (0.5 * dt / dx_) * (f(seqN(i, n)) - f(seqN(i - 1, n)));
  }

  double dx_;
  double gamma_;
  int n_boundary_cells_;
  int n_domain_cells_;
};

}  // namespace cfd

#endif  // CFD_SPACIAL_RECONSTRUCTION_SCHEMES_HPP