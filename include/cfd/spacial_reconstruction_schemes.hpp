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
    using Eigen::ArrayXd, Eigen::MatrixXd, Eigen::seqN, Eigen::Map;

    Map<const ArrayXd> rho(&U(0, 0), U.rows());
    Map<const ArrayXd> rhou(&U(0, 1), U.rows());
    Map<const ArrayXd> rhoE(&U(0, 2), U.rows());

    const auto i = n_boundary_cells_;
    const auto n = n_domain_cells_ + 1;
    const auto rng1 = seqN(i, n);
    const auto rng2 = seqN(i - 1, n);

    const ArrayXd rho_m = 0.5 * (rho(rng1) + rho(rng2)) -
                          (0.5 * dt / dx_) * (rhou(rng1) - rhou(rng2));

    const ArrayXd u = rhou / rho;
    const ArrayXd p = (gamma_ - 1) * (rhoE - 0.5 * rho * u.square());
    const ArrayXd f = rhou * u + p;
    const ArrayXd rhou_m = 0.5 * (rhou(rng1) + rhou(rng2)) -
                           (0.5 * dt / dx_) * (f(rng1) - f(rng2));
    const ArrayXd g = u * (rhoE + p);
    const ArrayXd rhoE_m = 0.5 * (rhoE(rng1) + rhoE(rng2)) -
                           (0.5 * dt / dx_) * (g(rng1) - g(rng2));
    const ArrayXd u_m = rhou_m / rho_m;
    const ArrayXd p_m = (gamma_ - 1) * (rhoE_m - 0.5 * rho_m * u_m.square());

    MatrixXd F(n, 3);
    F.col(0) = rhou_m;
    F.col(1) = (rhou_m * u_m + p_m).matrix();
    F.col(2) = (u_m * (rhoE_m + p_m)).matrix();
    return F;
  }

 private:
  double dx_;
  double gamma_;
  int n_boundary_cells_;
  int n_domain_cells_;
};

}  // namespace cfd

#endif  // CFD_SPACIAL_RECONSTRUCTION_SCHEMES_HPP