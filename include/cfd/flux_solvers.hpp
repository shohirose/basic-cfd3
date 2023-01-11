#ifndef CFD_FLUX_SOLVERS_HPP
#define CFD_FLUX_SOLVERS_HPP

#include <Eigen/Core>
#include <tuple>

#include "cfd/functions.hpp"
#include "cfd/problem_parameters.hpp"

namespace cfd {

namespace detail {

struct PositiveFlux {
  template <typename Derived>
  Eigen::ArrayXd operator()(const Eigen::ArrayBase<Derived>& u) const noexcept {
    return u.max(0.0);
  }
};

struct NegativeFlux {
  template <typename Derived>
  Eigen::ArrayXd operator()(const Eigen::ArrayBase<Derived>& u) const noexcept {
    return u.min(0.0);
  }
};

}  // namespace detail

template <typename SpacialReconstructor>
class StegerWarmingRiemannSolver {
 public:
  StegerWarmingRiemannSolver(const ProblemParameters& params)
      : gamma_{params.specific_heat_ratio}, reconstructor_{params} {}

  template <typename Derived>
  Eigen::MatrixXd calc_flux(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    using Eigen::MatrixXd;
    const MatrixXd Ul = reconstructor_.calc_left(U);
    const MatrixXd Ur = reconstructor_.calc_right(U);
    const MatrixXd Fp = this->calc_flux_impl(Ul, detail::PositiveFlux{});
    const MatrixXd Fm = this->calc_flux_impl(Ur, detail::NegativeFlux{});
    return Fp + Fm;
  }

 private:
  /**
   * @brief Compute positive numerical flux
   *
   * @param U Conservation variables vector at the LHS of cell interfaces
   * @return Eigen::MatrixXd Numerical flux vector
   */
  template <typename Derived, typename F>
  Eigen::MatrixXd calc_flux_impl(const Eigen::MatrixBase<Derived>& U,
                                 F&& f) const noexcept {
    using Eigen::ArrayXd, Eigen::MatrixXd, Eigen::Map;

    Map<const ArrayXd> rho(&U(0, 0), U.rows());
    Map<const ArrayXd> rhou(&U(0, 1), U.rows());
    Map<const ArrayXd> rhoE(&U(0, 2), U.rows());

    const ArrayXd u = calc_velocity(rho, rhou);
    const ArrayXd p = calc_pressure(rho, u, rhoE, gamma_);
    const ArrayXd c = calc_sonic_velocity(rho, p, gamma_);

    const ArrayXd up = u + c;
    const ArrayXd um = u - c;
    const ArrayXd lambda1 = f(u);
    const ArrayXd lambda2 = f(up);
    const ArrayXd lambda3 = f(um);

    const ArrayXd x1 = ((gamma_ - 1.0) / gamma_) * rho * lambda1;
    const ArrayXd x2 = rho * lambda2 / (2 * gamma_);
    const ArrayXd x3 = rho * lambda3 / (2 * gamma_);
    const ArrayXd uu = 0.5 * u.square();
    const ArrayXd cc = (1 / (gamma_ - 1)) * c.square();
    const ArrayXd uc = u * c;

    MatrixXd F(U.rows(), 3);
    F.col(0) = (x1 + x2 + x3).matrix();
    F.col(1) = (x1 * u + x2 * up + x3 * um).matrix();
    F.col(2) = (x1 * uu + x2 * (uu + cc + uc) + x3 * (uu + cc - uc)).matrix();
    return F;
  }

  double gamma_;  ///> Specific heat ratio
  SpacialReconstructor reconstructor_;
};

template <typename SpacialReconstructor>
class RoeRiemannSolver {
 public:
  RoeRiemannSolver(const ProblemParameters& params)
      : gamma_{params.specific_heat_ratio}, reconstructor_{params} {}

  /**
   * @brief Compute numerical flux vector
   *
   * @tparam Derived
   * @param U Conservation variables
   * @return Eigen::MatrixXd Numerical flux
   */
  template <typename Derived>
  Eigen::MatrixXd calc_flux(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    using Eigen::MatrixXd;
    const MatrixXd Ul = reconstructor_.calc_left(U);
    const MatrixXd Ur = reconstructor_.calc_right(U);
    return this->calc_flux_impl(Ul, Ur);
  }

 private:
  /**
   * @brief Compute numerical flux
   *
   * @tparam Derived1
   * @tparam Derived2
   * @param Ul Conservation variables vector at the LHS of cell interfaces
   * @param Ur Conservation variables vector at the RHS of cell interfaces
   * @return Eigen::MatrixXd Numerical flux vector
   *
   * Ul and Ur have a shape of (n_domain_cells + 1, 3). Each column contains
   * density, moment density, and total energy density, respectively.
   *
   * U(:, 0) = density
   * U(:, 1) = moment density
   * U(:, 2) = total energy density
   *
   * F has a shape of (n_domain_cells + 1, 3). Each column contains
   * @f$ \rho u \f$, @f$ \rho u^2 + p @f$, and @f$ u (E^t + p) @f$, where
   * @f$ u @f$ is velocity, @f$ \rho @f$ is density, @f$ p @f$ is pressure,
   * and @f$ E^t @f$ is total energy density.
   */
  template <typename Derived1, typename Derived2>
  Eigen::MatrixXd calc_flux_impl(
      const Eigen::MatrixBase<Derived1>& Ul,
      const Eigen::MatrixBase<Derived2>& Ur) const noexcept {
    using Eigen::ArrayXd, Eigen::MatrixXd, Eigen::Map;

    Map<const ArrayXd> rhol(&Ul(0, 0), Ul.rows());
    Map<const ArrayXd> rhoul(&Ul(0, 1), Ul.rows());
    Map<const ArrayXd> rhoEl(&Ul(0, 2), Ul.rows());

    Map<const ArrayXd> rhor(&Ur(0, 0), Ur.rows());
    Map<const ArrayXd> rhour(&Ur(0, 1), Ur.rows());
    Map<const ArrayXd> rhoEr(&Ur(0, 2), Ur.rows());

    const ArrayXd ul = calc_velocity(rhol, rhoul);
    const ArrayXd ur = calc_velocity(rhor, rhour);
    const ArrayXd pl = calc_pressure(rhol, ul, rhoEl, gamma_);
    const ArrayXd pr = calc_pressure(rhor, ur, rhoEr, gamma_);
    const ArrayXd hl = calc_enthalpy(rhol, rhoEl, pl);
    const ArrayXd hr = calc_enthalpy(rhor, rhoEr, pr);
    const ArrayXd rho_m = calc_average_density(rhol, rhor);
    const ArrayXd w = rhol.sqrt() / (rhol.sqrt() + rhor.sqrt());
    const ArrayXd u_m = calc_average_velocity(ul, ur, w);
    const ArrayXd h_m = calc_average_enthalpy(hl, hr, w);
    const ArrayXd c_m = calc_average_sonic_velocity(h_m, u_m);

    const ArrayXd up = u_m + c_m;
    const ArrayXd um = u_m - c_m;
    const ArrayXd ld1 = calc_lambda(u_m);
    const ArrayXd ld2 = calc_lambda(up);
    const ArrayXd ld3 = calc_lambda(um);

    const ArrayXd dw1 = rhor - rhol - (pr - pl) / c_m.square();
    const ArrayXd dp = (pr - pl) / (rho_m * c_m);
    const ArrayXd a = 0.5 * rho_m / c_m;
    const ArrayXd dw2 = a * (ur - ul + dp);
    const ArrayXd dw3 = -a * (ur - ul - dp);

    const ArrayXd x1 = ld1 * dw1;
    const ArrayXd x2 = ld2 * dw2;
    const ArrayXd x3 = ld3 * dw3;

    MatrixXd F(u_m.size(), 3);
    F.col(0).array() = 0.5 * (rhoul + rhour) - 0.5 * (x1 + x2 + x3);
    F.col(1).array() = 0.5 * (rhoul * ul + pl + rhour * ur + pr) -
                       0.5 * (x1 * u_m + x2 * up + x3 * um);
    F.col(2).array() = 0.5 * ((rhoEl + pl) * ul + (rhoEr + pr) * ur) -
                       0.5 * (0.5 * x1 * u_m.square() + x2 * (h_m + c_m * u_m) +
                              x3 * (h_m - c_m * u_m));
    return F;
  }

  template <typename Derived1, typename Derived2>
  static Eigen::ArrayXd calc_average_density(
      const Eigen::ArrayBase<Derived1>& rhol,
      const Eigen::ArrayBase<Derived2>& rhor) noexcept {
    return (rhol * rhor).sqrt();
  }

  template <typename Derived1, typename Derived2, typename Derived3>
  static Eigen::ArrayXd calc_average_velocity(
      const Eigen::ArrayBase<Derived1>& ul,
      const Eigen::ArrayBase<Derived2>& ur,
      const Eigen::ArrayBase<Derived3>& w) noexcept {
    return ul * w + ur * (1 - w);
  }

  template <typename Derived1, typename Derived2, typename Derived3>
  static Eigen::ArrayXd calc_average_enthalpy(
      const Eigen::ArrayBase<Derived1>& hl,
      const Eigen::ArrayBase<Derived2>& hr,
      const Eigen::ArrayBase<Derived3>& w) noexcept {
    return hl * w + hr * (1 - w);
  }

  template <typename Derived1, typename Derived2>
  Eigen::ArrayXd calc_average_sonic_velocity(
      const Eigen::ArrayBase<Derived1>& h,
      const Eigen::ArrayBase<Derived2>& u) const noexcept {
    return ((gamma_ - 1) * (h - 0.5 * u.square())).sqrt();
  }

  template <typename Derived>
  static Eigen::ArrayXd calc_lambda(
      const Eigen::ArrayBase<Derived>& u) noexcept {
    constexpr double eps = 0.15;
    return u.abs().unaryExpr([eps](double x) {
      return (x > 2 * eps) ? x : (x * x / (4 * eps) + eps);
    });
  }

  double gamma_;  ///> Specific heat ratio
  SpacialReconstructor reconstructor_;
};

class LaxWendroffSolver {
 public:
  LaxWendroffSolver(const ProblemParameters& params)
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
    F.col(0) = rhou_m.matrix();
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

#endif  // CFD_FLUX_SOLVERS_HPP