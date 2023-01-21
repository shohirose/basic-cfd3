#ifndef CFD_RIEMANN_SOLVERS_HPP
#define CFD_RIEMANN_SOLVERS_HPP

#include <Eigen/Core>

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

/**
 * @brief Riemann solver using Flux Vector Splitting scheme.
 */
class StegerWarmingRiemannSolver {
 public:
  StegerWarmingRiemannSolver(const ProblemParameters& params)
      : gamma_{params.specific_heat_ratio} {}

  /**
   * @brief Compute flux
   *
   * @param Ul Conservation variables at the LHS of cell interfaces
   * @param Ur Conservation variables at the RHS of cell interfaces
   * @return Flux at cell interfaces
   */
  template <typename Derived1, typename Derived2>
  Eigen::MatrixXd calc_flux(
      const Eigen::MatrixBase<Derived1>& Ul,
      const Eigen::MatrixBase<Derived2>& Ur) const noexcept {
    using Eigen::MatrixXd;
    const MatrixXd Fp = this->calc_flux_impl(Ul, detail::PositiveFlux{});
    const MatrixXd Fm = this->calc_flux_impl(Ur, detail::NegativeFlux{});
    return Fp + Fm;
  }

 private:
  /**
   * @brief Compute positive numerical
   *
   * @param U Conservation variables at cell interfaces
   * @return Numerical flux
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
};

/**
 * @brief Riemann solver using Flux Difference Splitting scheme.
 */
class RoeRiemannSolver {
 public:
  RoeRiemannSolver(const ProblemParameters& params)
      : gamma_{params.specific_heat_ratio} {}

  /**
   * @brief Compute numerical flux
   *
   * @param Ul Conservation variables at the LHS of cell interfaces
   * @param Ur Conservation variables at the RHS of cell interfaces
   * @return Numerical flux
   */
  template <typename Derived1, typename Derived2>
  Eigen::MatrixXd calc_flux(
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

 private:
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
};

}  // namespace cfd

#endif  // CFD_RIEMANN_SOLVERS_HPP