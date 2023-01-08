#ifndef CFD_RIEMANN_SOLVERS_HPP
#define CFD_RIEMANN_SOLVERS_HPP

#include <Eigen/Core>
#include <tuple>

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

class NoRiemannSolver {};

/**
 * @brief Steger-Warming Riemann solver
 *
 * This Riemann solver computes numerical flux using the flux vector splitting
 * scheme.
 */
class StegerWarmingRiemannSolver {
 public:
  /**
   * @brief Construct a new Steger Warming Riemann Solver object
   *
   * @param gamma Specific heat ratio
   */
  StegerWarmingRiemannSolver(double gamma) : gamma_{gamma} {}

  /**
   * @brief Construct a new Steger Warming Riemann Solver object
   *
   * @param params Problem parameters
   */
  StegerWarmingRiemannSolver(const ProblemParameters& params)
      : gamma_{params.specific_heat_ratio} {}

  /**
   * @brief Compute numerical flux at cell interfaces
   *
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
   * @brief Compute positive numerical flux
   *
   * @param Ul Conservation variables vector at the LHS of cell interfaces
   * @return Eigen::MatrixXd Numerical flux vector
   */
  template <typename Derived, typename F>
  Eigen::MatrixXd calc_flux_impl(const Eigen::MatrixBase<Derived>& U,
                                 F&& f) const noexcept {
    using Eigen::ArrayXd, Eigen::MatrixXd, Eigen::Map;

    Map<const ArrayXd> rho(&U(0, 0), U.rows());
    Map<const ArrayXd> rhou(&U(0, 1), U.rows());
    Map<const ArrayXd> rhoE(&U(0, 2), U.rows());

    const ArrayXd u = rhou / rho;
    const ArrayXd p = (gamma_ - 1) * (rhoE - 0.5 * rho * u.square());
    const ArrayXd c = (gamma_ * p / rho).sqrt();

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

class RoeRiemannSolver {
 public:
  /**
   * @brief Construct a new Roe Riemann Solver object
   *
   * @param gamma Specific heat ratio
   */
  RoeRiemannSolver(double gamma) : gamma_{gamma} {}

  RoeRiemannSolver(const ProblemParameters& params)
      : gamma_{params.specific_heat_ratio} {}

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
    const ArrayXd pl = calc_pressure(rhol, rhoEl, ul);
    const ArrayXd pr = calc_pressure(rhor, rhoEr, ur);
    const ArrayXd hl = calc_enthalpy(rhol, ul, pl);
    const ArrayXd hr = calc_enthalpy(rhor, ur, pr);
    const ArrayXd rho_m = calc_average_density(rhol, rhor);
    const ArrayXd u_m = calc_average_velocity(ul, ur, rhol, rhor);
    const ArrayXd h_m = calc_average_enthalpy(hl, hr, rhol, rhor);
    const ArrayXd c_m = calc_average_sonic_velocity(h_m, u_m);

    const ArrayXd up = u_m + c_m;
    const ArrayXd um = u_m - c_m;
    const ArrayXd ld1 = calc_lambda(u_m);
    const ArrayXd ld2 = calc_lambda(up);
    const ArrayXd ld3 = calc_lambda(um);

    const ArrayXd dw1 = rhor - rhol - (pr - pl) / c_m.square();
    const ArrayXd dw2 = 0.5 * (rho_m * (ur - ul) + (pr - pl) / c_m.square());
    const ArrayXd dw3 = 0.5 * (-rho_m * (ur - ul) + (pr - pl) / c_m.square());

    const auto [R1, R2, R3] = this->calc_r(u_m, h_m, c_m, up, um);

    const auto Fl = this->calc_flux(ul, rhol, pl, rhoEl);
    const auto Fr = this->calc_flux(ur, rhor, pr, rhoEr);

    const MatrixXd dF1 =
        (ld1 * dw1).matrix().replicate<1, 3>().cwiseProduct(R1);
    const MatrixXd dF2 =
        (ld2 * dw2).matrix().replicate<1, 3>().cwiseProduct(R2);
    const MatrixXd dF3 =
        (ld3 * dw3).matrix().replicate<1, 3>().cwiseProduct(R3);

    return 0.5 * ((Fl + Fr) - (dF1 + dF2 + dF3));
  }

 private:
  template <typename Derived1, typename Derived2>
  static Eigen::ArrayXd calc_velocity(
      const Eigen::ArrayBase<Derived1>& rho,
      const Eigen::ArrayBase<Derived2>& rhou) noexcept {
    return rhou / rho;
  }

  template <typename Derived1, typename Derived2, typename Derived3>
  Eigen::ArrayXd calc_pressure(
      const Eigen::ArrayBase<Derived1>& rho,
      const Eigen::ArrayBase<Derived2>& rhoE,
      const Eigen::ArrayBase<Derived3>& u) const noexcept {
    return (gamma_ - 1) * (rhoE - 0.5 * rho * u.square());
  }

  template <typename Derived1, typename Derived2, typename Derived3>
  Eigen::ArrayXd calc_enthalpy(
      const Eigen::ArrayBase<Derived1>& rho,
      const Eigen::ArrayBase<Derived2>& u,
      const Eigen::ArrayBase<Derived3>& p) const noexcept {
    return 0.5 * rho * u.square() + p / (gamma_ - 1);
  }

  template <typename Derived1, typename Derived2>
  static Eigen::ArrayXd calc_average_density(
      const Eigen::ArrayBase<Derived1>& rhol,
      const Eigen::ArrayBase<Derived2>& rhor) noexcept {
    return (rhol * rhor).sqrt();
  }

  template <typename Derived1, typename Derived2, typename Derived3,
            typename Derived4>
  static Eigen::ArrayXd calc_average_velocity(
      const Eigen::ArrayBase<Derived1>& ul,
      const Eigen::ArrayBase<Derived2>& ur,
      const Eigen::ArrayBase<Derived3>& rhol,
      const Eigen::ArrayBase<Derived4>& rhor) noexcept {
    const Eigen::ArrayXd w = 1 / (1 + (rhor / rhol).sqrt());
    return ul * w + ur * (1 - w);
  }

  template <typename Derived1, typename Derived2, typename Derived3,
            typename Derived4>
  static Eigen::ArrayXd calc_average_enthalpy(
      const Eigen::ArrayBase<Derived1>& hl,
      const Eigen::ArrayBase<Derived2>& hr,
      const Eigen::ArrayBase<Derived3>& rhol,
      const Eigen::ArrayBase<Derived4>& rhor) noexcept {
    const Eigen::ArrayXd w = 1 / (1 + (rhor / rhol).sqrt());
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

  template <typename Derived1, typename Derived2, typename Derived3,
            typename Derived4>
  static Eigen::MatrixXd calc_flux(
      const Eigen::ArrayBase<Derived1>& u,
      const Eigen::ArrayBase<Derived2>& rho,
      const Eigen::ArrayBase<Derived3>& p,
      const Eigen::ArrayBase<Derived4>& e) noexcept {
    Eigen::MatrixXd F(u.size(), 3);
    F.col(0) = (rho * u).matrix();
    F.col(1) = (rho * u.square() + p).matrix();
    F.col(2) = ((e + p) * u).matrix();
    return F;
  }

  /**
   * @brief Compute eigenvectors
   */
  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> calc_r(
      const Eigen::ArrayXd& u_m, const Eigen::ArrayXd& h_m,
      const Eigen::ArrayXd& c_m, const Eigen::ArrayXd& up,
      const Eigen::ArrayXd& um) const noexcept {
    using Eigen::MatrixXd, std::move;

    MatrixXd R1(u_m.size(), 3);
    R1.col(0).array() = 1.0;
    R1.col(1) = u_m.matrix();
    R1.col(2) = (0.5 * u_m.square()).matrix();

    MatrixXd R2(u_m.size(), 3);
    R2.col(0).array() = 1.0;
    R2.col(1) = up.matrix();
    R2.col(2) = (h_m + c_m * u_m).matrix();

    MatrixXd R3(u_m.size(), 3);
    R3.col(0).array() = 1.0;
    R3.col(1) = um.matrix();
    R3.col(2) = (h_m - c_m * u_m).matrix();

    return std::make_tuple(move(R1), move(R2), move(R3));
  }

  double gamma_;  ///> Specific heat ratio
};

}  // namespace cfd

#endif  // CFD_RIEMANN_SOLVERS_HPP