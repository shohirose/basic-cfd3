#ifndef CFD_RIEMANN_SOLVERS_HPP
#define CFD_RIEMANN_SOLVERS_HPP

#include <Eigen/Core>

#include "cfd/problem_parameters.hpp"

namespace cfd {

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
    const MatrixXd Fp = this->calc_positive_flux(Ul);
    const MatrixXd Fm = this->calc_negative_flux(Ur);
    return Fp + Fm;
  }

 private:
  /**
   * @brief Compute positive numerical flux
   *
   * @param Ul Conservation variables vector at the LHS of cell interfaces
   * @return Eigen::MatrixXd Numerical flux vector
   */
  template <typename Derived>
  Eigen::MatrixXd calc_positive_flux(
      const Eigen::MatrixBase<Derived>& Ul) const noexcept {
    using Eigen::ArrayXd, Eigen::MatrixXd;

    const auto [u, p, c] = calc_velocity_pressure_sonic_velocity(Ul, gamma_);
    const ArrayXd up = u + c;
    const ArrayXd um = u - c;
    const ArrayXd lambda1 = u.max(0.0);
    const ArrayXd lambda2 = up.max(0.0);
    const ArrayXd lambda3 = um.max(0.0);

    const ArrayXd x1 = ((gamma_ - 1.0) / gamma_) * Ul.col(0).array() * lambda1;
    const ArrayXd x2 = Ul.col(0).array() * lambda2 / (2 * gamma_);
    const ArrayXd x3 = Ul.col(0).array() * lambda3 / (2 * gamma_);
    const ArrayXd uu = 0.5 * u.square();
    const ArrayXd cc = (1 / (gamma_ - 1)) * c.square();
    const ArrayXd uc = u * c;

    MatrixXd F(Ul.rows(), 3);
    F.col(0) = (x1 + x2 + x3).matrix();
    F.col(1) = (x1 * u + x2 * up + x3 * um).matrix();
    F.col(2) = (x1 * uu + x2 * (uu + cc + uc) + x3 * (uu + cc - uc)).matrix();
    return F;
  }

  /**
   * @brief Compute negative numerical flux
   *
   * @param Ur Conservation variables vector at the RHS of cell interfaces
   * @return Eigen::MatrixXd Numerical flux vector
   */
  template <typename Derived>
  Eigen::MatrixXd calc_negative_flux(
      const Eigen::MatrixBase<Derived>& Ur) const noexcept {
    using Eigen::ArrayXd, Eigen::MatrixXd;

    const auto [u, p, c] = calc_velocity_pressure_sonic_velocity(Ur, gamma_);
    const ArrayXd up = u + c;
    const ArrayXd um = u - c;
    const ArrayXd lambda1 = u.min(0.0);
    const ArrayXd lambda2 = up.min(0.0);
    const ArrayXd lambda3 = um.min(0.0);

    const ArrayXd x1 = ((gamma_ - 1.0) / gamma_) * Ur.col(0).array() * lambda1;
    const ArrayXd x2 = Ur.col(0).array() * lambda2 / (2 * gamma_);
    const ArrayXd x3 = Ur.col(0).array() * lambda3 / (2 * gamma_);
    const ArrayXd uu = 0.5 * u.square();
    const ArrayXd cc = (1 / (gamma_ - 1)) * c.square();
    const ArrayXd uc = u * c;

    MatrixXd F(Ur.rows(), 3);
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
    using Eigen::ArrayXd, Eigen::MatrixXd;

    const ArrayXd rhol_sqrt = Ul.col(0).array().sqrt();
    const ArrayXd rhor_sqrt = Ur.col(0).array().sqrt();
    const ArrayXd rho_m = rhol_sqrt * rhor_sqrt;

    const auto [ul, pl, hl] = calc_velocity_pressure_enthalpy(Ul, gamma_);
    const auto [ur, pr, hr] = calc_velocity_pressure_enthalpy(Ur, gamma_);

    const ArrayXd u_m =
        (ul * rhol_sqrt + ur * rhor_sqrt) / (rhol_sqrt + rhor_sqrt);
    const ArrayXd h_m =
        (hl * rhol_sqrt + hr * rhor_sqrt) / (rhol_sqrt + rhor_sqrt);
    const ArrayXd c_m = ((gamma_ - 1) * (h_m - 0.5 * u_m.square())).sqrt();

    const ArrayXd up = u_m + c_m;
    const ArrayXd um = u_m - c_m;
    constexpr double eps = 0.15;
    const ArrayXd lambda1 = u_m.abs().unaryExpr([eps](double x) {
      return x > 2 * eps ? x : (x * x / (4 * eps) + eps);
    });
    const ArrayXd lambda2 = up.abs().unaryExpr([eps](double x) {
      return x > 2 * eps ? x : (x * x / (4 * eps) + eps);
    });
    const ArrayXd lambda3 = um.abs().unaryExpr([eps](double x) {
      return x > 2 * eps ? x : (x * x / (4 * eps) + eps);
    });

    const ArrayXd dw1 =
        Ur.col(0).array() - Ul.col(0).array() - (pr - pl) / (c_m.square());
    const ArrayXd dw2 = ur - ul + (pr - pl) / (rho_m * c_m);
    const ArrayXd dw3 = ur - ul - (pr - pl) / (rho_m * c_m);
    const ArrayXd a1 = rho_m / (2 * c_m);

    MatrixXd F(Ul.rows(), 3);
    F.col(0) = 0.5 * ((Ul.col(0).array() * ul + Ur.col(0).array() * ur) -
                      (lambda1 * dw1 + a1 * (lambda2 * dw2 - lambda3 * dw3)))
                         .matrix();
    F.col(1) =
        0.5 *
        (Ul.col(0).array() * ul.square() + pl +
         Ur.col(0).array() * ur.square() + pr -
         (lambda1 * dw1 * u_m + a1 * (lambda2 * dw2 * up - lambda3 * dw3 * um)))
            .matrix();
    F.col(2) =
        0.5 * ((Ul.col(2).array() + pl) * ul + (Ur.col(2).array() + pr) * ur -
               (0.5 * lambda1 * dw1 * u_m.square() +
                a1 * (lambda2 * dw2 * (h_m + c_m * u_m) -
                      lambda3 * dw3 * (h_m - c_m * u_m))));
    return F;
  }

 private:
  double gamma_;  ///> Specific heat ratio
};

}  // namespace cfd

#endif  // CFD_RIEMANN_SOLVERS_HPP