#ifndef CFD_RIEMANN_SOLVERS_HPP
#define CFD_RIEMANN_SOLVERS_HPP

#include <Eigen/Core>
#include <tuple>

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

    const auto [ul, pl, hl] = calc_velocity_pressure_enthalpy(Ul, gamma_);
    const auto [ur, pr, hr] = calc_velocity_pressure_enthalpy(Ur, gamma_);
    const ArrayXd rhol = Ul.col(0).array();
    const ArrayXd rhor = Ur.col(0).array();
    const auto [rho_m, u_m, h_m, c_m] =
        this->calc_average_properties(rhol, rhor, ul, ur, hl, hr);

    const ArrayXd up = u_m + c_m;
    const ArrayXd um = u_m - c_m;
    const auto [ld1, ld2, ld3] = this->calc_lambda(u_m, up, um);
    const auto [dw1, dw2, dw3] =
        this->calc_dw(rhol, rhor, pl, pr, ul, ur, rho_m, c_m, u_m);
    const auto [R1, R2, R3] = this->calc_r(u_m, h_m, c_m, up, um);

    const ArrayXd el = Ul.col(2).array();
    const ArrayXd er = Ur.col(2).array();
    const auto Fl = this->calc_flux(ul, rhol, pl, el);
    const auto Fr = this->calc_flux(ur, rhor, pr, er);

    const MatrixXd dF1 =
        (ld1.cwiseProduct(dw1)).replicate<1, 3>().cwiseProduct(R1);
    const MatrixXd dF2 =
        (ld2.cwiseProduct(dw2)).replicate<1, 3>().cwiseProduct(R2);
    const MatrixXd dF3 =
        (ld3.cwiseProduct(dw3)).replicate<1, 3>().cwiseProduct(R3);

    return 0.5 * ((Fl + Fr) - (dF1 + dF2 + dF3));
  }

 private:
  std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd>
  calc_average_properties(const Eigen::ArrayXd& rhol,
                          const Eigen::ArrayXd& rhor, const Eigen::ArrayXd& ul,
                          const Eigen::ArrayXd& ur, const Eigen::ArrayXd& hl,
                          const Eigen::ArrayXd& hr) const noexcept {
    using Eigen::ArrayXd, std::move;
    const ArrayXd rhol_sqrt = rhol.sqrt();
    const ArrayXd rhor_sqrt = rhor.sqrt();
    const ArrayXd rho = rhol_sqrt * rhor_sqrt;
    const ArrayXd u =
        (ul * rhol_sqrt + ur * rhor_sqrt) / (rhol_sqrt + rhor_sqrt);
    const ArrayXd h =
        (hl * rhol_sqrt + hr * rhor_sqrt) / (rhol_sqrt + rhor_sqrt);
    const ArrayXd c = ((gamma_ - 1) * (h - 0.5 * u.square())).sqrt();
    return std::make_tuple(move(rho), move(u), move(h), move(c));
  }

  std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> calc_lambda(
      const Eigen::ArrayXd& u, const Eigen::ArrayXd& up,
      const Eigen::ArrayXd& um) const noexcept {
    using Eigen::VectorXd, std::move;
    constexpr double eps = 0.15;
    const VectorXd lambda1 = u.abs().matrix().unaryExpr([eps](double x) {
      return (x > 2 * eps) ? x : (x * x / (4 * eps) + eps);
    });
    const VectorXd lambda2 = up.abs().matrix().unaryExpr([eps](double x) {
      return (x > 2 * eps) ? x : (x * x / (4 * eps) + eps);
    });
    const VectorXd lambda3 = um.abs().matrix().unaryExpr([eps](double x) {
      return (x > 2 * eps) ? x : (x * x / (4 * eps) + eps);
    });
    return std::make_tuple(move(lambda1), move(lambda2), move(lambda3));
  }

  Eigen::MatrixXd calc_flux(const Eigen::ArrayXd& u, const Eigen::ArrayXd& rho,
                            const Eigen::ArrayXd& p,
                            const Eigen::ArrayXd& e) const noexcept {
    Eigen::MatrixXd F(u.size(), 3);
    F.col(0) = (rho * u).matrix();
    F.col(1) = (rho * u.square() + p).matrix();
    F.col(2) = ((e + p) * u).matrix();
    return F;
  }

  /**
   * @brief Compute characteristic variables
   */
  std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> calc_dw(
      const Eigen::ArrayXd& rhol, const Eigen::ArrayXd& rhor,
      const Eigen::ArrayXd& pl, const Eigen::ArrayXd& pr,
      const Eigen::ArrayXd& ul, const Eigen::ArrayXd& ur,
      const Eigen::ArrayXd& rho_m, const Eigen::ArrayXd& c_m,
      const Eigen::ArrayXd& u_m) const noexcept {
    using Eigen::VectorXd, Eigen::ArrayXd, std::move;
    const ArrayXd dp = pr - pl;
    const VectorXd dw1 = (rhor - rhol - dp / (c_m.square())).matrix();
    const ArrayXd a = rho_m / (2 * c_m);
    const ArrayXd du = ur - ul;
    const VectorXd dw2 = (a * (du + dp / (rho_m * c_m))).matrix();
    const VectorXd dw3 = (-a * (du - dp / (rho_m * c_m))).matrix();
    return std::make_tuple(move(dw1), move(dw2), move(dw3));
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