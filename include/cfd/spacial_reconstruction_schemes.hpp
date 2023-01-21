#ifndef CFD_SPACIAL_RECONSTRUCTION_SCHEMES_HPP
#define CFD_SPACIAL_RECONSTRUCTION_SCHEMES_HPP

#include <Eigen/Core>
#include <cmath>

#include "cfd/problem_parameters.hpp"

namespace cfd {

/**
 * @brief First-order spacial reconstruction scheme.
 *
 */
class FirstOrderSpacialReconstructor {
 public:
  FirstOrderSpacialReconstructor(const ProblemParameters& params)
      : n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  /**
   * @brief Compute numerical flux at LHS of cell interfaces
   *
   * @param U Conservation variables
   * @return Numerical flux at LHS of cells interfaces
   */
  template <typename Derived>
  Eigen::MatrixXd calc_left(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    assert(U.rows() == n_boundary_cells_ * 2 + n_domain_cells_);
    assert(U.cols() == 3);

    using Eigen::all, Eigen::seqN;
    return U(seqN(n_boundary_cells_ - 1, n_domain_cells_ + 1), all);
  }

  /**
   * @brief Compute numerical flux at RHS of cell interfaces
   *
   * @param U Conservation variables
   * @return Numerical flux at RHS of cells interfaces
   */
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

/**
 * @brief Second-order spacial reconstruction scheme using Total Variation
 * Diminishing (TVD).
 *
 * @tparam SlopeLimiter Slope limiter function
 */
template <typename SlopeLimiter>
class TvdSpacialReconstructor {
 public:
  TvdSpacialReconstructor(const ProblemParameters& params)
      : n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  /**
   * @brief Compute numerical flux at LHS of cell interfaces
   *
   * @param U Conservation variables
   * @return Numerical flux at LHS of cells interfaces
   */
  template <typename Derived>
  Eigen::MatrixXd calc_left(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    using Eigen::seqN, Eigen::Map, Eigen::MatrixXd, Eigen::ArrayXd;
    Map<const ArrayXd> rho(&U(0, 0), U.rows());
    Map<const ArrayXd> rhou(&U(0, 1), U.rows());
    Map<const ArrayXd> rhoE(&U(0, 2), U.rows());

    const ArrayXd r1 = this->calc_slope_ratio_left(rho);
    const ArrayXd r2 = this->calc_slope_ratio_left(rhou);
    const ArrayXd r3 = this->calc_slope_ratio_left(rhoE);

    const ArrayXd phi1 = SlopeLimiter::eval(r1);
    const ArrayXd phi2 = SlopeLimiter::eval(r2);
    const ArrayXd phi3 = SlopeLimiter::eval(r3);

    const auto i = n_boundary_cells_ - 1;
    const auto n = n_domain_cells_ + 1;
    const auto rng1 = seqN(i, n);
    const auto rng2 = seqN(i + 1, n);
    MatrixXd Ul(n, 3);
    Ul.col(0).array() = rho(rng1) + 0.5 * phi1 * (rho(rng2) - rho(rng1));
    Ul.col(1).array() = rhou(rng1) + 0.5 * phi2 * (rhou(rng2) - rhou(rng1));
    Ul.col(2).array() = rhoE(rng1) + 0.5 * phi3 * (rhoE(rng2) - rhoE(rng1));
    return Ul;
  }

  /**
   * @brief Compute numerical flux at RHS of cell interfaces
   *
   * @param U Conservation variables
   * @return Numerical flux at RHS of cells interfaces
   */
  template <typename Derived>
  Eigen::MatrixXd calc_right(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    using Eigen::seqN, Eigen::Map, Eigen::MatrixXd, Eigen::ArrayXd;
    Map<const ArrayXd> rho(&U(0, 0), U.rows());
    Map<const ArrayXd> rhou(&U(0, 1), U.rows());
    Map<const ArrayXd> rhoE(&U(0, 2), U.rows());

    const ArrayXd r1 = this->calc_slope_ratio_right(rho);
    const ArrayXd r2 = this->calc_slope_ratio_right(rhou);
    const ArrayXd r3 = this->calc_slope_ratio_right(rhoE);

    const ArrayXd phi1 = SlopeLimiter::eval(r1);
    const ArrayXd phi2 = SlopeLimiter::eval(r2);
    const ArrayXd phi3 = SlopeLimiter::eval(r3);

    const auto i = n_boundary_cells_ - 1;
    const auto n = n_domain_cells_ + 1;
    const auto rng1 = seqN(i, n);
    const auto rng2 = seqN(i + 1, n);
    MatrixXd Ur(n, 3);
    Ur.col(0).array() = rho(rng2) - 0.5 * phi1 * (rho(rng2) - rho(rng1));
    Ur.col(1).array() = rhou(rng2) - 0.5 * phi2 * (rhou(rng2) - rhou(rng1));
    Ur.col(2).array() = rhoE(rng2) - 0.5 * phi3 * (rhoE(rng2) - rhoE(rng1));
    return Ur;
  }

 private:
  template <typename Derived>
  Eigen::ArrayXd calc_slope_ratio_left(
      const Eigen::ArrayBase<Derived>& x) const noexcept {
    const auto i = n_boundary_cells_ - 1;
    const auto n = n_domain_cells_ + 1;
    const Eigen::ArrayXd dx = x(seqN(i + 1, n)) - x(seqN(i, n));
    return (x(seqN(i, n)) - x(seqN(i - 1, n))) /
           dx.unaryExpr(
               [eps = 1e-5](double x) { return x + std::copysign(eps, x); });
  }

  template <typename Derived>
  Eigen::ArrayXd calc_slope_ratio_right(
      const Eigen::ArrayBase<Derived>& x) const noexcept {
    const auto i = n_boundary_cells_ - 1;
    const auto n = n_domain_cells_ + 1;
    const Eigen::ArrayXd dx = x(seqN(i + 1, n)) - x(seqN(i, n));
    return (x(seqN(i + 2, n)) - x(seqN(i + 1, n))) /
           dx.unaryExpr(
               [eps = 1e-5](double x) { return x + std::copysign(eps, x); });
  }

  int n_boundary_cells_;
  int n_domain_cells_;
};

}  // namespace cfd

#endif  // CFD_SPACIAL_RECONSTRUCTION_SCHEMES_HPP