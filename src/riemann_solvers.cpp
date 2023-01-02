#include "cfd/riemann_solvers.hpp"

#include <cassert>

#include "cfd/functions.hpp"

namespace cfd {

FluxVectors StegerWarmingRiemannSolver::calc_flux(
    const ConservativeVariables& left,
    const ConservativeVariables& right) const noexcept {
  using Eigen::ArrayXd;
  const auto fp = this->calc_positive_flux(left);
  const auto fm = this->calc_negative_flux(right);
  return {fp[0] + fm[0], fp[1] + fm[1], fp[2] + fm[2]};
}

FluxVectors StegerWarmingRiemannSolver::calc_positive_flux(
    const ConservativeVariables& left) const noexcept {
  using Eigen::ArrayXd;

  const ArrayXd u = calc_velocity(left.momentum, left.density);
  const ArrayXd p =
      calc_pressure(left.momentum, left.density, left.total_energy, gamma_);
  const ArrayXd c = calc_sonic_velocity(p, left.density, gamma_);
  const ArrayXd up = u + c;
  const ArrayXd um = u - c;
  const ArrayXd lambda1 = u.max(0.0);
  const ArrayXd lambda2 = up.max(0.0);
  const ArrayXd lambda3 = um.max(0.0);

  const auto& rho = left.density;
  const ArrayXd x1 = ((gamma_ - 1.0) / gamma_) * rho * lambda1;
  const ArrayXd x2 = rho * lambda2 / (2 * gamma_);
  const ArrayXd x3 = rho * lambda3 / (2 * gamma_);
  const ArrayXd uu = 0.5 * u.square();
  const ArrayXd cc = (1 / (gamma_ - 1)) * c.square();
  const ArrayXd uc = u * c;

  FluxVectors f;
  f[0] = x1 + x2 + x3;
  f[1] = x1 * u + x2 * up + x3 * um;
  f[2] = x1 * uu + x2 * (uu + cc + uc) + x3 * (uu + cc - uc);
  return f;
}

FluxVectors StegerWarmingRiemannSolver::calc_negative_flux(
    const ConservativeVariables& right) const noexcept {
  using Eigen::ArrayXd;

  const ArrayXd u = calc_velocity(right.momentum, right.density);
  const ArrayXd p =
      calc_pressure(right.momentum, right.density, right.total_energy, gamma_);
  const ArrayXd c = calc_sonic_velocity(p, right.density, gamma_);
  const ArrayXd up = u + c;
  const ArrayXd um = u - c;
  const ArrayXd lambda1 = u.min(0.0);
  const ArrayXd lambda2 = up.min(0.0);
  const ArrayXd lambda3 = um.min(0.0);

  const auto& rho = right.density;
  const ArrayXd x1 = ((gamma_ - 1.0) / gamma_) * rho * lambda1;
  const ArrayXd x2 = rho * lambda2 / (2 * gamma_);
  const ArrayXd x3 = rho * lambda3 / (2 * gamma_);
  const ArrayXd uu = 0.5 * u.square();
  const ArrayXd cc = (1 / (gamma_ - 1)) * c.square();
  const ArrayXd uc = u * c;

  FluxVectors f;
  f[0] = x1 + x2 + x3;
  f[1] = x1 * u + x2 * up + x3 * um;
  f[2] = x1 * uu + x2 * (uu + cc + uc) + x3 * (uu + cc - uc);
  return f;
}

}  // namespace cfd
