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

  const ArrayXd u = calc_velocity(left.momentum_density, left.density);
  const ArrayXd p = calc_pressure(left.momentum_density, left.density,
                                  left.total_energy_density, gamma_);
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

  const ArrayXd u = calc_velocity(right.momentum_density, right.density);
  const ArrayXd p = calc_pressure(right.momentum_density, right.density,
                                  right.total_energy_density, gamma_);
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

FluxVectors RoeRiemannSolver::calc_flux(
    const ConservativeVariables& left,
    const ConservativeVariables& right) const noexcept {
  using Eigen::ArrayXd;
  const auto& rhol = left.density;
  const auto& rhor = right.density;
  const ArrayXd rhol_sqrt = rhol.sqrt();
  const ArrayXd rhor_sqrt = rhor.sqrt();
  const ArrayXd rho_m = rhol_sqrt * rhor_sqrt;

  const ArrayXd ul = calc_velocity(left.momentum_density, left.density);
  const ArrayXd ur = calc_velocity(right.momentum_density, right.density);
  const ArrayXd u_m =
      (ul * rhol_sqrt + ur * rhor_sqrt) / (rhol_sqrt + rhor_sqrt);

  const ArrayXd pl = calc_pressure(left.momentum_density, left.density,
                                   left.total_energy_density, gamma_);
  const ArrayXd pr = calc_pressure(right.momentum_density, right.density,
                                   right.total_energy_density, gamma_);

  const auto& el = left.total_energy_density;
  const auto& er = right.total_energy_density;
  const ArrayXd hl = calc_total_enthalpy(pl, rhol, el);
  const ArrayXd hr = calc_total_enthalpy(pr, rhor, er);
  const ArrayXd h_m =
      (hl * rhol_sqrt + hr * rhor_sqrt) / (rhol_sqrt + rhor_sqrt);
  const ArrayXd c_m = ((gamma_ - 1) * (h_m - 0.5 * u_m.square())).sqrt();

  const ArrayXd up = u_m + c_m;
  const ArrayXd um = u_m - c_m;
  constexpr double eps = 0.15;
  const ArrayXd lambda1 = u_m.abs().unaryExpr(
      [eps](double x) { return x > 2 * eps ? x : (x * x / (4 * eps) + eps); });
  const ArrayXd lambda2 = up.abs().unaryExpr(
      [eps](double x) { return x > 2 * eps ? x : (x * x / (4 * eps) + eps); });
  const ArrayXd lambda3 = um.abs().unaryExpr(
      [eps](double x) { return x > 2 * eps ? x : (x * x / (4 * eps) + eps); });

  const ArrayXd dw1 = rhor - rhol - (pr - pl) / (c_m.square());
  const ArrayXd dw2 = ur - ul + (pr - pl) / (rho_m * c_m);
  const ArrayXd dw3 = ur - ul - (pr - pl) / (rho_m * c_m);
  const ArrayXd a1 = rho_m / (2 * c_m);

  FluxVectors f;
  f[0] = 0.5 * ((rhol * ul + rhor * ur) -
                (lambda1 * dw1 + a1 * (lambda2 * dw2 - lambda3 * dw3)));
  f[1] =
      0.5 *
      (rhol * ul.square() + pl + rhor * ur.square() + pr -
       (lambda1 * dw1 * u_m + a1 * (lambda2 * dw2 * up - lambda3 * dw3 * um)));
  f[2] = 0.5 * (el * ul + pl * ul + er * ur + pr * ur -
                (0.5 * lambda1 * dw1 * u_m.square() +
                 a1 * (lambda2 * dw2 * (h_m + c_m * u_m) -
                       lambda3 * dw3 * (h_m - c_m * u_m))));
  return f;
}

}  // namespace cfd
