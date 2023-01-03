#include "cfd/spacial_reconstruction_schemes.hpp"

#include <cassert>

#include "cfd/functions.hpp"

namespace cfd {

ConservativeVariables FirstOrderSpacialReconstructor::calc_left(
    const ConservativeVariables& vars) const noexcept {
  assert(vars.density.size() == n_boundary_cells_ * 2 + n_domain_cells_);
  assert(vars.momentum.size() == n_boundary_cells_ * 2 + n_domain_cells_);
  assert(vars.total_energy.size() == n_boundary_cells_ * 2 + n_domain_cells_);

  ConservativeVariables left;
  const auto rng = Eigen::seqN(n_boundary_cells_ - 1, n_domain_cells_ + 1);
  left.density = vars.density(rng);
  left.momentum = vars.momentum(rng);
  left.total_energy = vars.total_energy(rng);
  return left;
}

ConservativeVariables FirstOrderSpacialReconstructor::calc_right(
    const ConservativeVariables& vars) const noexcept {
  assert(vars.density.size() == n_boundary_cells_ * 2 + n_domain_cells_);
  assert(vars.momentum.size() == n_boundary_cells_ * 2 + n_domain_cells_);
  assert(vars.total_energy.size() == n_boundary_cells_ * 2 + n_domain_cells_);

  ConservativeVariables right;
  const auto rng = Eigen::seqN(n_boundary_cells_, n_domain_cells_ + 1);
  right.density = vars.density(rng);
  right.momentum = vars.momentum(rng);
  right.total_energy = vars.total_energy(rng);
  return right;
}

FluxVectors LaxWendroffSpacialReconstructor::calc_flux(
    const ConservativeVariables& vars, double dt) const noexcept {
  using Eigen::ArrayXd;
  using Eigen::seqN;
  const auto& rho = vars.density;
  const auto& m = vars.momentum;
  const auto& e = vars.total_energy;
  const auto rng1 = seqN(n_boundary_cells_ - 1, n_domain_cells_ + 1);
  const auto rng2 = seqN(n_boundary_cells_, n_domain_cells_ + 1);
  const auto alpha = 0.5 * dt / dx_;

  const ArrayXd rho_s =
      0.5 * (rho(rng1) + rho(rng2)) - alpha * (m(rng2) - m(rng1));

  const ArrayXd p = calc_pressure(m, rho, e, gamma_);
  const ArrayXd u = calc_velocity(m, rho);
  const ArrayXd a1 = m * u + p;
  const ArrayXd m_s = 0.5 * (m(rng1) + m(rng2)) - alpha * (a1(rng2) - a1(rng1));

  const ArrayXd a2 = u * (e + p);
  const ArrayXd e_s = 0.5 * (e(rng1) + e(rng2)) - alpha * (a2(rng2) - a2(rng1));

  FluxVectors f;
  f[0] = m_s;
  const ArrayXd p_s = calc_pressure(m_s, rho_s, e_s, gamma_);
  const ArrayXd u_s = calc_velocity(m_s, rho_s);
  f[1] = m_s * u_s + p_s;
  f[2] = u_s * (e_s + p_s);
  return f;
}

}  // namespace cfd
