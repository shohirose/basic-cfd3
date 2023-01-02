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

}  // namespace cfd
