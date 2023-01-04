#include "cfd/boundary_conditions.hpp"

namespace cfd {

void NoFlowBoundary::apply(ConservativeVariables& vars) const noexcept {
  vars.density.head(n_boundary_cells_) = vars.density(n_boundary_cells_);
  vars.density.tail(n_boundary_cells_) =
      vars.density(n_boundary_cells_ + n_domain_cells_ - 1);

  vars.momentum_density.head(n_boundary_cells_) =
      vars.momentum_density(n_boundary_cells_);
  vars.momentum_density.tail(n_boundary_cells_) =
      vars.momentum_density(n_boundary_cells_ + n_domain_cells_ - 1);

  vars.total_energy_density.head(n_boundary_cells_) =
      vars.total_energy_density(n_boundary_cells_);
  vars.total_energy_density.tail(n_boundary_cells_) =
      vars.total_energy_density(n_boundary_cells_ + n_domain_cells_ - 1);
}

}  // namespace cfd