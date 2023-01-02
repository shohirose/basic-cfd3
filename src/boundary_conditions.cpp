#include "cfd/boundary_conditions.hpp"

namespace cfd {

void NoFlowBoundary::apply(ConservativeVariables& vars) const noexcept {
  vars.density.head(n_boundary_cells_) = vars.density(n_boundary_cells_);
  vars.density.tail(n_boundary_cells_) =
      vars.density(n_boundary_cells_ + n_domain_cells_ - 1);

  vars.momentum.head(n_boundary_cells_) = vars.momentum(n_boundary_cells_);
  vars.momentum.tail(n_boundary_cells_) =
      vars.momentum(n_boundary_cells_ + n_domain_cells_ - 1);

  vars.total_energy.head(n_boundary_cells_) =
      vars.total_energy(n_boundary_cells_);
  vars.total_energy.tail(n_boundary_cells_) =
      vars.total_energy(n_boundary_cells_ + n_domain_cells_ - 1);
}

}  // namespace cfd