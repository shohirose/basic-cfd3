#include "cfd/time_integration_schemes.hpp"

namespace cfd {

void ExplicitEulerScheme::update(ConservativeVariables& vars,
                                 const FluxVectors& f,
                                 double dt) const noexcept {
  const auto domain = Eigen::seqN(n_boundary_cells_, n_domain_cells_);
  vars.density(domain) -=
      (dt / dx_) * (f[0].tail(n_domain_cells_) - f[0].head(n_domain_cells_));
  vars.momentum(domain) -=
      (dt / dx_) * (f[1].tail(n_domain_cells_) - f[1].head(n_domain_cells_));
  vars.total_energy(domain) -=
      (dt / dx_) * (f[2].tail(n_domain_cells_) - f[2].head(n_domain_cells_));
}

}  // namespace cfd
