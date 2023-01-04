#include "cfd/variables.hpp"

#include "cfd/functions.hpp"

namespace cfd {

PrimitiveVariables to_primitive_vars(const ConservativeVariables& vars,
                                     double gamma) noexcept {
  PrimitiveVariables pvars;
  pvars.density = vars.density;
  pvars.pressure = calc_pressure(vars.momentum_density, vars.density,
                                 vars.total_energy_density, gamma);
  pvars.velocity = calc_velocity(vars.momentum_density, vars.density);
  return pvars;
}

ConservativeVariables to_conservative_vars(const PrimitiveVariables& pvars,
                                           double gamma) noexcept {
  ConservativeVariables vars;
  vars.density = pvars.density;
  vars.momentum_density = calc_momentum_density(pvars.density, pvars.velocity);
  vars.total_energy_density = calc_total_energy_density(
      pvars.pressure, pvars.velocity, vars.momentum_density, gamma);
  return vars;
}

}  // namespace cfd
