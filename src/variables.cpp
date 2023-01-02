#include "cfd/variables.hpp"

#include "cfd/functions.hpp"

namespace cfd {

PrimitiveVariables to_primitive_vars(const ConservativeVariables& vars,
                                     double gamma) noexcept {
  PrimitiveVariables pvars;
  pvars.density = vars.density;
  pvars.pressure =
      calc_pressure(vars.momentum, vars.density, vars.total_energy, gamma);
  pvars.velocity = calc_velocity(vars.momentum, vars.density);
  return pvars;
}

ConservativeVariables to_conservative_vars(const PrimitiveVariables& pvars,
                                           double gamma) noexcept {
  ConservativeVariables vars;
  vars.density = pvars.density;
  vars.momentum = calc_momentum(pvars.density, pvars.velocity);
  vars.total_energy =
      calc_total_energy(pvars.pressure, pvars.velocity, vars.momentum, gamma);
  return vars;
}

}  // namespace cfd
