#ifndef CFD_VARIABLES_HPP
#define CFD_VARIABLES_HPP

#include <Eigen/Core>
#include <array>

namespace cfd {

struct ConservativeVariables {
  Eigen::ArrayXd density;

  /// @brief Momentum density @f$ \rho u @f$
  Eigen::ArrayXd momentum_density;

  /**
   * @brief Total energy density @f$ E^t @f$
   *
   * Total energy density is defined by
   * @f[
   * E^t = \rho \left( e + \frac{1}{2} u^2 \right),
   * @f]
   * where @f$ e @f$ is internal energy, @f$ u @f$ is velocity, and @f$ \rho @f$
   * is density.
   */
  Eigen::ArrayXd total_energy_density;
};

struct PrimitiveVariables {
  Eigen::ArrayXd density;
  Eigen::ArrayXd velocity;
  Eigen::ArrayXd pressure;
};

using FluxVectors = std::array<Eigen::ArrayXd, 3>;

PrimitiveVariables to_primitive_vars(const ConservativeVariables& vars,
                                     double gamma) noexcept;

ConservativeVariables to_conservative_vars(const PrimitiveVariables& pvars,
                                           double gamma) noexcept;

}  // namespace cfd

#endif  // CFD_VARIABLES_HPP