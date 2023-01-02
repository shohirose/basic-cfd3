#ifndef CFD_VARIABLES_HPP
#define CFD_VARIABLES_HPP

#include <Eigen/Core>
#include <array>

namespace cfd {

struct ConservativeVariables {
  Eigen::ArrayXd density;
  Eigen::ArrayXd momentum;
  Eigen::ArrayXd total_energy;
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