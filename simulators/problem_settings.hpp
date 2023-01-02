#include "cfd/cfd.hpp"

using Eigen::ArrayXd;

inline cfd::PrimitiveVariables make_initial_condition(
    const cfd::ProblemParameters& params) {
  cfd::PrimitiveVariables pvars;
  const auto n = params.n_total_cells();
  pvars.density = ArrayXd::Zero(n);
  pvars.pressure = ArrayXd::Zero(n);
  pvars.velocity = ArrayXd::Zero(n);
  const auto left = Eigen::seqN(0, n / 2);
  const auto right = Eigen::seqN(n / 2, n - n / 2);
  pvars.density(left) = 1.0;
  pvars.pressure(left) = 1.0;
  pvars.density(right) = 0.125;
  pvars.pressure(right) = 0.1;
  return pvars;
}

inline cfd::ProblemParameters make_parameters() {
  return {
      0.02,  // dx
      1.4,   // Specific heat ratio
      0.48,  // Time end
      0.4,   // CFL number
      2,     // Number of boundary cells
      100,   // Number of domain cells
  };
}