#include <Eigen/Core>

#include "cfd/problem_parameters.hpp"

inline Eigen::MatrixXd make_initial_condition(
    const cfd::ProblemParameters& params) {
  const auto n = params.n_total_cells();
  Eigen::MatrixXd V(n, 3);
  const auto left = Eigen::seqN(0, n / 2);
  const auto right = Eigen::seqN(n / 2, n - n / 2);
  // density
  V(left, 0).array() = 1.0;
  V(right, 0).array() = 0.125;
  // velocity
  V(left, 1).array() = 0.0;
  V(right, 1).array() = 0.0;
  // pressure
  V(left, 2).array() = 1.0;
  V(right, 2).array() = 0.1;
  return V;
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