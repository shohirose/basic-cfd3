#ifndef CFD_TIME_INTEGRATION_SCHEMES_HPP
#define CFD_TIME_INTEGRATION_SCHEMES_HPP

#include <Eigen/Core>

#include "cfd/problem_parameters.hpp"
#include "cfd/variables.hpp"

namespace cfd {

class ExplicitEulerScheme {
 public:
  ExplicitEulerScheme(double dx, double gamma, int n_boundary_cells,
                      int n_domain_cells)
      : dx_{dx},
        gamma_{gamma},
        n_boundary_cells_{n_boundary_cells},
        n_domain_cells_{n_domain_cells} {}

  ExplicitEulerScheme(const ProblemParameters& params)
      : dx_{params.dx},
        gamma_{params.specific_heat_ratio},
        n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  void update(ConservativeVariables& vars, const FluxVectors& f,
              double dt) const noexcept;

 private:
  double dx_;
  double gamma_;
  int n_boundary_cells_;
  int n_domain_cells_;
};

}  // namespace cfd

#endif  // CFD_TIME_INTEGRATION_SCHEMES_HPP