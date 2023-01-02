#ifndef CFD_BOUNDARY_CONDITIONS_HPP
#define CFD_BOUNDARY_CONDITIONS_HPP

#include <Eigen/Core>

#include "cfd/problem_parameters.hpp"
#include "cfd/variables.hpp"

namespace cfd {

class NoFlowBoundary {
 public:
  NoFlowBoundary(int n_boundary_cells, int n_domain_cells)
      : n_boundary_cells_{n_boundary_cells}, n_domain_cells_{n_domain_cells} {}

  NoFlowBoundary(const ProblemParameters& params)
      : n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  void apply(ConservativeVariables& vars) const noexcept;

 private:
  int n_boundary_cells_;
  int n_domain_cells_;
};

}  // namespace cfd

#endif  // CFD_BOUNDARY_CONDITIONS_HPP