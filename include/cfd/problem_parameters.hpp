#ifndef CFD_PROBLEM_PARAMETERS_HPP
#define CFD_PROBLEM_PARAMETERS_HPP

namespace cfd {

struct ProblemParameters {
  double dx;
  double specific_heat_ratio;
  double tend;
  double cfl_number;
  double minimum_velocity;
  int n_bounary_cells;
  int n_domain_cells;

  int n_total_cells() const noexcept {
    return 2 * n_bounary_cells + n_domain_cells;
  }
};

}  // namespace cfd

#endif  // CFD_PROBLEM_PARAMETERS_HPP
