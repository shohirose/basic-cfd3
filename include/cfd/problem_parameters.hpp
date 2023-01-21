#ifndef CFD_PROBLEM_PARAMETERS_HPP
#define CFD_PROBLEM_PARAMETERS_HPP

namespace cfd {

struct ProblemParameters {
  double dx;                   ///> Grid length
  double specific_heat_ratio;  ///> Specific heat ratio
  double tend;                 ///> Simulation time end
  double cfl_number;           ///> CFL number for time step length calculation
  double
      minimum_velocity;  ///> Minimum velocity for time step length calculation
  int n_bounary_cells;   ///> Number of boundary cells
  int n_domain_cells;    ///> Number of domain cells

  int n_total_cells() const noexcept {
    return 2 * n_bounary_cells + n_domain_cells;
  }
};

}  // namespace cfd

#endif  // CFD_PROBLEM_PARAMETERS_HPP
