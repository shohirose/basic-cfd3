#ifndef CFD_RIEMANN_SOLVERS_HPP
#define CFD_RIEMANN_SOLVERS_HPP

#include <Eigen/Core>

#include "cfd/problem_parameters.hpp"
#include "cfd/variables.hpp"

namespace cfd {

class StaggerWarmingRiemannSolver {
 public:
  /**
   * @brief Construct a new Stagger Warming Riemann Solver object
   *
   * @param gamma Specific heat ratio
   */
  StaggerWarmingRiemannSolver(double gamma) : gamma_{gamma} {}

  /**
   * @brief Construct a new Stagger Warming Riemann Solver object
   *
   * @param params Problem parameters
   */
  StaggerWarmingRiemannSolver(const ProblemParameters& params)
      : gamma_{params.specific_heat_ratio} {}

  /**
   * @brief Compute numerical flux at cell interfaces
   *
   * @param left Properties at the LHS of cell interfaces
   * @param right Properties at the RHS of cell interfaces
   * @return FluxVectors Numerical flux
   */
  FluxVectors calc_flux(const ConservativeVariables& left,
                        const ConservativeVariables& right) const noexcept;

 private:
  /**
   * @brief Compute positive numerical flux
   *
   * @param left Properties at the LHS of cell interfaces
   * @return FluxVectors Numerical flux
   */
  FluxVectors calc_positive_flux(
      const ConservativeVariables& left) const noexcept;

  /**
   * @brief Compute negative numerical flux
   *
   * @param right Properties at the RHS of cell interfaces
   * @return FluxVectors Numerical flux
   */
  FluxVectors calc_negative_flux(
      const ConservativeVariables& right) const noexcept;

  double gamma_;  ///> Specific heat ratio
};

}  // namespace cfd

#endif  // CFD_RIEMANN_SOLVERS_HPP