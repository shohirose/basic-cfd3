#ifndef CFD_EULER_EQUATION_SIMULATOR_1D_HPP
#define CFD_EULER_EQUATION_SIMULATOR_1D_HPP

#include <iostream>
#include <type_traits>

#include "cfd/boundary_conditions.hpp"
#include "cfd/functions.hpp"
#include "cfd/problem_parameters.hpp"
#include "cfd/time_integration_schemes.hpp"

namespace cfd {

class NoRiemannSolver;
class LaxWendroffSpacialReconstructor;

template <typename SpacialReconstructor, typename RiemannSolver>
class EulerEquationSimulator1d {
 public:
  static_assert(
      (std::is_same_v<SpacialReconstructor, LaxWendroffSpacialReconstructor> &&
       std::is_same_v<RiemannSolver, NoRiemannSolver>) ||
          !std::is_same_v<SpacialReconstructor,
                          LaxWendroffSpacialReconstructor> &&
              !std::is_same_v<RiemannSolver, NoRiemannSolver>,
      "NoRiemannSolver must be selected only with "
      "LaxWendroffSpacialReconstructor.");

  EulerEquationSimulator1d(const ProblemParameters& params,
                           const SpacialReconstructor& reconstructor,
                           const RiemannSolver& solver)
      : dx_{params.dx},
        gamma_{params.specific_heat_ratio},
        tend_{params.tend},
        cfl_number_{params.cfl_number},
        n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells},
        boundary_{params},
        reconstructor_{reconstructor},
        solver_{solver},
        integrator_{params} {}

  PrimitiveVariables run(const PrimitiveVariables& pvars) const noexcept {
    assert(pvars.velocity.size() == this->total_cells());
    assert(pvars.density.size() == this->total_cells());
    assert(pvars.pressure.size() == this->total_cells());
    using Eigen::ArrayXd;

    auto vars = to_conservative_vars(pvars, gamma_);
    boundary_.apply(vars);

    double t = 0.0;
    int tsteps = 0;

    while (t < tend_) {
      auto dt = this->calc_timestep_length(vars);
      if (t + dt > tend_) {
        dt = tend_ - t;
      }
      tsteps += 1;
      t += dt;

      if constexpr (is_lax_wendroff()) {
        const auto f = reconstructor_.calc_flux(vars, dt);
        integrator_.update(vars, f, dt);
        boundary_.apply(vars);
      } else {
        const auto left = reconstructor_.calc_left(vars);
        const auto right = reconstructor_.calc_right(vars);
        const auto f = solver_.calc_flux(left, right);
        integrator_.update(vars, f, dt);
        boundary_.apply(vars);
      }
    }

    return to_primitive_vars(vars, gamma_);
  }

 private:
  static constexpr bool is_lax_wendroff() {
    return std::is_same_v<SpacialReconstructor,
                          LaxWendroffSpacialReconstructor>;
  }

  int total_cells() const noexcept {
    return 2 * n_boundary_cells_ + n_domain_cells_;
  }

  double calc_timestep_length(
      const ConservativeVariables& vars) const noexcept {
    using Eigen::ArrayXd;
    const ArrayXd u = calc_velocity(vars.momentum, vars.density);
    const ArrayXd p =
        calc_pressure(vars.momentum, vars.density, vars.total_energy, gamma_);
    const ArrayXd c = calc_sonic_velocity(p, vars.density, gamma_);

    const auto up = (u + c).abs().maxCoeff();
    const auto um = (u - c).abs().maxCoeff();
    return (cfl_number_ * dx_) / std::max({up, um, 0.1});
  }

  double dx_;
  double gamma_;
  double tend_;
  double cfl_number_;
  int n_boundary_cells_;
  int n_domain_cells_;
  NoFlowBoundary boundary_;
  SpacialReconstructor reconstructor_;
  RiemannSolver solver_;
  ExplicitEulerScheme integrator_;
};

template <typename SpacialReconstructor, typename RiemannSolver>
EulerEquationSimulator1d<SpacialReconstructor, RiemannSolver> make_simulator(
    const ProblemParameters& params, const SpacialReconstructor& reconstructor,
    const RiemannSolver& solver) {
  return {params, reconstructor, solver};
}

}  // namespace cfd

#endif  // CFD_EULER_EQUATION_SIMULATOR_1D_HPP