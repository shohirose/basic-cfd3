#ifndef CFD_EULER_EQUATION_SIMULATOR_1D_HPP
#define CFD_EULER_EQUATION_SIMULATOR_1D_HPP

#include <iostream>
#include <type_traits>

#include "cfd/boundary_conditions.hpp"
#include "cfd/functions.hpp"
#include "cfd/problem_parameters.hpp"
#include "cfd/time_integration_schemes.hpp"

namespace cfd {

// Forward declaration
class NoRiemannSolver;
class LaxWendroffSpacialReconstructor;

/**
 * @brief 1-D Euler equation simulator.
 *
 * @tparam SpacialReconstructor Spacial reconstruction scheme
 * @tparam RiemannSolver Riemann solver
 */
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

  /**
   * @brief Run a simulation case.
   *
   * @tparam Derived
   * @param V Primitive variables vector at the initial condition.
   * @return Eigen::MatrixXd Primitive Variables vector at the end of time
   * steps.
   *
   * Primitive variables vector is defined by
   * @f[
   * \mathbf{V} =
   * \begin{bmatrix}
   * \rho \\ u \\ p
   * \end{bmatrix}
   * where @f$ \rho @f$ is density, @f$ u @f$ is velocity, and @f$ p @f$ is
   * pressure.
   * @f]
   */
  template <typename Derived>
  Eigen::MatrixXd run(const Eigen::MatrixBase<Derived>& V) const noexcept {
    assert(V.rows() == this->total_cells());
    assert(V.cols() == 3);
    using Eigen::MatrixXd;

    MatrixXd U = to_conservation_vars(V, gamma_);
    boundary_.apply(U);

    double t = 0.0;
    int tsteps = 0;

    while (t < tend_) {
      auto dt = this->calc_timestep_length(U);
      if (t + dt > tend_) {
        dt = tend_ - t;
      }
      tsteps += 1;
      t += dt;

      if constexpr (is_lax_wendroff()) {
        const auto F = reconstructor_.calc_flux(U, dt);
        integrator_.update(U, F, dt);
        boundary_.apply(U);
      } else {
        const auto Ul = reconstructor_.calc_left(U);
        const auto Ur = reconstructor_.calc_right(U);
        const auto F = solver_.calc_flux(Ul, Ur);
        integrator_.update(U, F, dt);
        boundary_.apply(U);
      }
    }

    return to_primitive_vars(U, gamma_);
  }

 private:
  static constexpr bool is_lax_wendroff() {
    return std::is_same_v<SpacialReconstructor,
                          LaxWendroffSpacialReconstructor>;
  }

  int total_cells() const noexcept {
    return 2 * n_boundary_cells_ + n_domain_cells_;
  }

  /**
   * @brief Compute time step length based on CFL number.
   *
   * @tparam Derived
   * @param U Conservation variables vector
   * @return double Time step length
   */
  template <typename Derived>
  double calc_timestep_length(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    using Eigen::ArrayXd;
    const auto [u, p, c] = calc_velocity_pressure_sonic_velocity(U, gamma_);
    const auto up = (u + c).abs().maxCoeff();
    const auto um = (u - c).abs().maxCoeff();
    return (cfl_number_ * dx_) / std::max({up, um, 0.1});
  }

  double dx_;                           ///> Grid length
  double gamma_;                        ///> Specific heat ratio
  double tend_;                         ///> End time of a simulation
  double cfl_number_;                   ///> CFL number
  int n_boundary_cells_;                ///> Number of boundary cells
  int n_domain_cells_;                  ///> Number of domain cells
  NoFlowBoundary boundary_;             ///> Boundary condition
  SpacialReconstructor reconstructor_;  ///> Spacial reconstruction scheme
  RiemannSolver solver_;                ///> Riemann solver
  ExplicitEulerScheme integrator_;      ///> Time integration scheme
};

template <typename SpacialReconstructor, typename RiemannSolver>
EulerEquationSimulator1d<SpacialReconstructor, RiemannSolver> make_simulator(
    const ProblemParameters& params, const SpacialReconstructor& reconstructor,
    const RiemannSolver& solver) {
  return {params, reconstructor, solver};
}

}  // namespace cfd

#endif  // CFD_EULER_EQUATION_SIMULATOR_1D_HPP