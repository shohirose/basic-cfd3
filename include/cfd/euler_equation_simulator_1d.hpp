#ifndef CFD_EULER_EQUATION_SIMULATOR_1D_HPP
#define CFD_EULER_EQUATION_SIMULATOR_1D_HPP

#include "cfd/boundary_conditions.hpp"
#include "cfd/problem_parameters.hpp"
#include "cfd/time_integration_schemes.hpp"

namespace cfd {

/**
 * @brief 1-D Euler equation simulator.
 *
 * @tparam SpacialReconstructor Spacial reconstruction scheme
 * @tparam RiemannSolver Riemann solver
 */
template <typename FluxSolver>
class EulerEquationSimulator1d {
 public:
  EulerEquationSimulator1d(const ProblemParameters& params,
                           const FluxSolver& solver)
      : dx_{params.dx},
        gamma_{params.specific_heat_ratio},
        tend_{params.tend},
        cfl_number_{params.cfl_number},
        n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells},
        boundary_{params},
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

    MatrixXd U = this->to_conservation_vars(V);
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

      integrator_.update(U, dt, solver_, boundary_);
    }

    return this->to_primitive_vars(U);
  }

 private:
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
    using Eigen::ArrayXd, Eigen::Map, Eigen::MatrixXd;
    constexpr double minimum_velocity = 0.1;
    Map<const ArrayXd> rho(&U(0, 0), U.rows());
    Map<const ArrayXd> rhou(&U(0, 1), U.rows());
    Map<const ArrayXd> rhoE(&U(0, 2), U.rows());
    const ArrayXd u = rhou / rho;
    const ArrayXd p = (gamma_ - 1.0) * (rhoE - 0.5 * rhou.square() / rho);
    const ArrayXd c = (gamma_ * p / rho).sqrt();
    const auto up = (u + c).abs().maxCoeff();
    const auto um = (u - c).abs().maxCoeff();
    return (cfl_number_ * dx_) / std::max({up, um, minimum_velocity});
  }

  /**
   * @brief Convert conservation variables vector to primitive variables vector.
   *
   * @tparam Derived
   * @param U Conservation variables vector
   * @return Eigen::MatrixXd Primitive variables vector
   *
   */
  template <typename Derived>
  Eigen::MatrixXd to_primitive_vars(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    using Eigen::Map, Eigen::MatrixXd, Eigen::ArrayXd;
    MatrixXd V(U.rows(), 3);
    Map<const ArrayXd> rho(&U(0, 0), U.rows());
    Map<const ArrayXd> rhou(&U(0, 1), U.rows());
    Map<const ArrayXd> rhoE(&U(0, 2), U.rows());

    // Density
    V.col(0) = rho.matrix();
    // Velocity
    V.col(1) = (rhou / rho).matrix();
    // Pressure
    V.col(2) = (gamma_ - 1.0) * (rhoE - 0.5 * rhou.square() / rho).matrix();

    return V;
  }

  /**
   * @brief Convert primitive varialbes vector to conservation variables vector.
   *
   * @tparam Derived
   * @param V Primitive variables vector
   * @return Eigen::MatrixXd Conservation variables vector
   *
   */
  template <typename Derived>
  Eigen::MatrixXd to_conservation_vars(
      const Eigen::MatrixBase<Derived>& V) const noexcept {
    using Eigen::Map, Eigen::MatrixXd, Eigen::ArrayXd;
    MatrixXd U(V.rows(), 3);
    Map<const ArrayXd> rho(&V(0, 0), V.rows());
    Map<const ArrayXd> u(&V(0, 1), V.rows());
    Map<const ArrayXd> p(&V(0, 2), V.rows());

    // Density
    U.col(0) = rho;
    // Momentum density
    U.col(1) = rho * u;
    // Total energy density
    U.col(2) = 0.5 * rho * u + p / (gamma_ - 1.0);

    return U;
  }

  double dx_;                       ///> Grid length
  double gamma_;                    ///> Specific heat ratio
  double tend_;                     ///> End time of a simulation
  double cfl_number_;               ///> CFL number
  int n_boundary_cells_;            ///> Number of boundary cells
  int n_domain_cells_;              ///> Number of domain cells
  NoFlowBoundary boundary_;         ///> Boundary condition
  FluxSolver solver_;               ///> Flux solver
  ExplicitEulerScheme integrator_;  ///> Time integration scheme
};

template <typename FluxSolver>
EulerEquationSimulator1d<FluxSolver> make_simulator(
    const ProblemParameters& params, const FluxSolver& solver) {
  return {params, solver};
}

}  // namespace cfd

#endif  // CFD_EULER_EQUATION_SIMULATOR_1D_HPP