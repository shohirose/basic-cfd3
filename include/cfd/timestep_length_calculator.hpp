#ifndef CFD_TIMESTEP_LENGTH_CALCULATOR_HPP
#define CFD_TIMESTEP_LENGTH_CALCULATOR_HPP

#include <Eigen/Core>
#include <algorithm>

#include "cfd/functions.hpp"
#include "cfd/problem_parameters.hpp"

namespace cfd {

/**
 * @brief Time step length calculator
 *
 */
class TimestepLengthCalculator {
 public:
  TimestepLengthCalculator(const ProblemParameters& params)
      : cfl_number_{params.cfl_number},
        dx_{params.dx},
        minimum_velocity_{params.minimum_velocity},
        specific_heat_ratio_{params.specific_heat_ratio} {}
  /**
   * @brief Compute time step length based on CFL number.
   *
   * @tparam Derived
   * @param U Conservation variables vector
   * @return double Time step length
   */
  template <typename Derived>
  double compute(const Eigen::MatrixBase<Derived>& U) const noexcept {
    using Eigen::ArrayXd;
    const ArrayXd u = calc_velocity(U.col(0).array(), U.col(1).array());
    const ArrayXd p = calc_pressure(U.col(0).array(), U.col(1).array(),
                                    U.col(2).array(), specific_heat_ratio_);
    const ArrayXd c =
        calc_sonic_velocity(U.col(0).array(), p, specific_heat_ratio_);
    const auto up = (u + c).abs().maxCoeff();
    const auto um = (u - c).abs().maxCoeff();
    return (cfl_number_ * dx_) / std::max({up, um, minimum_velocity_});
  }

 private:
  double cfl_number_;
  double dx_;
  double minimum_velocity_;
  double specific_heat_ratio_;
};

}  // namespace cfd

#endif  // CFD_TIMESTEP_LENGTH_CALCULATOR_HPP