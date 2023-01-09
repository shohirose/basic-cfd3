#ifndef CFD_FUNCTIONS_HPP
#define CFD_FUNCTIONS_HPP

#include <Eigen/Core>
#include <tuple>

namespace cfd {

template <typename Derived1, typename Derived2>
Eigen::ArrayXd calc_velocity(
    const Eigen::ArrayBase<Derived1>& momentum_density,
    const Eigen::ArrayBase<Derived2>& density) noexcept {
  return momentum_density / density;
}

template <typename Derived1, typename Derived2, typename Derived3>
Eigen::ArrayXd calc_pressure(const Eigen::ArrayBase<Derived1>& momentum_density,
                             const Eigen::ArrayBase<Derived2>& density,
                             const Eigen::ArrayBase<Derived3>& total_energy,
                             double specific_heat_ratio) noexcept {
  return (specific_heat_ratio - 1.0) *
         (total_energy - 0.5 * momentum_density.square() / density);
}

template <typename Derived1, typename Derived2>
Eigen::ArrayXd calc_momentum_density(
    const Eigen::ArrayBase<Derived1>& density,
    const Eigen::ArrayBase<Derived2>& velocity) noexcept {
  return density * velocity;
}

template <typename Derived1, typename Derived2>
Eigen::ArrayXd calc_sonic_velocity(const Eigen::ArrayBase<Derived1>& pressure,
                                   const Eigen::ArrayBase<Derived2>& density,
                                   double specific_heat_ratio) noexcept {
  return (specific_heat_ratio * pressure / density).sqrt();
}

template <typename Derived1, typename Derived2, typename Derived3>
Eigen::ArrayXd calc_total_energy_density(
    const Eigen::ArrayBase<Derived1>& pressure,
    const Eigen::ArrayBase<Derived2>& velocity,
    const Eigen::ArrayBase<Derived3>& momentum,
    double specific_heat_ratio) noexcept {
  return 0.5 * momentum * velocity + pressure / (specific_heat_ratio - 1.0);
}

template <typename Derived1, typename Derived2, typename Derived3>
Eigen::ArrayXd calc_total_enthalpy(
    const Eigen::ArrayBase<Derived1>& pressure,
    const Eigen::ArrayBase<Derived2>& density,
    const Eigen::ArrayBase<Derived3>& total_energy_density) noexcept {
  return (total_energy_density + pressure) / density;
}

/**
 * @brief Compute velocity, pressure, total enthalpy.
 *
 * @tparam Derived
 * @param U Conservation variables vector
 * @param specific_heat_ratio Specific heat ratio
 * @return std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd>
 * velocity, pressure, and total enthalpy
 */
template <typename Derived>
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd>
calc_velocity_pressure_enthalpy(const Eigen::MatrixBase<Derived>& U,
                                double specific_heat_ratio) noexcept {
  using Eigen::ArrayXd;
  const ArrayXd u = calc_velocity(U.col(1).array(), U.col(0).array());
  const ArrayXd p = calc_pressure(U.col(1).array(), U.col(0).array(),
                                  U.col(2).array(), specific_heat_ratio);
  const ArrayXd h = calc_total_enthalpy(p, U.col(0).array(), U.col(2).array());
  return std::make_tuple(std::move(u), std::move(p), std::move(h));
}

/**
 * @brief Compute velocity, pressure, sonic velocity
 *
 * @tparam Derived
 * @param U Conservation variables vector
 * @param specific_heat_ratio Specific heat ratio
 * @return std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd>
 * velocity, pressure, and sonic velocity
 */
template <typename Derived>
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd>
calc_velocity_pressure_sonic_velocity(const Eigen::MatrixBase<Derived>& U,
                                      double specific_heat_ratio) noexcept {
  using Eigen::ArrayXd;
  const ArrayXd u = calc_velocity(U.col(1).array(), U.col(0).array());
  const ArrayXd p = calc_pressure(U.col(1).array(), U.col(0).array(),
                                  U.col(2).array(), specific_heat_ratio);
  const ArrayXd c =
      calc_sonic_velocity(p, U.col(0).array(), specific_heat_ratio);
  return std::make_tuple(std::move(u), std::move(p), std::move(c));
}

}  // namespace cfd

#endif  // CFD_FUNCTIONS_HPP