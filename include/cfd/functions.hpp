#ifndef CFD_FUNCTIONS_HPP
#define CFD_FUNCTIONS_HPP

#include <Eigen/Core>

namespace cfd {

template <typename Derived1, typename Derived2>
Eigen::ArrayXd calc_velocity(
    const Eigen::ArrayBase<Derived1>& momentum,
    const Eigen::ArrayBase<Derived2>& density) noexcept {
  return momentum / density;
}

template <typename Derived1, typename Derived2, typename Derived3>
Eigen::ArrayXd calc_pressure(const Eigen::ArrayBase<Derived1>& momentum,
                             const Eigen::ArrayBase<Derived2>& density,
                             const Eigen::ArrayBase<Derived3>& total_energy,
                             double specific_heat_ratio) noexcept {
  return (specific_heat_ratio - 1.0) *
         (total_energy - 0.5 * momentum.square() / density);
}

template <typename Derived1, typename Derived2>
Eigen::ArrayXd calc_momentum(
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
Eigen::ArrayXd calc_total_energy(const Eigen::ArrayBase<Derived1>& pressure,
                                 const Eigen::ArrayBase<Derived2>& velocity,
                                 const Eigen::ArrayBase<Derived3>& momentum,
                                 double specific_heat_ratio) noexcept {
  return 0.5 * momentum * velocity + pressure / (specific_heat_ratio - 1.0);
}

template <typename Derived1, typename Derived2, typename Derived3>
Eigen::ArrayXd calc_total_enthalpy(
    const Eigen::ArrayBase<Derived1>& pressure,
    const Eigen::ArrayBase<Derived2>& density,
    const Eigen::ArrayBase<Derived3>& total_energy) noexcept {
  return (total_energy + pressure) / density;
}

}  // namespace cfd

#endif  // CFD_FUNCTIONS_HPP