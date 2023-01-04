#ifndef CFD_FUNCTIONS_HPP
#define CFD_FUNCTIONS_HPP

#include <Eigen/Core>

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

template <typename Derived>
Eigen::MatrixXd to_primitive_vars(const Eigen::DenseBase<Derived>& U,
                                  double gamma) noexcept {
  Eigen::MatrixXd V(U.rows(), 3);
  V.col(0) = U.col(0);
  V.col(1) = calc_velocity(U.col(1).array(), U.col(0).array());
  V.col(2) =
      calc_pressure(U.col(1).array(), U.col(0).array(), U.col(2).array(), gamma)
          .matrix();
  return V;
}

template <typename Derived>
Eigen::MatrixXd to_conservative_vars(const Eigen::DenseBase<Derived>& V,
                                     double gamma) noexcept {
  Eigen::MatrixXd U(V.rows(), 3);
  U.col(0) = V.col(0);
  U.col(1) = calc_momentum_density(V.col(0).array(), V.col(1).array()).matrix();
  U.col(2) = calc_total_energy_density(V.col(2).array(), V.col(1).array(),
                                       V.col(0).array(), gamma)
                 .matrix();
  return U;
}

}  // namespace cfd

#endif  // CFD_FUNCTIONS_HPP