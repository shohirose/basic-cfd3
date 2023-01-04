#ifndef CFD_FUNCTIONS_HPP
#define CFD_FUNCTIONS_HPP

#include <Eigen/Core>
#include <tuple>

namespace cfd {

namespace matrix_api {

/**
 * @brief Compute primitive variables (velocity, pressure, total enthalpy)
 *
 * @tparam Derived
 * @param U Conservation variables vector
 * @param specific_heat_ratio Specific heat ratio
 * @return std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
 * velocity, pressure, and total enthalpy
 */
template <typename Derived>
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> to_primitive_vars(
    const Eigen::MatrixBase<Derived>& U, double specific_heat_ratio) noexcept {
  using Eigen::VectorXd;
  const VectorXd velocity = U.col(1).cwiseQuotient(U.col(0));
  const VectorXd pressure =
      (specific_heat_ratio - 1) *
      (U.col(2) -
       0.5 * U.col(1).array().square().matrix().cwiseQuotient(U.col(0)));
  const VectorXd total_enthalpy = (U.col(2) + pressure).cwiseQuotient(U.col(0));
  return std::make_tuple(std::move(velocity), std::move(pressure),
                         std::move(total_enthalpy));
}

template <typename Derived>
Eigen::VectorXd velocity(const Eigen::MatrixBase<Derived>& U) noexcept {
  return U.col(1).cwiseQuotient(U.col(0));
}

template <typename Derived>
Eigen::VectorXd pressure(const Eigen::MatrixBase<Derived>& U,
                         double specific_heat_ratio) noexcept {
  return (specific_heat_ratio - 1) *
         (U.col(2) -
          0.5 * U.col(1).array().square().matrix().cwiseQuotient(U.col(0)));
}

template <typename Derived1, typename Derived2>
Eigen::VectorXd sonic_velocity(const Eigen::MatrixBase<Derived1>& pressure,
                               const Eigen::MatrixBase<Derived2>& density,
                               double specific_heat_ratio) noexcept {
  return (specific_heat_ratio * pressure.cwiseQuotient(density)).cwiseSqrt();
}

}  // namespace matrix_api

namespace array_api {

/**
 * @brief Compute primitive variables (velocity, pressure, total enthalpy)
 *
 * @tparam Derived
 * @param U Conservation variables vector
 * @param specific_heat_ratio Specific heat ratio
 * @return std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd>
 * velocity, pressure, and total enthalpy
 */
template <typename Derived>
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd> to_primitive_vars(
    const Eigen::MatrixBase<Derived>& U, double specific_heat_ratio) noexcept {
  using Eigen::ArrayXd;
  const ArrayXd velocity = U.col(1).array() / U.col(0).array();
  const ArrayXd pressure =
      (specific_heat_ratio - 1) *
      (U.col(2).array() - 0.5 * U.col(1).array().square() / U.col(0).array());
  const ArrayXd total_enthalpy =
      (U.col(2).array() + pressure) / U.col(0).array();
  return std::make_tuple(std::move(velocity), std::move(pressure),
                         std::move(total_enthalpy));
}

template <typename Derived>
Eigen::ArrayXd velocity(const Eigen::MatrixBase<Derived>& U) noexcept {
  return U.col(1).cwiseQuotient(U.col(0));
}

template <typename Derived>
Eigen::ArrayXd pressure(const Eigen::MatrixBase<Derived>& U,
                        double specific_heat_ratio) noexcept {
  return (specific_heat_ratio - 1) *
         (U.col(2) -
          0.5 * U.col(1).array().square().matrix().cwiseQuotient(U.col(0)));
}

template <typename Derived1, typename Derived2>
Eigen::ArrayXd sonic_velocity(const Eigen::ArrayBase<Derived1>& pressure,
                              const Eigen::ArrayBase<Derived2>& density,
                              double specific_heat_ratio) noexcept {
  return (specific_heat_ratio * pressure / density).sqrt();
}

}  // namespace array_api

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
 * @brief Convert conservation variables vector to primitive variables vector.
 *
 * @tparam Derived
 * @param U Conservation variables vector
 * @param specific_heat_ratio Specific heat ratio
 * @return Eigen::MatrixXd Primitive variables vector
 *
 * Conservation variables vector @f \mathbf{U} @f$ is defined by
 * @f[
 * \mathbf{U} =
 * \begin{bmatrix}
 * \rho \\ \rho u \\ \rho E
 * \end{bmatrix}
 * where @f$ \rho @f$ is density, @f$ u @f$ is velocity, and @f$ E @f$ is total
 * energy. Primitive variables vector is defined by
 * @f[
 * \mathbf{V} =
 * \begin{bmatrix}
 * \rho \\ u \\ p
 * \end{bmatrix}
 * where @f$ p @f$ is pressure.
 * @f]
 */
template <typename Derived>
Eigen::MatrixXd to_primitive_vars(const Eigen::MatrixBase<Derived>& U,
                                  double specific_heat_ratio) noexcept {
  Eigen::MatrixXd V(U.rows(), 3);
  V.col(0) = U.col(0);
  V.col(1) = calc_velocity(U.col(1).array(), U.col(0).array());
  V.col(2) = calc_pressure(U.col(1).array(), U.col(0).array(), U.col(2).array(),
                           specific_heat_ratio)
                 .matrix();
  return V;
}

/**
 * @brief Convert primitive varialbes vector to conservation variables vector.
 *
 * @tparam Derived
 * @param V Primitive variables vector
 * @param specific_heat_ratio
 * @return Eigen::MatrixXd Conservation variables vector
 *
 * Conservation variables vector @f \mathbf{U} @f$ is defined by
 * @f[
 * \mathbf{U} =
 * \begin{bmatrix}
 * \rho \\ \rho u \\ \rho E
 * \end{bmatrix}
 * where @f$ \rho @f$ is density, @f$ u @f$ is velocity, and @f$ E @f$ is total
 * energy. Primitive variables vector is defined by
 * @f[
 * \mathbf{V} =
 * \begin{bmatrix}
 * \rho \\ u \\ p
 * \end{bmatrix}
 * where @f$ p @f$ is pressure.
 * @f]
 */
template <typename Derived>
Eigen::MatrixXd to_conservation_vars(const Eigen::MatrixBase<Derived>& V,
                                     double specific_heat_ratio) noexcept {
  Eigen::MatrixXd U(V.rows(), 3);
  U.col(0) = V.col(0);
  U.col(1) = calc_momentum_density(V.col(0).array(), V.col(1).array()).matrix();
  U.col(2) = calc_total_energy_density(V.col(2).array(), V.col(1).array(),
                                       V.col(0).array(), specific_heat_ratio)
                 .matrix();
  return U;
}

}  // namespace cfd

#endif  // CFD_FUNCTIONS_HPP