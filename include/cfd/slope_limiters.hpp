#ifndef CFD_SLOPE_LIMITERS_HPP
#define CFD_SLOPE_LIMITERS_HPP

#include <Eigen/Core>

namespace cfd {

/**
 * @brief Minmod limiter
 *
 * @f[
 * \Phi (r) = minmod(1, r) = max(0, min(1, r))
 * @f]
 */
struct MinmodLimiter {
  template <typename Derived>
  static Eigen::ArrayXd eval(const Eigen::ArrayBase<Derived>& r) noexcept {
    return r.min(1.0).max(0.0);
  }
};

/**
 * @brief Superbee limiter
 *
 * @f[
 * \Phi (r) = max(0, min(1, 2r), min(2, r))
 * @f]
 */
struct SuperbeeLimiter {
  template <typename Derived>
  static Eigen::ArrayXd eval(const Eigen::ArrayBase<Derived>& r) noexcept {
    return (2 * r).min(1).max(r.min(2)).max(0);
  }
};

/**
 * @brief van Leer limiter
 *
 * @f[
 * \Phi (r) = \frac{r + |r|}{1 + |r|}
 * @f]
 */
struct VanLeerLimiter {
  template <typename Derived>
  static Eigen::ArrayXd eval(const Eigen::ArrayBase<Derived>& r) noexcept {
    return (r + r.abs()) / (1 + r.abs());
  }
};

/**
 * @brief van Albada limiter
 *
 * @f[
 * \Phi (r) = \frac{r + r^2}{1 + r^2}
 * @f]
 */
struct VanAlbadaLimiter {
  template <typename Derived>
  static Eigen::ArrayXd eval(const Eigen::ArrayBase<Derived>& r) noexcept {
    const Eigen::ArrayXd r2 = r.square();
    return (r + r2) / (1 + r2);
  }
};

}  // namespace cfd

#endif  // CFD_SLOPE_LIMITERS_HPP