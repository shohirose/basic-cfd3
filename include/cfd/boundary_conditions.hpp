#ifndef CFD_BOUNDARY_CONDITIONS_HPP
#define CFD_BOUNDARY_CONDITIONS_HPP

#include <Eigen/Core>

#include "cfd/problem_parameters.hpp"

namespace cfd {

class NoFlowBoundary {
 public:
  NoFlowBoundary(int n_boundary_cells, int n_domain_cells)
      : n_boundary_cells_{n_boundary_cells}, n_domain_cells_{n_domain_cells} {}

  NoFlowBoundary(const ProblemParameters& params)
      : n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  /**
   * @brief Impose no-flow boundary condition
   *
   * @param U Conservation variables vector
   *
   * U has a shape of (n_total_cells, 3). Each column contains density, moment
   * density, and total energy density, respectively.
   *
   * U(:, 0) = density
   * U(:, 1) = moment density
   * U(:, 2) = total energy density
   *
   * TODO: This is actually not a no-flow boundary condition. Modification is
   * required.
   */
  template <typename Derived>
  void apply(Eigen::MatrixBase<Derived>& U) const noexcept {
    using Eigen::seqN, Eigen::all;
    const auto n = n_boundary_cells_;
    const auto j = n_boundary_cells_ + n_domain_cells_;
    for (int i = 0; i < 3; ++i) {
      U(seqN(0, n), i).array() = U(n, i);
      U(seqN(j, n), i).array() = U(j - 1, i);
    }
  }

 private:
  int n_boundary_cells_;  ///> Number of boundary cells
  int n_domain_cells_;    ///> Number of domain cells
};

}  // namespace cfd

#endif  // CFD_BOUNDARY_CONDITIONS_HPP