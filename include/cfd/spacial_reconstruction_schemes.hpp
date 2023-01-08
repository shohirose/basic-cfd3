#ifndef CFD_SPACIAL_RECONSTRUCTION_SCHEMES_HPP
#define CFD_SPACIAL_RECONSTRUCTION_SCHEMES_HPP

#include <Eigen/Core>

#include "cfd/problem_parameters.hpp"

namespace cfd {

class FirstOrderSpacialReconstructor {
 public:
  FirstOrderSpacialReconstructor(int n_boundary_cells, int n_domain_cells)
      : n_boundary_cells_{n_boundary_cells},
        n_domain_cells_{n_boundary_cells} {}

  FirstOrderSpacialReconstructor(const ProblemParameters& params)
      : n_boundary_cells_{params.n_bounary_cells},
        n_domain_cells_{params.n_domain_cells} {}

  template <typename Derived>
  Eigen::MatrixXd calc_left(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    assert(U.rows() == n_boundary_cells_ * 2 + n_domain_cells_);
    assert(U.cols() == 3);

    using Eigen::all, Eigen::seqN;
    return U(seqN(n_boundary_cells_ - 1, n_domain_cells_ + 1), all);
  }

  template <typename Derived>
  Eigen::MatrixXd calc_right(
      const Eigen::MatrixBase<Derived>& U) const noexcept {
    assert(U.rows() == n_boundary_cells_ * 2 + n_domain_cells_);
    assert(U.cols() == 3);

    using Eigen::all, Eigen::seqN;
    return U(seqN(n_boundary_cells_, n_domain_cells_ + 1), all);
  }

 private:
  int n_boundary_cells_;
  int n_domain_cells_;
};

}  // namespace cfd

#endif  // CFD_SPACIAL_RECONSTRUCTION_SCHEMES_HPP