#include "cfd/cfd.hpp"
#include "problem_settings.hpp"

namespace fs = std::filesystem;

using Eigen::VectorXd, Eigen::seqN;

using Simulator =
    cfd::EulerEquationSimulator1d<cfd::FirstOrderSpacialReconstructor,
                                  cfd::StegerWarmingRiemannSolver>;

Simulator make_simulator(const cfd::ProblemParameters& params) {
  return {params, cfd::FirstOrderSpacialReconstructor{params},
          cfd::StegerWarmingRiemannSolver{params}};
}

int main(int argc, char** argv) {
  const auto params = make_parameters();
  const auto simulator = make_simulator(params);
  const auto V0 = make_initial_condition(params);
  const auto Vn = simulator.run(V0);

  const auto domain = seqN(params.n_bounary_cells, params.n_domain_cells);
  cfd::TextFileWriter writer(fs::path("result/sw_1st_order"));
  writer.write(Vn(domain, 0), "rho.txt");
  writer.write(Vn(domain, 1), "u.txt");
  writer.write(Vn(domain, 2), "p.txt");

  const auto n = params.n_domain_cells;
  const VectorXd xe = VectorXd::LinSpaced(n + 1, -1.0, 1.0);
  const VectorXd x = 0.5 * (xe.head(n) + xe.tail(n));
  writer.write(x, "x.txt");
}
