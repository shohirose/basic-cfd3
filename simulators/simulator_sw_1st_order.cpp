#include "problem_settings.hpp"

namespace fs = std::filesystem;

using Eigen::ArrayXd;
using Eigen::seqN;

using Simulator =
    cfd::EulerEquationSimulator1d<cfd::FirstOrderSpacialReconstructor,
                                  cfd::StaggerWarmingRiemannSolver>;

Simulator make_simulator(const cfd::ProblemParameters& params) {
  return {params, cfd::FirstOrderSpacialReconstructor{params},
          cfd::StaggerWarmingRiemannSolver{params}};
}

cfd::PrimitiveVariables make_initial_condition(
    const cfd::ProblemParameters& params);

int main(int argc, char** argv) {
  const auto params = make_parameters();
  const auto simulator = make_simulator(params);
  const auto pvars0 = make_initial_condition(params);
  const auto pvarsN = simulator.run(pvars0);

  const auto domain = seqN(params.n_bounary_cells, params.n_domain_cells);
  const ArrayXd rho = pvarsN.density(domain);
  const ArrayXd u = pvarsN.velocity(domain);
  const ArrayXd p = pvarsN.pressure(domain);

  cfd::TextFileWriter writer(fs::path("result/sw_1st_order"));
  writer.write(rho, "rho.txt");
  writer.write(u, "u.txt");
  writer.write(p, "p.txt");

  const auto n = params.n_domain_cells;
  const ArrayXd xe = ArrayXd::LinSpaced(n + 1, -1.0, 1.0);
  const ArrayXd x = 0.5 * (xe.head(n) + xe.tail(n));
  writer.write(x, "x.txt");
}
