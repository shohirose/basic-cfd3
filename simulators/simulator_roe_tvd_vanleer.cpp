#include "cfd/cfd.hpp"
#include "problem_settings.hpp"

namespace fs = std::filesystem;

using Eigen::VectorXd, Eigen::seqN;

using Simulator = cfd::EulerEquationSimulator1d<
    cfd::RiemannFluxCalculator<
        cfd::TvdSpacialReconstructor<cfd::VanLeerLimiter>,
        cfd::RoeRiemannSolver>,
    cfd::RungeKutta2ndOrderTimeIntegrator>;

int main(int argc, char** argv) {
  const auto params = make_parameters();
  const auto simulator = Simulator{params};
  const auto V0 = make_initial_condition(params);
  const auto Vn = simulator.run(V0);

  const auto domain = seqN(params.n_bounary_cells, params.n_domain_cells);
  cfd::TextFileWriter writer(fs::path("result/roe_tvd_vanleer"));
  writer.write(Vn(domain, 0), "rho.txt");
  writer.write(Vn(domain, 1), "u.txt");
  writer.write(Vn(domain, 2), "p.txt");

  const VectorXd x = make_x(params);
  writer.write(x, "x.txt");
}
