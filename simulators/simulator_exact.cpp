#include <Eigen/Core>
#include <cmath>
#include <iostream>

#include "cfd/text_file_writer.hpp"

using Eigen::ArrayXd;
namespace fs = std::filesystem;

double calc_sonic_velocity(double pressure, double density, double gamma) {
  return std::sqrt(gamma * pressure / density);
}

double fun(double p21, double pL, double uL, double rhoL, double cL, double pR,
           double uR, double rhoR, double cR, double gamma);

template <typename F>
double solve(F&& objfun, double x0, double dx, double eps, int itermax) {
  double x = x0;

  int iter = 0;
  double err = 1;
  do {
    ++iter;
    const auto f1 = objfun(x);
    const auto f2 = objfun(x + dx);
    const auto dfdx = (f2 - f1) / dx;
    const auto x_new = x - f1 / dfdx;
    err = std::fabs(x_new - x) / x;
    x = x_new;
  } while (err > eps && iter <= itermax);

  if (iter > itermax) {
    std::cerr << "Warning: Max iteration reached!" << std::endl;
  }

  return x;
}

int main(int argc, char** argv) {
  const double gamma = 1.4;

  const double rho1 = 0.125;
  const double p1 = 0.1;
  const double u1 = 0.0;
  const double c1 = calc_sonic_velocity(p1, rho1, gamma);

  const double rho5 = 1.0;
  const double p5 = 1.0;
  const double u5 = 0.0;
  const double c5 = calc_sonic_velocity(p5, rho5, gamma);

  const auto p21 = solve(
      [=](double p21) -> double {
        return fun(p21, p5, u5, rho5, c5, p1, u1, rho1, c1, gamma);
      },
      p1 / p5, 1e-6, 1e-8, 30);

  const auto b = (gamma - 1) / (gamma + 1);
  const auto rho2 = rho1 * (p21 + b) / (b * p21 + 1);
  const auto u2 = u1 + c1 * std::sqrt(2 / gamma) * (p21 - 1) /
                           std::sqrt(gamma - 1 + p21 * (gamma + 1));
  const auto p2 = p21 * p1;
  const auto c2 = calc_sonic_velocity(p2, rho2, gamma);

  const auto u3 = u2;
  const auto p3 = p2;
  const auto rho3 = rho5 * std::pow(p3 / p5, 1 / gamma);
  const auto c3 = calc_sonic_velocity(p3, rho3, gamma);

  const auto vs =
      u1 + c1 * std::sqrt((gamma + 1) / (2 * gamma) * (p21 - 1) + 1);
  const auto vc = u3;
  const auto vrt = u3 - c3;
  const auto vrh = u5 - c5;

  const double x0 = 0;
  const double t = 0.48;
  const auto xs = x0 + vs * t;
  const auto xc = x0 + vc * t;
  const auto xrt = x0 + vrt * t;
  const auto xrh = x0 + vrh * t;

  const int n = 500;
  const ArrayXd xe = ArrayXd::LinSpaced(n + 1, -1.0, 1.0);
  const ArrayXd x = 0.5 * (xe.head(n) + xe.tail(n));
  Eigen::ArrayXd rho(n);
  Eigen::ArrayXd p(n);
  Eigen::ArrayXd u(n);

  for (int i = 0; i < n; ++i) {
    if (x(i) < xrh) {
      rho(i) = rho5;
      p(i) = p5;
      u(i) = u5;
    } else if (x(i) < xrt) {
      u(i) = 2 / (gamma + 1) * (0.5 * (gamma - 1) * u5 + c5 + (x(i) - x0) / t);
      const auto c4 = c5 - 0.5 * (gamma - 1) * (u(i) - u5);
      p(i) = p5 * std::pow(c4 / c5, 2 * gamma / (gamma - 1));
      rho(i) = rho5 * std::pow(p(i) / p5, 1 / gamma);
    } else if (x(i) < xc) {
      rho(i) = rho3;
      p(i) = p3;
      u(i) = u3;
    } else if (x(i) < xs) {
      rho(i) = rho2;
      p(i) = p2;
      u(i) = u2;
    } else {
      rho(i) = rho1;
      p(i) = p1;
      u(i) = u1;
    }
  }

  cfd::TextFileWriter writer(fs::path("exact_solution"));
  writer.write(x.matrix(), "x.txt");
  writer.write(rho.matrix(), "rho.txt");
  writer.write(u.matrix(), "u.txt");
  writer.write(p.matrix(), "p.txt");
}

double fun(double p21, double pL, double uL, double rhoL, double cL, double pR,
           double uR, double rhoR, double cR, double gamma) {
  const auto a1 = 0.5 * (gamma + 1) / gamma * (p21 - 1) + 1;
  const auto a2 = uL - uR - cR * (p21 - 1) / (gamma * std::sqrt(a1));
  const auto b1 = 2 * gamma / (gamma - 1);
  return pL / pR * std::pow(1 + 0.5 * (gamma - 1) / cL * a2, b1) - p21;
}
