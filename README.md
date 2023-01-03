# basic-cfd3: Basic CFD
C++ source code for basic computational fluid dynamics.
Problems are taken from [1].

1-D Euler equation is given by

$$
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}}{\partial x} = 0
$$

where $\mathbf{U}$ is conservative variable, and $\mathbf{F}$ is flux function. $\mathbf{U}$ and $\mathbf{F}$ are given by

$$
\mathbf{U} = 
\begin{bmatrix}
\rho \\
\rho u \\
\rho E
\end{bmatrix}
, \quad
\mathbf{F} = 
\begin{bmatrix}
\rho u \\
\rho u^2 + p \\
\rho E u + pu
\end{bmatrix}
$$

where $p$ is pressure, $u$ is velocity, $E$ is total energy, and $\rho$ is density.

The problem is solved by using the finie difference method. The following Riemann solvers are implemented:

- Steger-Warming Riemann solver (Flux vector splitting scheme)
- Roe Riemann solver (Flux difference splitting scheme)

For time integration, the explicit Euler scheme is used. No-flow boundary condition is imposed for left and right boundaries.

Please refer to [1] for the details of each scheme.

# How to compile

Run the following commands under the root directory of the project:

```
$ cmake -S . -B build
$ cmake --build build
```

Please note that the project depends on [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library, which is automatically downloaded and built by CMake using the `FetchContent` module.

Then, run all simulators and get results. For Linux,

```
$ ./run_all.sh
```

For Windows,

```
$ .\run_all.ps1
```

To visualize results, open [`plot.ipynb`](./plot.ipynb) with Jupyter Lab, and run all cells.

# References
1. 肖鋒・長﨑孝夫　2020　数値流体解析の基礎 －Visual C++とgnuplotによる圧縮性・非圧縮性流体解析－　コロナ社