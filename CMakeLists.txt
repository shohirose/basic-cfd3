cmake_minimum_required(VERSION 3.13)

include(FetchContent)

# ----------------------------- Eigen library ----------------------------------
FetchContent_Declare(
    Eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)

# FetchContent_Declare(
#     fmt
#     URL https://github.com/fmtlib/fmt/releases/download/9.1.0/fmt-9.1.0.zip
#     DOWNLOAD_EXTRACT_TIMESTAMP ON
#     )

FetchContent_MakeAvailable(Eigen)
# FetchContent_MakeAvailable(fmt)

project(basic-cfd3 CXX)

add_library(cfd
        src/boundary_conditions.cpp
        src/riemann_solvers.cpp
        src/spacial_reconstruction_schemes.cpp
        src/time_integration_schemes.cpp
        src/variables.cpp
        src/text_file_writer.cpp
    )
target_include_directories(cfd PUBLIC include/)
target_link_libraries(cfd
    PUBLIC
        Eigen3::Eigen
    )
target_compile_features(cfd PUBLIC cxx_std_17)
target_compile_options(cfd
    PUBLIC
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W3>
        $<$<CXX_COMPILER_ID:Intel>:$<IF:$<PLATFORM_ID:Windows>,/W3,-w3>>
        $<$<CXX_COMPILER_ID:IntelLLVM>:$<IF:$<PLATFORM_ID:Windows>,/W3,-w3>>
    )

function(add_simulator name)
    add_executable(${name} simulators/${name}.cpp)
    target_link_libraries(${name} PRIVATE cfd)
    target_compile_definitions(${name}
        PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES NOMINMAX>
        )
    target_include_directories(${name} PRIVATE simulators/)
endfunction()

add_simulator(simulator_sw_1st_order)
add_simulator(simulator_roe_1st_order)
add_simulator(simulator_lax_wendroff)
add_simulator(simulator_exact)