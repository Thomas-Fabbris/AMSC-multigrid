cmake_minimum_required(VERSION 3.16)
project(multigrid_amg LANGUAGES CXX)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No CMAKE_BUILD_TYPE specified. Defaulting to Release.")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find Eigen
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
	if(DEFINED ENV{mkEigenInc})
		message(STATUS "Eigen3 not found via CMake; using mkEigenInc=$ENV{mkEigenInc}")
		add_library(Eigen3::Eigen INTERFACE IMPORTED)
		set_target_properties(Eigen3::Eigen PROPERTIES
			INTERFACE_INCLUDE_DIRECTORIES "$ENV{mkEigenInc}")
	else()
		message(FATAL_ERROR "Eigen3 not found. Install Eigen3 or export mkEigenInc to the include path (inside container).")
	endif()
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Project include directories
set(PROJECT_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Main executable (loads matrix/RHS from file)
add_executable(main
	src/main.cpp
	src/AMG.cpp
)
target_include_directories(main PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(main PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_options(main PRIVATE -Wall -Wextra -pedantic)

# Example executable (generates test problems)
add_executable(example
	tests/example.cpp
	src/AMG.cpp
)
target_include_directories(example PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(example PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_options(example PRIVATE -Wall -Wextra -pedantic)

enable_testing()
add_test(NAME main COMMAND main matrix_1d_poisson.mtx rhs_1d.txt)
add_test(NAME example COMMAND example)

# Convergence test executable (generates convergence data)
add_executable(convergence_test
	tests/convergence_test.cpp
	src/AMG.cpp
)
target_include_directories(convergence_test PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(convergence_test PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_options(convergence_test PRIVATE -Wall -Wextra -pedantic)
