cmake_minimum_required(VERSION 3.6)


#============= PROJECT cppoptlib ===================
project(cppoptlib)

set ( PROJECT_FOLDER_NAME ${PROJECT_NAME})

set ( PROJECT_PATH ${PROJECT_NAME})
set ( OUTPUT_NAME "${PROJECT_FOLDER_NAME}" )
message ( "-- ++ CREATING PROJECT OR MAKEFILE ---> ${OUTPUT_NAME}" )

set(CMAKE_CXX_FLAGS "-g -Wall")

# Find Eigen lib
find_package(Eigen3 REQUIRED)

include_directories(
        include/cppoptlib
        include/cpppoptlib/linesearch
        include/cppoptlib/solver
        ${EIGEN3_INCLUDE_DIR}
)

set(HEADER_FILES
        include/cppoptlib/boundedproblem.h
        include/cppoptlib/meta.h
        include/cppoptlib/problem.h
        include/cppoptlib/timer.h
        include/cppoptlib/linesearch/armijo.h
        include/cppoptlib/linesearch/morethuente.h
        include/cppoptlib/linesearch/wolfeheuristic.h
        include/cppoptlib/solver/bfgssolver.h
        include/cppoptlib/solver/lbfgssolver.h
        include/cppoptlib/solver/lbfgsbsolver.h
        include/cppoptlib/solver/cmaesbsolver.h
        include/cppoptlib/solver/cmaesbsolver.h
        include/cppoptlib/solver/cmaessolver.h
        include/cppoptlib/solver/conjugatedgradientdescentsolver.h
        include/cppoptlib/solver/gradientdescentsolver.h
        include/cppoptlib/solver/isolver.h
        include/cppoptlib/solver/neldermeadsolver.h
        include/cppoptlib/solver/newtondescentsolver.h
        )

set(SOURCE_FILES
        include/cppoptlib/empty.cpp
        )#TODO this is a very bad solution used instead of setting the target properties

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} ${HEADER_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )
#set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

add_library(Externals::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
