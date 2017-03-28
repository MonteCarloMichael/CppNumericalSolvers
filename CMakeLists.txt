cmake_minimum_required(VERSION 3.5)


#============= PROJECT cppoptlib ===================
project(cppoptlib)

set ( PROJECT_FOLDER_NAME ${PROJECT_NAME})

set ( PROJECT_PATH ${PROJECT_NAME})
set ( OUTPUT_NAME "${PROJECT_FOLDER_NAME}" )
message ( "-- ++ CREATING PROJECT OR MAKEFILE ---> ${OUTPUT_NAME}" )

set(CMAKE_CXX_FLAGS "-g -Wall")

include_directories(
        include
        include/linesearch
        include/solver
        )

set(HEADER_FILES
        include/boundedproblem.h
        include/meta.h
        include/problem.h
        include/timer.h
        include/linesearch/armijo.h
        include/linesearch/morethuente.h
        include/linesearch/wolfeheuristic.h
        include/solver/bfgssolver.h
        include/solver/lbfgssolver.h
        include/solver/lbfgsbsolver.h
        include/solver/cmaesbsolver.h
        include/solver/cmaesbsolver.h
        include/solver/cmaessolver.h
        include/solver/conjugatedgradientdescentsolver.h
        include/solver/gradientdescentsolver.h
        include/solver/gradientdescentnssolver.h
        include/solver/gradientdescentsimplesolver.h
        include/solver/isolver.h
        include/solver/neldermeadsolver.h
        include/solver/newtondescentsolver.h
        include/solver/newtonraphsonsolver.h
        include/solver/bfgsnssolver.h
        include/linesearch/armijowolfe.h
        include/linesearch/smallestvectorinconvexhullfinder.h
        )

set(SOURCE_FILES
        include/empty.cpp
        )#TODO this is a very bad solution used instead of setting the target properties
add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} ${HEADER_FILES})

#add_library(${PROJECT_NAME} STATIC ${HEADER_FILES})
#set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(${PROJECT_NAME}
        eigen
        )

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

add_library(Externals::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

add_subdirectory(include/tests)