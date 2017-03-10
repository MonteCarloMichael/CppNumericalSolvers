//
// Created by Michael Heuer on 06.02.17.
//

#include <gmock/gmock.h>
#include "newtonraphsonsolver.h"
#include "TestProblems.cpp"
using namespace testing;

class ANewtonRaphsonSolverTest : public Test {};

TEST_F(ANewtonRaphsonSolverTest, Minimum) {
  Eigen::VectorXd  x(2);
  x << 1.0, 1.0;
  Minimum2DProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::NewtonRaphsonSolver<Minimum2DProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 0.0, 0.0;
  ASSERT_TRUE(x.isApprox(xref));
}

TEST_F(ANewtonRaphsonSolverTest, SaddlePoint) {
  Eigen::VectorXd  x(2);
  x << 0.3, 0.3;
  SaddlePointProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::NewtonRaphsonSolver<SaddlePointProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 0.0, 0.0;
  ASSERT_TRUE(x.isApprox(xref));
}