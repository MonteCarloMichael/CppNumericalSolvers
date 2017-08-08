//
// Created by Michael Heuer on 06.02.17.
//

#include <gmock/gmock.h>
#include "solver/timeintegrationsolver.h"
#include "TestProblems.cpp"

using namespace testing;

class ATimeIntegrationSolverTest : public Test {};

TEST_F(ATimeIntegrationSolverTest, Minimum) {
  Eigen::VectorXd  x(2);
  x << 1.0, 1.0;
  QuadraticMinimum2DProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 1000;
  crit.gradNorm = 1e-8;
  cppoptlib::TimeIntegrationSolver<QuadraticMinimum2DProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 0.0, 0.0;
  ASSERT_TRUE((x-xref).norm() < 1e-6);
}
