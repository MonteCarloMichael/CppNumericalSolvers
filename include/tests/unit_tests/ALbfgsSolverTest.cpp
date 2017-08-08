//
// Created by Michael Heuer on 06.02.17.
//

#include <gmock/gmock.h>
#include "lbfgssolver.h"
#include "TestProblems.cpp"

using namespace testing;

class ALbfgsSolverTest : public Test {};



TEST_F(ALbfgsSolverTest, Minimum) {
  Eigen::VectorXd  x(2);
  x << 1.0, 1.0;
  QuadraticMinimum2DProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::LbfgsSolver<QuadraticMinimum2DProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 0.0, 0.0;
  ASSERT_TRUE(x.isApprox(xref));
}

TEST_F(ALbfgsSolverTest, Cusp) {
  Eigen::VectorXd  x(1);
  x << 1.0;
  CuspProblem1D f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::LbfgsSolver<CuspProblem1D> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);
  std::cout << "f in argmin " << f(x) << std::endl;
  std::cout << "Solver status: " << solver.status() << std::endl;
  std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;

  Eigen::VectorXd xref(1);
  xref << 0.0;
  ASSERT_TRUE(x.isApprox(xref));
}