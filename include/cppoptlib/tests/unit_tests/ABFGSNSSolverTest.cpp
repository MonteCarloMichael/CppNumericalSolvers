//
// Created by Michael Heuer on 06.02.17.
//

#include <gmock/gmock.h>
#include "bfgsnssolver.h"
#include "TestProblems.h"

using namespace testing;

class ABfgsNsSolverTest : public Test {};



TEST_F(ABfgsNsSolverTest, Minimum) {
  Eigen::VectorXd  x(2);
  x << 1.0, 1.0;
  MinimumProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::BfgsNsSolver<MinimumProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 0.0, 0.0;
  ASSERT_TRUE(x.isApprox(xref));
}

TEST_F(ABfgsNsSolverTest, Cusp) {
  Eigen::VectorXd  x(1);
  x << 3.0;
  CuspProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::BfgsNsSolver<CuspProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);
  std::cout << "f in argmin " << f(x) << std::endl;
  std::cout << "Solver status: " << solver.status() << std::endl;
  std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;

  Eigen::VectorXd xref(1);
  xref << 0.0;
  std::cout << x << std::endl;
  //ASSERT_TRUE(x.isApprox(xref,0.1));
  ASSERT_TRUE( (x-xref).norm() < 0.001);

}