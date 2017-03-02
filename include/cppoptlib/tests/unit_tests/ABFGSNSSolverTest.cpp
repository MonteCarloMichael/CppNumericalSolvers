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

  //Eigen::VectorXd xreftest(1);
  //xreftest << 0.00000000001;
  std::cout << x << std::endl;

  //ASSERT_TRUE(xref.isApprox(xreftest,0.001));
  //ASSERT_NEAR(x,xref,0.01);
  //ASSERT_EQ(x,xref)
  //ASSERT_TRUE(x.isApprox(xref,0.01));
  ASSERT_TRUE( (x-xref).norm() < 0.001);

}

TEST_F(ABfgsNsSolverTest, Cusp2D) {
  Eigen::VectorXd  x(2);
  x << 3.0,3.0;
  CuspProblem2D f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::BfgsNsSolver<CuspProblem2D> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);
  std::cout << "f in argmin " << f(x) << std::endl;
  std::cout << "Solver status: " << solver.status() << std::endl;
  std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;

  Eigen::VectorXd xref(2);
  xref << 0.0,0.0;
  std::cout << x << std::endl;
  //ASSERT_TRUE(x.isApprox(xref,0.1));
  ASSERT_TRUE( (x-xref).norm() < 0.001);

}

TEST_F(ABfgsNsSolverTest, Cusp3D) {
  Eigen::VectorXd  x(3);
  x << 3.0,3.0,3.0;
  CuspProblem3D f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::BfgsNsSolver<CuspProblem3D> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);
  std::cout << "f in argmin " << f(x) << std::endl;
  std::cout << "Solver status: " << solver.status() << std::endl;
  std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;

  Eigen::VectorXd xref(3);
  xref << 0.0,0.0,0.0;
  std::cout << x << std::endl;
  //ASSERT_TRUE(x.isApprox(xref,0.1));
  ASSERT_TRUE( (x-xref).norm() < 0.001);

}