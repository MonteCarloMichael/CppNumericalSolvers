//
// Created by Michael Heuer on 06.02.17.
//

#include <gmock/gmock.h>
#include "bfgsnssolver.h"
#include "TestProblems.cpp"

using namespace testing;

class ABfgsNsSolverTest : public Test {};



TEST_F(ABfgsNsSolverTest, Minimum) {
  Eigen::VectorXd  x(2);
  x << 1.0, 1.0;
  MinimumProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::BfgsnsSolver<MinimumProblem> solver;
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
  cppoptlib::BfgsnsSolver<CuspProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(1);
  xref << 0.0;
  ASSERT_TRUE( (x-xref).norm() < 0.001);
}

TEST_F(ABfgsNsSolverTest, Cusp2D) {
  Eigen::VectorXd  x(2);
  x << 3.0,3.0;
  CuspProblem2D f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::BfgsnsSolver<CuspProblem2D> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 0.0,0.0;
  ASSERT_TRUE( (x-xref).norm() < 0.001);
}

TEST_F(ABfgsNsSolverTest, Cusp3D) {
  Eigen::VectorXd  x(3);
  x << 3.0,3.0,3.0;
  CuspProblem3D f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::BfgsnsSolver<CuspProblem3D> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(3);
  xref << 0.0,0.0,0.0;
  ASSERT_TRUE( (x-xref).norm() < 0.001);
}

TEST_F(ABfgsNsSolverTest, Rosenbrock2D) {
  Eigen::VectorXd  x(2);
  x << -1.0,2.0; // 3.0, 3.0 works
  Rosenbrock2D f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::BfgsnsSolver<Rosenbrock2D> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 1.0,1.0;
  EXPECT_NEAR( 0.0, f(x), 0.0001 );
  ASSERT_TRUE( (x-xref).norm() < 0.0001);
}

TEST_F(ABfgsNsSolverTest, NesterovFirst2D) {
  Eigen::VectorXd  x(2);
  x << -1.0,0.0; // 3.0, 3.0 works
  NesterovFirst2D f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::BfgsnsSolver<NesterovFirst2D> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 1.0,1.0;
  EXPECT_NEAR( 0.0, f(x), 0.0001 );
  ASSERT_TRUE( (x-xref).norm() < 0.0001);
}

