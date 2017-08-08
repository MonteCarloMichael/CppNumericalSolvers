//
// Created by Michael Heuer on 06.02.17.
//

#include <gmock/gmock.h>
#include "newtonraphsonsolver.h"
#include "TestProblems.cpp"
#include <Eigen/Eigenvalues>

using namespace testing;

class ANewtonRaphsonSolverTest : public Test {};

TEST_F(ANewtonRaphsonSolverTest, QuadraticMinimum) {
  Eigen::VectorXd  x(2);
  x << 4.0, 1.0;
  QuadraticMinimum2DProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  crit.gradNorm = 1e-7;
  cppoptlib::NewtonRaphsonSolver<QuadraticMinimum2DProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 0.0, 0.0;
  ASSERT_TRUE(x.isApprox(xref));
}

TEST_F(ANewtonRaphsonSolverTest, QuarticMinimum) {
  Eigen::VectorXd  x(2);
  x << 1.0, 1.0;
  QuarticMinimum2DProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  crit.gradNorm = 1e-7;
  cppoptlib::NewtonRaphsonSolver<QuarticMinimum2DProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  std::cout << x << std::endl;
  Eigen::VectorXd xref(2);
  xref << 0.0, 0.0;
  ASSERT_TRUE(x.isApprox(xref));
}

TEST_F(ANewtonRaphsonSolverTest, QuadraticSaddlePoint) {
  Eigen::VectorXd  x(2);
  x << 1.0, 1.0;
  QuadraticSaddlePointProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  crit.gradNorm = 1e-7;
  cppoptlib::NewtonRaphsonSolver<QuadraticSaddlePointProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 0.0, 0.0;
  ASSERT_TRUE(x.isApprox(xref));
}

TEST_F(ANewtonRaphsonSolverTest, QuarticSaddlePoint) {
  Eigen::VectorXd  x(2);
  x << 1.0, 1.0;
  QuarticSaddlePoint2DProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  crit.gradNorm = 1e-7;
  cppoptlib::NewtonRaphsonSolver<QuarticSaddlePoint2DProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  std::cout << x << std::endl;
  Eigen::VectorXd xref(2);
  xref << 0.0, 0.0;
  ASSERT_TRUE(x.isApprox(xref));
}
