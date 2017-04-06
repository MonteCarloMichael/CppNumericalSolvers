//
// Created by Michael Heuer on 06.02.17.
//

#include <gmock/gmock.h>
#include "bfgsnssolver.h"
#include "gradientdescentsolver.h"
#include "gradientdescentnssolver.h"
#include "gradientdescentsimplesolver.h"
#include "TestProblems.cpp"

using namespace testing;

class ABfgsNsSolverTest : public Test {};



TEST_F(ABfgsNsSolverTest, Minimum2D) {
  Eigen::VectorXd  x(2);
  x << 1.0, 1.0;
  Minimum2DProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100;
  cppoptlib::BfgsnsSolver<Minimum2DProblem> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(2);
  xref << 0.0, 0.0;
  ASSERT_TRUE(x.isApprox(xref));
}

TEST_F(ABfgsNsSolverTest, Absolute1D) {
  Eigen::VectorXd  x(1);
  x << 3.0;
  AbsoluteProblem1D f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::nonsmoothDefaults();
  crit.iterations = 100;
  cppoptlib::BfgsnsSolver<AbsoluteProblem1D> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(1);
  xref << 0.0;
  ASSERT_TRUE( (x-xref).norm() < 0.001);
}

TEST_F(ABfgsNsSolverTest, Cusp1D) {
  Eigen::VectorXd  x(1);
  x << 3.0;
  CuspProblem1D f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::nonsmoothDefaults();
  crit.iterations = 100;
  cppoptlib::BfgsnsSolver<CuspProblem1D> solver;
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

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::nonsmoothDefaults();
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

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::nonsmoothDefaults();
  crit.iterations = 100;
  cppoptlib::BfgsnsSolver<CuspProblem3D> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(3);
  xref << 0.0,0.0,0.0;
  ASSERT_TRUE( (x-xref).norm() < 0.001);
}

TEST_F(ABfgsNsSolverTest, CuspXD) {
  Eigen::VectorXd  x(3);
  x << 3.0,3.0,3.0;
  CuspProblemXD f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::nonsmoothDefaults();
  crit.iterations = 100;
  cppoptlib::BfgsnsSolver<CuspProblemXD> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  Eigen::VectorXd xref(3);
  xref << 0.0,0.0,0.0;
  ASSERT_TRUE( (x-xref).norm() < 0.001);
}

TEST_F(ABfgsNsSolverTest, H2likeProblem) {
  Eigen::VectorXd  x(6);
  x << 0.02,-0.5,1.0,0.0,+0.05,-2.0;
  H2likeProblem f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 100000;
  cppoptlib::BfgsnsSolver<H2likeProblem> solver;

  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);

  std::cout << "f in argmin " << f(x) << std::endl;
  std::cout << "Solver status: " << solver.status() << std::endl;
  std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;

  Eigen::VectorXd xref(6);
  xref << 0.0,0.0,+0.7,0.0,0.0,-0.7;
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
  x << 1.2, 1.2;
  NesterovFirst2D f;

  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 300;
  crit.gradNorm = 1e-4;
  cppoptlib::BfgsnsSolver<NesterovFirst2D> solver;
  solver.setDebug(cppoptlib::DebugLevel::High);
  solver.setStopCriteria(crit);
  solver.minimize(f, x);
  std::cout << "f in argmin " << f(x) << std::endl;
  std::cout << "Solver status: " << solver.status() << std::endl;
  std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;

  Eigen::VectorXd xref(2);
  xref << 1.0,1.0;
  std::cout << x.transpose() << std::endl;
  EXPECT_NEAR( 0.0, f(x), 0.0001 );
  ASSERT_TRUE( (x-xref).norm() < 0.0001);
}

