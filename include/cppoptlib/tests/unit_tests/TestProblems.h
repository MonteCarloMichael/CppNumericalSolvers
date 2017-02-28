//
// Created by Michael Heuer on 06.02.17.
//
#ifndef TESTPROBLEMS_H
#define TESTPROBLEMS_H

#include <problem.h>
#include <iomanip>

class MinimumProblem : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return x(0)*x(0) + x(1)*x(1);
  }

  void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    grad(0) = 2*x(0);
    grad(1) = 2*x(1);
  }

  void hessian(const TVector &x, THessian &hessian) {
    hessian << 2,0,0,2;
  }

  /*bool callback(const cppoptlib::Criteria<double> &state, const Eigen::VectorXd &x) {
    std::cout << "(" << std::setw(2) << state.iterations << ")"
              << " ||dx|| = " << std::fixed << std::setw(8) << std::setprecision(4) << state.gradNorm
              << " ||x|| = "  << std::setw(6) << x.norm()
              << " f(x) = "   << std::setw(8) << value(x)
              << " x = [" << std::setprecision(16) << x.transpose() << "]" << std::endl;
    return true;
  }*/
};

class SaddlePointProblem : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return x(0)*x(0) - x(1)*x(1);
  }

  void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    grad(0) =  2*x(0);
    grad(1) = -2*x(1);
  }

  void hessian(const TVector &x, THessian &hessian) {
    hessian << 2,0,0,-2;
  }

  /*bool callback(const cppoptlib::Criteria<double> &state, const Eigen::VectorXd &x) {
    std::cout << "(" << std::setw(2) << state.iterations << ")"
              << " ||dx|| = " << std::fixed << std::setw(8) << std::setprecision(4) << state.gradNorm
              << " ||x|| = "  << std::setw(6) << x.norm()
              << " f(x) = "   << std::setw(8) << value(x)
              << " x = [" << std::setprecision(16) << x.transpose() << "]" << std::endl;
    return true;
  }*/
};

class CuspProblem : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return -exp(-abs(x(0)));
  }

  void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    if(x(0)>=0) grad(0) = +exp(-abs(x(0)));
    else        grad(0) = -exp(-abs(x(0)));
  }

  /*bool callback(const cppoptlib::Criteria<double> &state, const Eigen::VectorXd &x) {
    std::cout << "(" << std::setw(2) << state.iterations << ")"
              << " ||dx|| = " << std::fixed << std::setw(8) << std::setprecision(4) << state.gradNorm
              << " ||x|| = " << std::setw(6) << x.norm()
              << " f(x) = " << std::setw(8) << value(x)
              << " x = [" << std::setprecision(16) << x.transpose() << "]" << std::endl;
    return true;
  }*/
};

class CuspProblem2D : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return -exp(-abs(x(0))) + x(1)*x(1);
  }

  void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    if(x(0)>=0) grad(0) = +exp(-abs(x(0)));
    else        grad(0) = -exp(-abs(x(0)));
    grad(1) = 2*x(1);
  }
};

#endif //TESTPROBLEMS_H
