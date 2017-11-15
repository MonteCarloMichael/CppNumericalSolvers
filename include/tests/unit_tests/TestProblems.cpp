//
// Created by Michael Heuer on 06.02.17.
//
#ifndef TESTPROBLEMS_H
#define TESTPROBLEMS_H

#include <problem.h>
#include <iomanip>
#include <cmath>

class QuadraticMinimum2DProblem : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return x(0)*x(0) + x(1)*x(1);
  }

  void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    grad(0) = 2*x(0);
    grad(1) = 2*x(1);
  }
/*
  void hessian(const TVector &x, THessian &hessian) {
    hessian << 2,0,0,2;
  }*/
};

class QuarticMinimum2DProblem : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
    double value(const Eigen::VectorXd &x) {
      return x(0)*x(0)*x(0)*x(0) + x(1)*x(1)*x(1)*x(1);
    }

    void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
      grad(0) = 4*x(0)*x(0)*x(0);
      grad(1) = 4*x(1)*x(1)*x(1);
    }
/*
    void hessian(const TVector &x, THessian &hessian) {
      hessian << 12*x(0)*x(0),0,0,12*x(1)*x(1);
    }*/
};

class QuadraticSaddlePointProblem : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return x(0)*x(0) - x(1)*x(1);
  }

  void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    grad(0) =  2*x(0);
    grad(1) = -2*x(1);
  }
/*
  void hessian(const TVector &x, THessian &hessian) {
    hessian << 2,0,0,-2;
  }*/
};

class QuarticSaddlePoint2DProblem : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
    double value(const Eigen::VectorXd &x) {
      return x(0)*x(0)*x(0)*x(0) - x(1)*x(1)*x(1)*x(1);
    }

    void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
      grad(0) = 4*x(0)*x(0)*x(0);
      grad(1) = -4*x(1)*x(1)*x(1);
    }
/*
    void hessian(const TVector &x, THessian &hessian) {
      hessian << 12*x(0)*x(0),0,0,-12*x(1)*x(1);
    }*/
    /*bool callback(const cppoptlib::Criteria<double> &state, const Eigen::VectorXd &x) {
      Eigen::VectorXd grad(x);
      gradient(x,grad);
      std::cout << "(" << std::setw(2) << state.iterations << ")"
                << " f(x) = "     << std::fixed << std::setw(8) << std::setprecision(8) << value(x)
                << " gradNorm = " << std::setw(8) << state.gradNorm
                //<< " xDelta = "   << std::setw(8) << state.xDelta
                << " g = [" << std::setprecision(16) << grad.transpose() << "]"
                //<< " x = [" << std::setprecision(16) << x.transpose() << "]"
                << std::endl;
      return true;
    }*/
};


class AbsoluteProblem1D : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return std::abs(x(0));
  }

  void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    if(x(0)>=0) grad(0) =  1.0;
    else        grad(0) = -1.0;
  }
  bool callback(const cppoptlib::Criteria<double> &state, const Eigen::VectorXd &x) {
    Eigen::VectorXd grad(x);
    gradient(x,grad);
    std::cout << "(" << std::setw(2) << state.iterations << ")"
              << " f(x) = "     << std::fixed << std::setw(8) << std::setprecision(8) << value(x)
              << " gradNorm = " << std::setw(8) << state.gradNorm
              //<< " xDelta = "   << std::setw(8) << state.xDelta
              << " g = [" << std::setprecision(16) << grad.transpose() << "]"
              //<< " x = [" << std::setprecision(16) << x.transpose() << "]"
              << std::endl;
    return true;
  }
};

class CuspProblem1D : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return -std::exp(-std::abs(x(0)));
  }

  void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    if(x(0)>=0) grad(0) = +std::exp(-std::abs(x(0)));
    else        grad(0) = -std::exp(-std::abs(x(0)));
  }
};

class CuspProblem2D : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return -std::exp(-std::abs(x(0))) + x(1)*x(1);
  }

  void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    if(x(0)>=0) grad(0) = +std::exp(-std::abs(x(0)));
    else        grad(0) = -std::exp(-std::abs(x(0)));
    grad(1) = 2*x(1);
  }
};

class CuspProblem3D : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return -std::exp(-std::abs(x(0))) + x(1)*x(1) + x(2)*x(2);
  }

  void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    if(x(0)>=0) grad(0) = +std::exp(-std::abs(x(0)));
    else        grad(0) = -std::exp(-std::abs(x(0)));
    grad(1) = 2*x(1);
    grad(2) = 2*x(2);
  }

    Eigen::VectorXd getNucleiPositions(){
        return Eigen::Vector3d(1,2,3);
    }
};

class CuspProblemXD : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  double value(const Eigen::VectorXd &x) {
    return -std::exp(-x.norm());
  }
};

class H2likeProblem : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  Eigen::Vector3d shift = Eigen::Vector3d(0.0,0.0,0.7);

  double value(const Eigen::VectorXd &x) {
    return -( std::exp(-(x.head(3)-shift).norm())+ std::exp(-(x.tail(3)+shift).norm()) );
  }
  bool callback(const cppoptlib::Criteria<double> &state, const Eigen::VectorXd &x) {
    std::cout << "(" << std::setw(2) << state.iterations << ")"
              << " f(x) = "     << std::fixed << std::setw(8) << std::setprecision(8) << value(x)
              << " gradNorm = " << std::setw(8) << state.gradNorm
              << " xDelta = "   << std::setw(8) << state.xDelta
              << " x = [" << std::setprecision(16) << x.transpose() << "]"
              << std::endl;
    return true;
  }
};

class NesterovFirst2D : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  using typename cppoptlib::Problem<double>::Scalar;
  using typename cppoptlib::Problem<double>::TVector;

  double value(const TVector &x) {
    return   0.25*(x[0] -1)*(x[0] -1) + std::abs(x[1] - 2*x[0]*x[0] + 1);
  }
};

class Rosenbrock2D : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
  using typename cppoptlib::Problem<double>::Scalar;
  using typename cppoptlib::Problem<double>::TVector;

  double value(const TVector &x) {
    const double t1 = (1 - x[0]);
    const double t2 = (x[1] - x[0] * x[0]);
    return   t1 * t1 + 100 * t2 * t2;
  }
  void gradient(const TVector &x, TVector &grad) {
    grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
    grad[1]  = 200 * (x[1] - x[0] * x[0]);
  }
};

class UmrigarCuspProblem3D : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
    double value(const Eigen::VectorXd &x) {
        return -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
    }

    void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
        if(x(0)>=0) grad(0) = +std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        else        grad(0) = -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        if(x(1)>=0) grad(1) = +std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        else        grad(1) = -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        if(x(2)>=0) grad(2) = +std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        else        grad(2) = -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
    }

    Eigen::VectorXd getNucleiPositions(){
        return Eigen::Vector3d(0,0,0);
    }
};

class UmrigarCuspSmoothQuadradicProblem6D : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
    double value(const Eigen::VectorXd &x) {
        return -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))))
               +(x(3)-1)*(x(3)-1)+(x(4)-1)*(x(4)-1)+(x(5)-1)*(x(5)-1);
    }

    void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
        if(x(0)>=0) grad(0) = +std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        else        grad(0) = -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        if(x(1)>=0) grad(1) = +std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        else        grad(1) = -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        if(x(2)>=0) grad(2) = +std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        else        grad(2) = -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));

        grad(3) = 2*x(3)-2;
        grad(4) = 2*x(4)-2;
        grad(5) = 2*x(5)-2;
    }

    Eigen::VectorXd getNucleiPositions(){
        return Eigen::Vector3d(0,0,0);
    }
};

class UmrigarTwoCuspsOneNucleusSmoothQuadradicProblem9D : public cppoptlib::Problem<double,Eigen::Dynamic> {
public:
    double value(const Eigen::VectorXd &x) {
        return -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))))
               -std::exp(-(std::abs(x(3))+std::abs(x(4))+std::abs(x(5))))
               +(x(6)-1)*(x(7)-1)+(x(8)-1)*(x(4)-1)+(x(5)-1)*(x(5)-1);
    }

    void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
        if(x(0)>=0) grad(0) = +std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        else        grad(0) = -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        if(x(1)>=0) grad(1) = +std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        else        grad(1) = -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        if(x(2)>=0) grad(2) = +std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));
        else        grad(2) = -std::exp(-(std::abs(x(0))+std::abs(x(1))+std::abs(x(2))));

        if(x(0)>=0) grad(3) = +std::exp(-(std::abs(x(3))+std::abs(x(4))+std::abs(x(5))));
        else        grad(3) = -std::exp(-(std::abs(x(3))+std::abs(x(4))+std::abs(x(5))));
        if(x(1)>=0) grad(4) = +std::exp(-(std::abs(x(3))+std::abs(x(4))+std::abs(x(5))));
        else        grad(4) = -std::exp(-(std::abs(x(3))+std::abs(x(4))+std::abs(x(5))));
        if(x(2)>=0) grad(5) = +std::exp(-(std::abs(x(3))+std::abs(x(4))+std::abs(x(5))));
        else        grad(5) = -std::exp(-(std::abs(x(3))+std::abs(x(4))+std::abs(x(5))));

        grad(6) = 2*x(6)-2;
        grad(7) = 2*x(7)-2;
        grad(8) = 2*x(8)-2;
    }

    Eigen::VectorXd getNucleiPositions(){
        return Eigen::Vector3d(0,0,0);
    }
};

#endif //TESTPROBLEMS_H
