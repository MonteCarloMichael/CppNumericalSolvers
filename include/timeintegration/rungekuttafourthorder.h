// CppNumericalSolver
#ifndef RUNGEKUTTAFOURTHORDER_H_
#define RUNGEKUTTAFOURTHORDER_H_

#include <Eigen/Core>

namespace cppoptlib {

    template<typename ProblemType>
    class RungeKuttaFourthOrder {
    public:
        using Scalar = typename ProblemType::Scalar;
        using TVector = typename ProblemType::TVector;

        static TVector getStep(const TVector &x, ProblemType &objFunc, const Scalar timeStep = 1.0) {
          TVector grad(x.rows());

          objFunc.gradient(x, grad);
          Eigen::VectorXd k1 = timeStep * grad;

          objFunc.gradient(x + 0.5*k1, grad);
          Eigen::VectorXd k2 = timeStep * grad;

          objFunc.gradient(x + 0.5*k2, grad);
          Eigen::VectorXd k3 = timeStep * grad;

          objFunc.gradient(x + k3, grad);
          Eigen::VectorXd k4 = timeStep * grad;

          return -k1/6.0 -k2/3.0 -k3/3.0 -k4/6.0;
        }
    };
} /* namespace cppoptlib */

#endif /* RUNGEKUTTAFOURTHORDER_H_ */
