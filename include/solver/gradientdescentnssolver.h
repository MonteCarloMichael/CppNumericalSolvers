// CppNumericalSolver
#ifndef GRADIENTDESCENTNSSOLVER_H_
#define GRADIENTDESCENTNSSOLVER_H_

#include <Eigen/Core>
#include "isolver.h"
#include "../linesearch/armijowolfe.h"
#include "../linesearch/smallestvectorinconvexhullfinder.h"

namespace cppoptlib {

template<typename ProblemType>
class GradientDescentnsSolver : public ISolver<ProblemType, 1> {

public:
  using Superclass = ISolver<ProblemType, 1>;
  using typename Superclass::Scalar;
  using typename Superclass::TVector;

  /**
   * @brief minimize
   * @details [long description]
   *
   * @param objFunc [description]
   */
  void minimize(ProblemType &objFunc, TVector &x0) {
    const size_t DIM = x0.rows();
    const size_t MaxIt = Superclass::m_stop.iterations+1;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    TVector grad(DIM);
    objFunc.gradient(x0, grad);

    TVector x_old = x0;

    // members for nonsmooth opt
    MatrixType gradientSet = MatrixType::Zero(DIM,MaxIt);
    gradientSet.col(0) = grad;

    const int J = 50; // J = 1 reduces to the usual stopping condition, when f is smooth
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> j(MaxIt,1);
    j(0) = 1;

    int k;
    this->m_current.reset();
    do {
      k = this->m_current.iterations;

      const Scalar rate = ArmijoWolfe<ProblemType, 1>::linesearch(x0, -grad, objFunc);
      x0 = x0 - rate * grad;
      this->m_current.xDelta = (x_old - x0).norm();

      TVector grad_old = grad;
      objFunc.gradient(x0, grad);
      gradientSet.col(k) = grad;

      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gradientSetSelection;

      // if the current difference in the parameter vector is 10 times as large as the stop criterion, store only
      // the current gradient to do a normal BFGS step
      if ( this->m_current.xDelta > this->m_stop.xDelta *10 ){
        j(k) = 1;
        gradientSetSelection.resize(DIM,1);
        gradientSetSelection.col(0) = grad;
      }
        // else add gradient to the set of prior gradients to do a BFGS-NS step including a convex hull search
        // which calculates a new current gradient estimate
      else {
        // check second last element
        if ( j(k-1) < J ) {
          j(k) = j(k-1) + 1;
          gradientSetSelection.resize(DIM,j(k));
          gradientSetSelection = gradientSet.block(0,k-j(k)+1,DIM,j(k));// jk elements //k - ( k-j(k)+1 ) +1

        }
        else { // j(k-1) == J)
          j(k) = J;
          gradientSetSelection.resize(DIM,J);
          gradientSetSelection = gradientSet.block(0,k-J+1,DIM,J); // J elements
        }
      }
      if( j(k) > 1 ) {
        SmallestVectorInConvexHullFinder<Scalar> finder;
        finder.resizeFinder(DIM, gradientSetSelection.rows());
        grad = finder.findSmallestVectorInConvexHull(gradientSetSelection,this->m_stop.xDelta,this->m_stop.xDelta).second;
      }

      x_old = x0;


      this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity>();
      // std::cout << "iter: "<<iter<< " f = " <<  objFunc.value(x0) << " ||g||_inf "<<gradNorm  << std::endl;
      ++this->m_current.iterations;
      this->m_status = checkConvergence(this->m_stop, this->m_current);
    } while (objFunc.callback(this->m_current, x0, grad) && (this->m_status == Status::Continue));
    if (this->m_debug > DebugLevel::None) {
        std::cout << "Stop status was: " << this->m_status << std::endl;
        std::cout << "Stop criteria were: " << std::endl << this->m_stop << std::endl;
        std::cout << "Current values are: " << std::endl << this->m_current << std::endl;
    }
  }

};

} /* namespace cppoptlib */

#endif /* GRADIENTDESCENTNSSOLVER_H_ */
