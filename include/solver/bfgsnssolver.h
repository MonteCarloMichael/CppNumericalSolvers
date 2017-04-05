// CppNumericalSolver
//
// Created by Michael Heuer on 06.02.17.
//
#include <iostream>
#include <Eigen/Dense>
#include "isolver.h"
#include "../linesearch/armijowolfe.h"
#include "../linesearch/smallestvectorinconvexhullfinder.h"

#ifndef BFGSNSSOLVER_H_
#define BFGSNSSOLVER_H_

namespace cppoptlib {

  template<typename ProblemType>
  class BfgsnsSolver : public ISolver<ProblemType, 1> {
  public:
    using Superclass = ISolver<ProblemType, 1>;
    using typename Superclass::Scalar;
    using typename Superclass::TVector;
    using typename Superclass::THessian;
    using TCriteria = typename ProblemType::TCriteria;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    BfgsnsSolver() : ISolver<ProblemType,1>( TCriteria::nonsmoothDefaults() ){};

    void minimize(ProblemType &objFunc, TVector & x0) {
      const size_t DIM = x0.rows();
      const size_t MaxIt = Superclass::m_stop.iterations+1;

      THessian H = THessian::Identity(DIM, DIM);
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

        TVector searchDir = -1 * H * grad;
        // check for positive definiteness
        Eigen::LLT<THessian ,Eigen::UpLoType::Lower> choleskyDecomposer(H);
        if ( choleskyDecomposer.info() == Eigen::NumericalIssue ){
          //std::cout << "Hessian not positive definite" << std::endl;
          H = THessian::Identity(DIM, DIM);
          searchDir = -grad;
        }

        // do step
        const Scalar rate = ArmijoWolfe<ProblemType, 1>::linesearch(x0, searchDir, objFunc);
        TVector s = rate * searchDir;
        x0 = x0 + s;
        this->m_current.xDelta = (x_old - x0).norm();


        // store gradient from previous step and calculate the current gradient for the gradient set
        TVector grad_old = grad;
        objFunc.gradient(x0, grad);
        gradientSet.col(k) = grad;

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gradientSetSelection;

        // if the current difference in the parameter vector is 10 times as large as the stop criterion, store only
        // the current gradient to do a normal BFGS step
        if ( this->m_current.xDelta > this->m_stop.xDeltaNonsmooth *10 ){
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
          grad = finder.findSmallestVectorInConvexHull(gradientSetSelection,
                                                       this->m_stop.xDeltaNonsmooth,
                                                       this->m_stop.rsDeltaNonsmooth).second;
        }

        x_old = x0;

        // update the hessian, based on the current gradient (that can be the result of the convex hull search in the
        // gradient set) to prepare for the next iteration step
        TVector y = grad - grad_old;
        const Scalar rho = 1.0 / y.dot(s);
        H = H - rho * (s * (y.transpose() * H) + (H * y) * s.transpose())
            + rho * rho * (y.dot(H * y) + 1.0 / rho) * (s * s.transpose());


        ++this->m_current.iterations;
        this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity>();
        // std::cout << "iter: "<<iter<< " f = " <<  objFunc.value(x0) << " ||g||_inf "<<gradNorm   << std::endl;
        this->m_status = checkConvergence(this->m_stop, this->m_current);
      } while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue));
    }
  };

}
/* namespace cppoptlib */

#endif /* BFGSNSSOLVER_H_ */
