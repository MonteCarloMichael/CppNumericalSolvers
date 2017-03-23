// CppNumericalSolver
//
// Created by Michael Heuer on 06.02.17.
//
#include <iostream>
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

    BfgsnsSolver() : ISolver<ProblemType,1>(TCriteria::nonsmoothDefaults()){};

    void minimize(ProblemType &objFunc, TVector & x0) {
      const size_t DIM = x0.rows();
      const size_t MaxIt = Superclass::m_stop.iterations+1;

      THessian H = THessian::Identity(DIM, DIM);
      TVector grad(DIM);
      objFunc.gradient(x0, grad);

      MatrixType gradientSet = MatrixType::Zero(DIM,MaxIt);
      gradientSet.col(0) = grad;

      TVector x_old = x0;

      // members for nonsmooth opt
      const int J = 50; // J = 1 reduces to the usual stopping condition, when f is smooth
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> j(MaxIt,1);
      j(0) = 1;

      int k;
      this->m_current.reset();
      do {
        k = this->m_current.iterations;
        TVector searchDir = -1 * H * grad;
        // check "positive definite"
        Scalar phi = grad.dot(searchDir);

        // Check if search direction is a descent direction
        if (phi > 0) {
          // no, we reset the hessian approximation
          H = THessian::Identity(DIM, DIM);
          searchDir = -grad;
        }

        // do step
        //TODO give linesearch the initial gradient
        const Scalar stepLength = ArmijoWolfe<ProblemType, 1>::linesearch(x0, searchDir, //grad,
                                                                    objFunc);
        TVector step = stepLength * searchDir;
        x0 = x0 + step;
        this->m_current.xDelta = (x_old - x0).norm();


        TVector grad_old = grad;
        objFunc.gradient(x0, grad);
        gradientSet.col(k) = grad;

        TVector y = grad - grad_old;

        // prepare next step
        // Update the hessian
        const Scalar rho = 1.0 / y.dot(step);
        H = H - rho * (step * (y.transpose() * H) + (H * y) * step.transpose())
            + rho * rho * (y.dot(H * y) + 1.0 / rho) * (step * step.transpose());

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gradientSetSelection;

        //if ( (x0 - x_old).norm() > xTolerance){
        if ( this->m_current.xDelta > this->m_stop.xDelta ){
          j(k) = 1;
          gradientSetSelection.resize(DIM,1);
          gradientSetSelection.col(0) = grad;
        }
        else { //TODO CHECK FOR k > 0
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
          SmallestVectorInConvexHullFinder<Scalar> finder; // TODO add max values with MaxIt
          finder.resizeFinder(DIM, gradientSetSelection.rows());
          grad = finder.findSmallestVectorInConvexHull(gradientSetSelection).second;
        }

        x_old = x0;

        ++this->m_current.iterations;
        this->m_current.gradNorm = grad.norm(); //TODO calculate only for smooth variables, HOW TO IDENTIFY THEM?
        this->m_status = checkConvergence(this->m_stop, this->m_current);
      } while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue));
    }
  };

}
/* namespace cppoptlib */

#endif /* BFGSNSSOLVER_H_ */
