// CppNumericalSolver
#include <iostream>
#include <Eigen/LU>
#include "isolver.h"
#include "../linesearch/armijowolfe.h"
#include "../linesearch/smallestvectorinconvexhullfinder.h"

#ifndef BFGSNSSOLVER_H_
#define BFGSNSSOLVER_H_

namespace cppoptlib {

  template<typename ProblemType>
  class BfgsNsSolver : public ISolver<ProblemType, 1> {
  public:
    //static const int Dim = ProblemType::Dim;
    //static const size_t MaxIt = static_cast<size_t>(m_stop.iterations);
    using Superclass = ISolver<ProblemType, 1>;
    using typename Superclass::Scalar;
    using typename Superclass::TVector;
    using typename Superclass::THessian;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    void minimize(ProblemType &objFunc, TVector & x0) {
      const size_t DIM = x0.rows();
      const size_t MaxIt = Superclass::m_stop.iterations+1;

      THessian H = THessian::Identity(DIM, DIM);
      TVector grad(DIM);
      objFunc.gradient(x0, grad);

      MatrixType gradientSet = MatrixType::Zero(DIM,MaxIt); // TODO better define size?
      gradientSet.col(0) = grad;

      TVector x_old = x0;

      // members for nonsmooth opt
      const int J = 50; // J = 1 reduces to the usual stopping condition, when f is smooth
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> j(MaxIt,1);

      const Scalar xTolerance = 1e-4;

      int k;

      this->m_current.reset();
      do {
        k = this->m_current.iterations;

        TVector searchDir = -1 * H * grad;
        // check "positive definite"
        Scalar phi = grad.dot(searchDir);

        // Check if search direction is a descent direction (positive definit)
        if (phi > 0) {
          // no, we reset the hessian approximation
          H = THessian::Identity(DIM, DIM);
          searchDir = -1 * grad;
        }

        // do step
        const Scalar rate = ArmijoWolfe<ProblemType, 1>::linesearch(x0, searchDir, objFunc) ;
        x0 = x0 + rate * searchDir;
        TVector s = rate * searchDir;

        TVector grad_old = grad;
        objFunc.gradient(x0, grad);
        gradientSet.col(k) = grad;

        TVector y = grad - grad_old;

        // prepare next step
        // Update the hessian
        const Scalar rho = 1.0 / y.dot(s);
        H = H - rho * (s * (y.transpose() * H) + (H * y) * s.transpose())
            + rho * rho * (y.dot(H * y) + 1.0 / rho) * (s * s.transpose());

        j(0) = 1;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gradientSetSelection; //TODO CHECK J+1?
        //TODO make a matrix and then resize it

        if ( (x0 - x_old).norm() > xTolerance ){
          j(k) = 1;
          gradientSetSelection.resize(DIM,1);
          gradientSetSelection.col(0) = grad;
        }
        else {
          assert( j(k-1) <= J );
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
        Scalar dknorm;
        if( j(k) > 1 ) {
          SmallestVectorInConvexHullFinder<Scalar>finder; // TODO add max values with MaxIt
          finder.resizeFinder(DIM, gradientSetSelection.rows());
          dknorm = finder.findSmallestVectorInConvexHull(gradientSetSelection).second.norm();
        } else{
          dknorm = grad.norm();
        }

        x_old = x0;

        ++this->m_current.iterations;
        this->m_current.gradNorm = dknorm;
        this->m_status = checkConvergence(this->m_stop, this->m_current);
      } while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue));
    }
  };

}
/* namespace cppoptlib */

#endif /* BFGSNSSOLVER_H_ */
