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

      SmallestVectorInConvexHullFinder<Scalar>finder; // TODO add max values with MaxIt


      THessian H = THessian::Identity(DIM, DIM);
      TVector grad(DIM);
      objFunc.gradient(x0, grad);

      MatrixType gradientSet = MatrixType::Zero(MaxIt,DIM); // TODO better define size?
      gradientSet.row(0) = grad;

      TVector x_old = x0;

      // members for nonsmooth opt
      const int J = 50; // J = 1 reduces to the usual stopping condition, when f is smooth
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> j(MaxIt,1);

      const Scalar xTolerance = 1e-4;
      //const Scalar dTolerance = 1e-4;

      int k;

      this->m_current.reset();
      do {
        k = this->m_current.iterations;

        TVector searchDir = -1 * H * grad;
        // check "positive definite"
        Scalar phi = grad.dot(searchDir);

        // positive definit ?
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
        gradientSet.row(k) = grad;

        TVector y = grad - grad_old;

        // prepare next step
        // Update the hessian
        const Scalar rho = 1.0 / y.dot(s);
        H = H - rho * (s * (y.transpose() * H) + (H * y) * s.transpose())
            + rho * rho * (y.dot(H * y) + 1.0 / rho) * (s * s.transpose());
        // std::cout << "iter: "<<iter<< " f = " <<  objFunc.value(x0) << " ||g||_inf "<<gradNorm   << std::endl;


        // original BFGS stopping condition
        //if( (x_old-x0).template lpNorm<Eigen::Infinity>() < 1e-7  ) break;

        // ---------- MY PART --------//TODO
        // stopping condition for non-smooth optimization
        j(0) = 1;

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,0,J> gradientSetSelection; //TODO CHECK J+1?

        // TODO make a matrix and then resize it

        if ( (x0 - x_old).norm() > xTolerance ){
          j(k) = 1;
          gradientSetSelection.resize(1,DIM);//gradientSetSelection = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Zero(1,DIM);
          gradientSetSelection.row(0) = grad;
        }
        else {
          assert( j(k-1) <= J );
          // check second last element
          if ( j(k-1) < J ) {

            j(k) = j(k-1) + 1;
            gradientSetSelection.resize(j(k),DIM);
            gradientSetSelection = gradientSet.block(k-j(k)+1,0,j(k),DIM);// jk elements //k - ( k-j(k)+1 ) +1
          }
          else { // j(k-1) == J)
            j(k) = J;
            gradientSetSelection.resize(J,DIM);
            gradientSetSelection = gradientSet.block( k-J+1, 0, J, DIM); // J elements
          }
        }
        /*TODO Problem: during compile-time the size of gradientSetSelection is not known*/

        finder.resizeFinder(DIM,gradientSetSelection.rows());
        TVector dk = finder.findSmallestVectorInConvexHull(gradientSetSelection);



        // ---------- MY PART END --------//TODO

        x_old = x0;

        ++this->m_current.iterations;
        this->m_current.gradNorm = dk.norm();//grad.template lpNorm<Eigen::Infinity>();
        this->m_status = checkConvergence(this->m_stop, this->m_current);
      } while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue));
    }
  };

}
/* namespace cppoptlib */

#endif /* BFGSNSSOLVER_H_ */
