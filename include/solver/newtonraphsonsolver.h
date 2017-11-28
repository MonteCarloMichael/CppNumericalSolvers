// CppNumericalSolver
#ifndef NEWTONRAPHSONSOLVER_H_
#define NEWTONRAPHSONSOLVER_H_

#include <Eigen/LU>
#include "isolver.h"
#include "../linesearch/morethuente.h"

namespace cppoptlib {

  template<typename ProblemType>
  class NewtonRaphsonSolver : public ISolver<ProblemType, 2> {
  public:
    using Superclass = ISolver<ProblemType, 2>;
    using typename Superclass::Scalar;
    using typename Superclass::TVector;
    using typename Superclass::THessian;

    void minimize(ProblemType &objFunc, TVector &x0) {
      const int DIM = x0.rows();
      TVector grad = TVector::Zero(DIM);
      THessian hessian = THessian::Zero(DIM, DIM);

      this->m_current.reset();
      do {
        objFunc.gradient(x0, grad);
        objFunc.hessian(x0, hessian);
        TVector searchDir = hessian.fullPivLu().solve(-grad);

        const Scalar rate = MoreThuente<ProblemType, 1>::linesearch(x0, searchDir, objFunc) ;
        TVector delta_x = rate * searchDir;
        x0 = x0 + delta_x;

        ++this->m_current.iterations;
        this->m_current.xDelta = delta_x.template lpNorm<Eigen::Infinity>();
        this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity>();
        this->m_status = checkConvergence(this->m_stop, this->m_current);
      } while (objFunc.callback(this->m_current, x0, grad) && (this->m_status == Status::Continue));
    }
  };

}
/* namespace cppoptlib */

#endif //NEWTONRAPHSONSOLVER_H_
