// CppNumericalSolver
#ifndef GRADIENTDESCENTSIMPLESOLVER_H_
#define GRADIENTDESCENTSIMPLESOLVER_H_

#include <Eigen/Core>
#include "isolver.h"
#include "../linesearch/morethuente.h"

namespace cppoptlib {

template<typename ProblemType>
class GradientDescentSimpleSolver : public ISolver<ProblemType, 1> {

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

    TVector direction(x0.rows());
    this->m_current.reset();
    do {
      ;
      objFunc.gradient(x0, direction);
      const Scalar rate = 1e-4;
      x0 = x0 - rate * direction;
      this->m_current.gradNorm = direction.template lpNorm<Eigen::Infinity>();
      // std::cout << "iter: "<<iter<< " f = " <<  objFunc.value(x0) << " ||g||_inf "<<gradNorm  << std::endl;
      ++this->m_current.iterations;
      this->m_status = checkConvergence(this->m_stop, this->m_current);
    } while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue));
    if (this->m_debug > DebugLevel::None) {
        std::cout << "Stop status was: " << this->m_status << std::endl;
        std::cout << "Stop criteria were: " << std::endl << this->m_stop << std::endl;
        std::cout << "Current values are: " << std::endl << this->m_current << std::endl;
    }
  }

};

} /* namespace cppoptlib */

#endif /* GRADIENTDESCENTSIMPLESOLVER_H_ */
