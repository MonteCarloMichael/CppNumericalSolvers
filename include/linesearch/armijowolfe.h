// CppNumericalSolver
#ifndef ARMIJOWOLFE_H_
#define ARMIJOWOLFE_H_

#include "../meta.h"
#include <cmath>

namespace cppoptlib {

  template<typename ProblemType, int Ord>
  class ArmijoWolfe {
  public:
    using Scalar = typename ProblemType::Scalar;
    using TVector = typename ProblemType::TVector;

    /**
     * @brief use Armijo Rule for (weak) Wolfe conditiions
     * @details [long description]
     *
     * @param searchDir search direction for next update step
     * @param objFunc handle to problem
     *
     * @return step-width
     */
    static Scalar linesearch(const TVector &x,
                             const TVector &searchDir,
                             ProblemType &objFunc,
                             const Scalar alpha_init = 1.0) {

      assert(searchDir.norm() != 0);

      const Scalar c1 = 0;//value from hanso 1e-4;
      const Scalar c2 = 0.5; //value from hanso //value suggested in Numerical Optimiziation book = 0.9;
      Scalar a = 0;
      Scalar b = std::numeric_limits<Scalar>::infinity();
      Scalar alpha = alpha_init; // important to try steplength one first

      // calculate initial values
      const Scalar f_init = objFunc.value(x);
      Scalar f = f_init;
      TVector grad(x.rows());

      objFunc.gradient(x, grad);

      const Scalar searchDirectionProjectedOnGradient_init = grad.dot(searchDir);
      Scalar projectedGradAtAlpha;

      const Scalar sc1 = c1 * searchDirectionProjectedOnGradient_init;
      const Scalar sc2 = c2 * searchDirectionProjectedOnGradient_init;

      // set parameters
      const Scalar searchDirNorm = searchDir.norm();

      const int nbisectmax = std::max(30, static_cast<int>(std::round(std::log2(1e5 * searchDirNorm))));
      //more iterations if ||d|| is big
      const int nexpandmax = std::max(10, static_cast<int>(std::round(std::log2(1e5 / searchDirNorm))));
      //more iterations if ||d|| small
      int nbisect = 0;
      int nexpand = 0;

      bool failed = false;

      while (!failed) {
        f = objFunc.value(x + alpha * searchDir);
        objFunc.gradient(x + alpha * searchDir, grad);
        projectedGradAtAlpha = grad.dot(searchDir);

        // evaluate armijo condition (if not <=)
        if ((f > f_init + alpha * sc1)
            || (f == std::numeric_limits<Scalar>::quiet_NaN()) ) b = alpha;
          // evaluate weak Wolfe condition (if not >=)
        else if ((projectedGradAtAlpha < sc2)
                 || (projectedGradAtAlpha == std::numeric_limits<Scalar>::quiet_NaN()) ) a = alpha;
        else {
          return alpha;
        }

      // calculate new alpha with stop criteria to prevent infinte bracketing
        if (b < std::numeric_limits<Scalar>::infinity()) {
          if (nbisect < nbisectmax) {
            nbisect++;
            alpha = 0.5 * (a + b);
          } else failed = true;
        } else {
          if (nexpand < nexpandmax) {
            nexpand++;
            alpha = 2.0 * a;
          } else failed = true;
        }
        // simple method to calculate new alpha
        /*if (b < std::numeric_limits<Scalar>::infinity()) alpha = static_cast<Scalar>(0.5) * (a + b);
          else alpha = static_cast<Scalar>(2.0) * a;*/
      }

      //TODO add as status
      //if ( b == std::numeric_limits<Scalar>::infinity()) {
      //  std::cout << "failed: minimizer never bracketed. "
      //    "Function maybe unbounded below." << std::endl;
      //}
      //else {
      //  std::cout << "failed: point satisfying Wolfe conditions was bracketed "
      //    "but weak wolfe condition was not satisfied" << std::endl;
      //}

      return alpha;
    };
  };
}

#endif /* ARMIJOWOLFE_H_ */
