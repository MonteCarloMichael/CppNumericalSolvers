// CppNumericalSolver
#ifndef ARMIJOWOLFE_H_
#define ARMIJOWOLFE_H_

#include "../meta.h"

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
      const Scalar c1 = 1e-4;
      const Scalar c2 = 0.5; //value from hanso //value suggested in Numerical Optimiziation book = 0.9;
      Scalar a = 0;
      Scalar b = std::numeric_limits::infinity();
      Scalar alpha = alpha_init; // important to try steplength one first

      // calculate initial values
      const Scalar f_init = objFunc.value(x);

      TVector grad(x.rows());
      objFunc.gradient(x, grad);

      const Scalar searchDirectionProjectedOnGradient_init = grad.dot(searchDir);
      assert(searchDirectionProjectedOnGradient_init >= 0 && "The search direction is not a descent direction.");

      const Scalar sc1 = c1 * searchDirectionProjectedOnGradient_init;
      const Scalar sc2 = c2 * searchDirectionProjectedOnGradient_init;

      // set parameters
      const Scalar searchDirNorm = searchDir.norm();

      const Scalar nbisectmax = std::max(30, std::round(std::log2(1e5 * searchDirNorm))); //more iterations if ||d|| big
      const Scalar nexpandmax = std::max(10,
                                         std::round(std::log2(1e5 / searchDirNorm))); //more iterations if ||d|| small
      Scalar nbisect = 0;
      Scalar nexpand = 0;

      //bool done = false;

      while (true) {
        // evaluate armijo condition (if not <=)
        if (objFunc.value(x + alpha * searchDir) > f_init + alpha * sc1) b = alpha;
          // evaluate weak Wolfe condition (if not >=)
        else if (objFunc.gradient(x + alpha * searchDir, grad).dot(searchDir) < sc2) a = alpha;
        else {
          return alpha;
        }

        // calculate new alpha
        if (b < std::numeric_limits::infinity()) alpha = 0.5 * (a + b);
        else alpha = 2.0 * a;

        /*// calculate new alpha with stop criteria to prevent infinte bracketing
        // calculate new alpha
        if( b < std::numeric_limits::infinity() ) {
          if( nbisect < nbisectmax ) {
            nbisect++;
            alpha = 0.5 * (a+b);
          }
          else {
            assert(false && "Line search failed to bracket point "
                              "satisfying weak Wolfe conditions, "
                              "function may be unbounded below.");
            return alpha_init;
          }
        }
        else {
          if( nexpand < nexpandmax ) {
            nexpand++;
            alpha = 2.0 * a;

          }
          else {
            assert(false && "Line search failed to satisfy weak "
                              "Wolfe conditions although point "
                              "satisfying conditions was bracketed.");
            return alpha_init;
        }
      }*/// calculate new alpha with stop criteria to prevent infinte bracketing
      }
    };
  };
}

#endif /* ARMIJOWOLFE_H_ */
