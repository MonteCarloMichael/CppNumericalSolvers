//
// Created by heuer on 19.05.17.
//

#ifndef AMOLQCGUI_TIMEINTEGRATIONSOLVER_H
#define AMOLQCGUI_TIMEINTEGRATIONSOLVER_H

#include "isolver.h"
#include "../timeintegration/firealgorithm.h"
#include "../timeintegration/velocityverlet.h"

namespace cppoptlib {

    template<typename ProblemType>
    class TimeIntegrationSolver : public ISolver<ProblemType, 1> {

    public:
        using Superclass = ISolver<ProblemType, 1>;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        void minimize(ProblemType &objFunc, TVector &x0) {

          TVector grad(x0.rows());
          TVector x_old(x0.rows());
          this->m_current.reset();

          fireAlgorithm.initialize(x0, objFunc);

          objFunc.gradient(x0,grad);
          do {
            TVector s = fireAlgorithm.performStep(x0, objFunc);
            x_old = x0;
            x0 = x0 + s;

            objFunc.gradient(x0,grad);

            this->m_current.xDelta = (x_old-x0).template lpNorm<Eigen::Infinity>();
            this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity>();
            ++this->m_current.iterations;
            this->m_status = checkConvergence(this->m_stop, this->m_current);
          } while (objFunc.callback(this->m_current, x0) && (this->m_status == Status::Continue));

          if (this->m_debug > DebugLevel::None) {
            std::cout << "Stop status was: " << this->m_status << std::endl;
            std::cout << "Stop criteria were: " << std::endl << this->m_stop << std::endl;
            std::cout << "Current values are: " << std::endl << this->m_current << std::endl;
          }
        }

    private:
        FIREAlgorithm<ProblemType> fireAlgorithm;
    };

} /* namespace cppoptlib */

#endif //AMOLQCGUI_TIMEINTEGRATIONSOLVER_H
