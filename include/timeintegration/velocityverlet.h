//
// Created by heuer on 19.05.17.
//

#ifndef AMOLQCGUI_VELOCITYVERLET_H
#define AMOLQCGUI_VELOCITYVERLET_H

#include <Eigen/Core>

namespace cppoptlib {

    template<typename ProblemType>
    class VelocityVerlet {
    public:
        using Scalar = typename ProblemType::Scalar;
        using TVector = typename ProblemType::TVector;

        VelocityVerlet():
                timeDelta_(0.1){};

        TVector performStep(const TVector &x, ProblemType &objFunc) {
          TVector grad(x.rows());

          // on first step, gradOld is not empty
          if(gradOld_.size() == 0 ) gradOld_ = grad;

          // Velocity Verlet step 2
          objFunc.gradient(x, grad);

          // Velocity Verlet step 3 (calculate velocities)
          TVector velocity = (gradOld_ + grad) * (-0.5 * timeDelta_);

          // prepare for next iteration
          gradOld_ = grad;

          // Velocity Verlet step 1 (return shift)
          return velocity * timeDelta_ + grad * (-0.5 * timeDelta_);
        }

    private:
        Scalar timeDelta_;
        TVector gradOld_;
    };
} /* namespace cppoptlib */


#endif //AMOLQCGUI_VELOCITYVERLET_H
