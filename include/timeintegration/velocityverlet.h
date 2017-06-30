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
                timeDelta_(0.01){};

        /* this has to be called once before performStep() to initialize the class variables
         * otherwise, they will not have the correct dimensionality due to the dynamic vector size*/
        void initialize(const TVector &x, ProblemType &objFunc) {
          TVector gradient(x.rows());
          objFunc.gradient(x, gradient);

          gradientOld_ = gradient;
          velocitiesOld_ = TVector::Zero(x.rows());
        }

        TVector performStep(const TVector &x, ProblemType &objFunc) {
          TVector gradient(x.rows());

          // Velocity Verlet step 2
          objFunc.gradient(x, gradient);

          // Velocity Verlet step 3 (calculate velocities)
          TVector velocities = velocitiesOld_ + (gradientOld_ + gradient) * (-0.5 * timeDelta_);

          // prepare for next iteration
          gradientOld_ = gradient;
          velocitiesOld_ = velocities;

          // Velocity Verlet step 1 (return shift)
          return velocities * timeDelta_ + gradient * (-0.5 * timeDelta_*timeDelta_);
        }

    private:
        Scalar timeDelta_;
        TVector gradientOld_;
        TVector velocitiesOld_;
    };
} /* namespace cppoptlib */


#endif //AMOLQCGUI_VELOCITYVERLET_H
