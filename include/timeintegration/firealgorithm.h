//
// Created by heuer on 19.05.17.
//

#ifndef AMOLQCGUI_FIREALGORITHM_H
#define AMOLQCGUI_FIREALGORITHM_H

#include <Eigen/Core>

namespace cppoptlib {

    template<typename ProblemType>
    class FIREAlgorithm {
    public:
        using Scalar = typename ProblemType::Scalar;
        using TVector = typename ProblemType::TVector;

        /* this has to be called once before performStep() to initialize the class variables
         * otherwise, the will not have the correct dimensionality due to the dynamic vector size*/
        FIREAlgorithm()
                :timeDelta_(0.01),
                 timeDeltaMax_(timeDelta_*10),
                 nMin_(5),
                 fInc_(1.1),
                 fDec_(0.5),
                 alphaStart_(0.1),
                 fAlpha_(0.99){

          alpha_ = alphaStart_;
          latencyCount_= 0;
        };

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

          // FIRE step 1
          projection_ = gradient.dot(velocities * -1.0);

          // FIRE step 2
          velocities *= (1 - alpha_);
          velocities += gradient.normalized() * (-1.0 * velocities.norm() * alpha_);

          // FIRE step 3
          ++latencyCount_; // increase counter for latency of the step size
          if ((projection_ > 0) && (latencyCount_ > nMin_)) { // going downhill
            timeDelta_ = std::min(timeDelta_ * fInc_, timeDeltaMax_);
            alpha_ = alpha_ * fAlpha_;

            latencyCount_ = 0;
          }
            // FIRE step 4
          else if (projection_ <= 0) { // going uphill
            timeDelta_ = timeDelta_ * fDec_;
            velocities.setZero();
            alpha_ = alphaStart_;
          }
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

        // FIRE variables
        unsigned latencyCount_;
        Scalar timeDeltaMax_;
        Scalar alpha_;
        Scalar projection_;

        // FIRE parameters
        unsigned nMin_;
        Scalar fInc_;
        Scalar fDec_;
        Scalar alphaStart_;
        Scalar fAlpha_;
    };
} /* namespace cppoptlib */

#endif //AMOLQCGUI_FIREALGORITHM_H
