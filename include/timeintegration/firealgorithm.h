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

        TVector performStep(const TVector &x, ProblemType &objFunc) {
          TVector grad(x.rows());

          // Velocity Verlet step 2
          objFunc.gradient(x, grad);

          // on first step, gradOld is not empty
          if(gradOld_.size() == 0 ) {
            gradOld_ = grad;
            //gradOld_ = Eigen::VectorXd::Zero(grad.size());
          } 


          // Velocity Verlet step 3 (calculate velocities)
          TVector velocity = (gradOld_ + grad) * (-0.5 * timeDelta_);

          // FIRE step 1
          projection_ = grad.dot(velocity*(-1));

          // FIRE step 2
          velocity = velocity * (1 - alpha_);
          velocity += grad.normalized() * (-1.0 * velocity.norm() * alpha_);

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
            velocity.setZero();
            alpha_ = alphaStart_;
          }
          // prepare for next iteration
          gradOld_ = grad;

          // Velocity Verlet step 1 (return shift)
          return velocity * timeDelta_ + grad * (-0.5 * timeDelta_);
        }

    private:
        Scalar timeDelta_;
        TVector gradOld_;

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
