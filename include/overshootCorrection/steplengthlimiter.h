//
// Created by dahl on 25.10.17.
//

#ifndef AMOLQCGUI_STEPLENGTHLIMITER_H
#define AMOLQCGUI_STEPLENGTHLIMITER_H

#include <Eigen/Core>

namespace cppoptlib {

    template<typename ProblemType>
    class StepLengthLimiter {
    public:
        using Scalar = typename ProblemType::Scalar;
        using TVector = typename ProblemType::TVector;

        /* this has to be called once before correctStep() to initialize the class variables
         * otherwise, they will not have the correct dimensionality due to the dynamic vector size*/
        StepLengthLimiter(Scalar maximalStepLengthValue = 1.0):
            maximalStepLength(maximalStepLengthValue),
            rate(1e-4){};

        TVector limitStep(const TVector &gradientCurrent) {
            TVector stepLength =  - gradientCurrent*rate;
            Scalar lambda = std::min(maximalStepLength,gradientCurrent.norm());

            TVector stepLengthNew =  lambda*stepLength/stepLength.norm();
            return stepLengthNew;
        }

        void setMaximalStepLength(Scalar maximalStepLengthValue){
            assert(maximalStepLengthValue > 0);
            maximalStepLength = maximalStepLengthValue;
        }

    private:
        Scalar rate;
        Scalar maximalStepLength;
    };
} /* namespace cppoptlib */

#endif //AMOLQCGUI_STEPLENGTHLIMITER_H
