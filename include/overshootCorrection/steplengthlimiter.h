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

        static void limitStepLength(TVector & step,Scalar maximalStepLength=1.0){
            Scalar currentStepLength = step.norm();
            Scalar limitedStepLength = std::min(maximalStepLength,currentStepLength);

            if (currentStepLength!=0){
                step *= limitedStepLength/currentStepLength;
            }
        };

        // TODO ich verstehe nicht, warum deine klasse so kompliziert ist.
        StepLengthLimiter(Scalar maximalStepLengthValue = 1.0, Scalar rateValue = 1e-4):
            maximalStepLength(maximalStepLengthValue),
            rate(rateValue){};


        TVector limitStep(const TVector &gradientCurrent) {
            TVector stepLength =  - gradientCurrent*rate;
            Scalar lambda = std::min(maximalStepLength,stepLength.norm());

            //preventing nan as result
            if (stepLength.norm()==0){
                return stepLength;
            }

            TVector stepLengthNew =  lambda*stepLength/stepLength.norm();
            return stepLengthNew;
        }

        void setMaximalStepLength(Scalar maximalStepLengthValue){
            assert(maximalStepLengthValue > 0);
            maximalStepLength = maximalStepLengthValue;
        }

        void setRate(Scalar rateValue){
            assert(rateValue > 0);
            rate = rateValue;
        }

    private:
        Scalar rate;
        Scalar maximalStepLength;
    };
} /* namespace cppoptlib */

#endif //AMOLQCGUI_STEPLENGTHLIMITER_H
