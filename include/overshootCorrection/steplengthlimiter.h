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
    };
} /* namespace cppoptlib */

#endif //AMOLQCGUI_STEPLENGTHLIMITER_H
