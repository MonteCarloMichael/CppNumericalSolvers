//
// Created by dahl on 25.10.17.
//

#ifndef AMOLQCGUI_GRADIENTDESCENTUMRIGARLIMITEDSTEPLENGTH_H
#define AMOLQCGUI_GRADIENTDESCENTUMRIGARLIMITEDSTEPLENGTH_H

#include <Eigen/Core>
#include "isolver.h"
#include "overshootCorrection/umrigarcorrection.h"
#include "overshootCorrection/steplengthlimiter.h"

using namespace Eigen;

namespace cppoptlib {

    template<typename ProblemType>
    class GradientDescentUmrigarLimitedSteplength : public ISolver<ProblemType, 1> {

    public:
        using Superclass = ISolver<ProblemType, 1>;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        /**
         * @brief minimize
         * @details [long description]
         *
         * @param objFunc [description]
         */

        void setMaxStepLength(Scalar maxStepLengthValue){
            stepLengthLimiter.setMaximalStepLength(maxStepLengthValue);
        }

        void setSteepestDescentRate(Scalar rateValue){
            stepLengthLimiter.setRate(rateValue);
        }

        void setDistanceCriteriaUmrigar (Scalar distanceCriteriaUmrigarValue){
            distanceCriteriaUmrigar = distanceCriteriaUmrigarValue;
        }

        void setThreshholdUmrigar (Scalar threshholdUmrigarValue){
            threshholdUmrigar = threshholdUmrigarValue;
        }

        void minimize(ProblemType &objFunc, TVector &electronsPositions0) {

            UmrigarCorrector<ProblemType> umrigarCorrector(electronsPositions0.rows()/3, threshholdUmrigar);

            Eigen::VectorXd nucleiPositions = objFunc.getNucleiPositions();

            // TODO warum nennst du den Gradienten direction?
            TVector direction(electronsPositions0.rows());
            TVector electronsPositionsOld(electronsPositions0.rows());
            this->m_current.reset();

            objFunc.gradient(electronsPositions0, direction);

            do {
                TVector stepLengthCurrent = stepLengthLimiter.limitStep(direction);

                //steepest descent step with adaptive step length
                electronsPositionsOld = electronsPositions0;
                electronsPositions0 = electronsPositions0 + stepLengthCurrent;

                electronsPositions0 = umrigarCorrector.correctStep(electronsPositionsOld,
                                                                   electronsPositions0,
                                                                   direction,
                                                                   stepLengthCurrent,
                                                                   nucleiPositions,
                                                                   distanceCriteriaUmrigar);

                objFunc.gradient(electronsPositions0, direction);

                std::vector<unsigned long> indicesOfElectronsNotAtCores = umrigarCorrector.getIndicesOfElectronsNotAtCores();
                //if there are electrons at the nucleus set the corresponding gradient to 0.0
                for(unsigned long electronIndex=0; electronIndex<(electronsPositions0.rows()/3); ++electronIndex) {
                    if(!(std::find(indicesOfElectronsNotAtCores.begin(), indicesOfElectronsNotAtCores.end(), electronIndex) != indicesOfElectronsNotAtCores.end())){
                        direction.segment(electronIndex*3,3)=Vector3d(0.0,0.0,0.0);
                    }
                }

                //std::cout << electronsPositions0 << std::endl;

                this->m_current.xDelta = (electronsPositionsOld-electronsPositions0).template lpNorm<Eigen::Infinity>();
                this->m_current.gradNorm = direction.template lpNorm<Eigen::Infinity>();
                ++this->m_current.iterations;
                this->m_status = checkConvergence(this->m_stop, this->m_current);
            } while (objFunc.callback(this->m_current, electronsPositions0) && (this->m_status == Status::Continue));
            if (this->m_debug > DebugLevel::None) {
                std::cout << "Stop status was: " << this->m_status << std::endl;
                std::cout << "Stop criteria were: " << std::endl << this->m_stop << std::endl;
                std::cout << "Current values are: " << std::endl << this->m_current << std::endl;
            }
        }

    private:
        StepLengthLimiter<ProblemType> stepLengthLimiter;
        Scalar distanceCriteriaUmrigar;
        Scalar threshholdUmrigar;
    };



} /* namespace cppoptlib */


#endif //AMOLQCGUI_GRADIENTDESCENTUMRIGARLIMITEDSTEPLENGTH_H
