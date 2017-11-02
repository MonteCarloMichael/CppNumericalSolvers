//
// Created by dahl on 25.10.17.
//

#ifndef AMOLQCGUI_GRADIENTDESCENTUMRIGARLIMITEDSTEPLENGTH_H
#define AMOLQCGUI_GRADIENTDESCENTUMRIGARLIMITEDSTEPLENGTH_H

#include <Eigen/Core>
#include "isolver.h"
#include "overshootCorrection/umrigarcorrection.h"
#include "overshootCorrection/steplengthlimiter.h"


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

        void minimize(ProblemType &objFunc, TVector &electronPosition0) {

            TVector direction(electronPosition0.rows());
            TVector electronPositionOld(electronPosition0.rows());
            this->m_current.reset();

            objFunc.gradient(electronPosition0, direction);
            do {
                // überprüfe, ob ein elektron in kritischer nähe zum kern ist
                    // wenn ja, dann umrigar korrektur für dieses elektron
                    // wenn nein, dann normalen steepest descent schritt
                electronPositionOld = electronPosition0;
                electronPosition0 = electronPosition0 + stepLengthLimiter.limitStep(direction);



                objFunc.gradient(electronPosition0, direction);

                this->m_current.xDelta = (electronPositionOld-electronPosition0).template lpNorm<Eigen::Infinity>();
                this->m_current.gradNorm = direction.template lpNorm<Eigen::Infinity>();
                ++this->m_current.iterations;
                this->m_status = checkConvergence(this->m_stop, this->m_current);
            } while (objFunc.callback(this->m_current, electronPosition0) && (this->m_status == Status::Continue));
            if (this->m_debug > DebugLevel::None) {
                std::cout << "Stop status was: " << this->m_status << std::endl;
                std::cout << "Stop criteria were: " << std::endl << this->m_stop << std::endl;
                std::cout << "Current values are: " << std::endl << this->m_current << std::endl;
            }
        }

        void correctUmrigar(ProblemType &objFunc, TVector &electronPosition0, TVector &nucleusPosition) {

            //variables for loops
            int l,n,nucleusNumber;
            //Besser int?
            double NucleusPositionDimension = nucleusPosition.rows();

            //TVector NICHT der richtige Datentyp!!!
            TVector direction3d(3),stepLengthCurrent3d(3),nearestElectron3d(3), nucleusPosition3d(3);

            TVector direction(electronPosition0.rows());
            TVector electronPositionOld(electronPosition0.rows());
            this->m_current.reset();

            objFunc.gradient(electronPosition0, direction);

            do {

                /*for (int i = 0; i < NucleusPositionDimension/3; ++i) {
                    Eigen::Matrix<Scalar,3,1> nucleusPosition3d = nucleusPosition.segment(i*3,3);
                }*/

                //correct step for each nucleus
                for(nucleusNumber=1; nucleusNumber<=(NucleusPositionDimension/3); ++nucleusNumber){

                    //save 3d coordinates of treated nucleus
                    for(n=3*nucleusNumber-3;n<3*nucleusNumber; ++nucleusNumber){
                        nucleusPosition3d(n-(3*nucleusNumber-3)) = nucleusPosition(n);

                    }

                    int nearestElectronNumber = umrigarCorrector.nearestElectronNumberReturn(electronPosition0,nucleusPosition3d);

                    electronPositionOld = electronPosition0;
                    TVector stepLengthCurrent = stepLengthLimiter.limitStep(direction);

                    //save information of the nearest electron
                    for(l=3*nearestElectronNumber-3;l<3*nearestElectronNumber; ++l){
                        direction3d(l-(3*nearestElectronNumber-3)) = direction(l);
                        stepLengthCurrent3d(l-(3*nearestElectronNumber-3))=stepLengthCurrent3d(l);
                        nearestElectron3d(l-(3*nearestElectronNumber-3)) = electronPosition0(l);
                    }

                    //correct step in 3d coordinate
                    nearestElectron3d = umrigarCorrector.correctStepMod(nearestElectron3d,nucleusPosition3d,direction3d,stepLengthCurrent3d);

                    //correct step in 3n dimensional coordinate
                    for(l=3*nearestElectronNumber-3;l<3*nearestElectronNumber; ++l){
                        electronPosition0(l) = nearestElectron3d(l-(3*nearestElectronNumber-3));
                    }

                    objFunc.gradient(electronPosition0, direction);

                }

                this->m_current.xDelta = (electronPositionOld-electronPosition0).template lpNorm<Eigen::Infinity>();
                this->m_current.gradNorm = direction.template lpNorm<Eigen::Infinity>();
                ++this->m_current.iterations;
                this->m_status = checkConvergence(this->m_stop, this->m_current);
            } while (objFunc.callback(this->m_current, electronPosition0) && (this->m_status == Status::Continue));
            if (this->m_debug > DebugLevel::None) {
                std::cout << "Stop status was: " << this->m_status << std::endl;
                std::cout << "Stop criteria were: " << std::endl << this->m_stop << std::endl;
                std::cout << "Current values are: " << std::endl << this->m_current << std::endl;
            }
        }

    private:
        UmrigarCorrector<ProblemType> umrigarCorrector;
        StepLengthLimiter<ProblemType> stepLengthLimiter;
    };



} /* namespace cppoptlib */


#endif //AMOLQCGUI_GRADIENTDESCENTUMRIGARLIMITEDSTEPLENGTH_H
