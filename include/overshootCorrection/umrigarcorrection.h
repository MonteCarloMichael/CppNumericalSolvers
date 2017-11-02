//
// Created by dahl on 25.10.17.
//

#ifndef AMOLQCGUI_UMRIGARCORRECTION_H
#define AMOLQCGUI_UMRIGARCORRECTION_H

#include <Eigen/Core>

namespace cppoptlib {

    template<typename ProblemType>
    class UmrigarCorrector {
    public:
        using Scalar = typename ProblemType::Scalar;
        using TVector = typename ProblemType::TVector;

        /* this has to be called once before correctStep() to initialize the class variables
         * otherwise, they will not have the correct dimensionality due to the dynamic vector size*/
        UmrigarCorrector(){};

        void initialize(const TVector &xElectron, const TVector &xNucleus3d) {
            //TVector minDistance(xEl.rows()*xNuc.rows());
            //TVector electronList(xNuc.rows());
            //ggf mehrere Vektoren--> Für jeden Fall ein Vektor?
        }


        int nearestElectronNumberReturn(const TVector &xElectron, const TVector &xNucleus3d){

            //variables for loops
            int i,j,k;

            int nearestElectronNumber;
            //Besser int?
            double xElectronDimension = xElectron.rows();
            TVector xNearestElectron3d(3), xElectronReference3d(3);
            Scalar smallestDistance, distanceReference;

            for (k=0; k<3; ++k){
                xNearestElectron3d(k)=xElectron(k);
            }

            smallestDistance = (xNearestElectron3d-xNucleus3d).norm();

            //search nearest electron and save correspondent electron number
            if (xElectronDimension>3){
                for (i=2; i<=(xElectronDimension/3); ++i){
                    for (j=(3*i-3); j<(3*i); ++j){
                        xElectronReference3d(j-(3*i-3))=xElectron(j);
                    }
                    distanceReference = (xElectronReference3d-xNucleus3d).norm();
                    if (distanceReference<smallestDistance){
                        smallestDistance = distanceReference;
                        //xNearestElectron3d = xElectronReference3d;
                        nearestElectronNumber = i;
                    }
                }
            }
            return nearestElectronNumber;
        }


        TVector correctStepMod(const TVector &xElectron3d, const TVector &xNucleus3d, const TVector &gradient3d, const TVector &stepLength3d) {

            TVector velocityElectron = -gradient3d;
            TVector distance = xElectron3d- xNucleus3d;
            TVector distanceNormalized = distance/(distance.norm());

            Scalar vz = velocityElectron.dot(distanceNormalized);

            TVector orthogonalToDistance = velocityElectron-(vz*distanceNormalized);
            TVector orthogonalToDistanceNormalized = orthogonalToDistance/(orthogonalToDistance.norm());
            Scalar orthogonalToDistanceNorm = (orthogonalToDistance).norm();
            
            Scalar rate = stepLength3d(1)/velocityElectron(1);
            
            //Parameters to calculate the new position
            Scalar zNew = std::max(distance.norm()+vz*rate,0.0);
            Scalar rhoNew = 2*orthogonalToDistanceNorm*rate*zNew/(distance.norm()+zNew);

            TVector xElectron3dNew = xNucleus3d + rhoNew*orthogonalToDistanceNormalized + zNew*distanceNormalized;
            return xElectron3dNew;
        }

        TVector correctStepCut(const TVector &xElectron3d, const TVector &xNucleus3d, const TVector &gradient3d, const TVector &stepLength3d){

            TVector distance = xElectron3d- xNucleus3d;
            TVector velocityElectronNormalized = (-gradient3d)/(-gradient3d).norm();

            //orthogonal projection of distVec on to velocityElNormalized, when the length of StepLength3D is bigger
            TVector orthogonalProjection = -distance.dot(velocityElectronNormalized)*velocityElectronNormalized;
            if (std::min(stepLength3d.norm(),(-distance.dot(velocityElectronNormalized)*velocityElectronNormalized).norm())==stepLength3d.norm()){
                orthogonalProjection = stepLength3d;
            }

            TVector xElectron3dNew = xElectron3d + orthogonalProjection;
            return xElectron3dNew;
        }

    private:

    };
} /* namespace cppoptlib */

#endif //AMOLQCGUI_UMRIGARCORRECTION_H
