//
// Created by dahl on 25.10.17.
//

#ifndef AMOLQCGUI_UMRIGARCORRECTION_H
#define AMOLQCGUI_UMRIGARCORRECTION_H

#include <Eigen/Core>

using namespace Eigen;

namespace cppoptlib {

    template<typename ProblemType>
    class UmrigarCorrector {
    public:
        using Scalar = typename ProblemType::Scalar;
        using TVector = typename ProblemType::TVector;

        UmrigarCorrector(unsigned long numberOfElectrons, Scalar threshhold=1e-5)
                : numberOfElectrons(numberOfElectrons),
                  threshhold(threshhold),
                  indicesOfElectronsNotAtCores()
                {
                    for(unsigned long i = 0; i < numberOfElectrons; i++ )
                        indicesOfElectronsNotAtCores.push_back(i);
                };

        VectorXd correctStep(TVector &electronsPositions,
                                  TVector &electronsPositionsChanged,
                                  TVector &direction,
                                  TVector &stepLengthCurrent,
                                  const TVector &nucleiPositions,
                                  Scalar distanceCriteriaUmrigar){

            assert(nucleiPositions.rows()%3 == 0 && "Vector dimension must be divideable by 3");

            Vector3d directionNearestElectron,stepLengthCurrentNearestElectron, nearestElectron, nucleusPosition;

            for(unsigned long nucleusIndex=0; nucleusIndex<(nucleiPositions.rows()/3); ++nucleusIndex) {

                //save 3d coordinates of treated nucleus
                nucleusPosition = nucleiPositions.segment(nucleusIndex * 3, 3);

                unsigned long nearestElectronIndex = nearestElectronIndexReturn(electronsPositions,
                                                                                nucleusPosition);

                Scalar nearestElectronDistance = nearestElectronDistanceReturn();

                /*check, if an electron has a critical distance to a nucleus
                if so, perform umrigar and overwrite the calculated steepest descent
                coordinates of the regarding electrons*/
                if (nearestElectronDistance <= distanceCriteriaUmrigar) {

                    //save information of the nearest electron
                    //TODO directionNearestElectron ist ein schlechter Name, weil man nicht weiß, wovon es die Direction ist
                    directionNearestElectron = direction.segment(nearestElectronIndex * 3, 3);
                    stepLengthCurrentNearestElectron = stepLengthCurrent.segment(nearestElectronIndex * 3, 3);
                    nearestElectron = electronsPositions.segment(nearestElectronIndex * 3, 3);

                    //correct step in 3d coordinate
                    nearestElectron = correctStepMod(nearestElectron,
                                                       nucleusPosition,
                                                       directionNearestElectron,
                                                       stepLengthCurrentNearestElectron,
                                                       nearestElectronIndex);

                    //correct step in 3n dimensional coordinate
                    electronsPositionsChanged.segment(nearestElectronIndex * 3, 3) = nearestElectron;
                }
            }
            return electronsPositionsChanged;
        }

        unsigned long nearestElectronIndexReturn(const TVector &electronsPositions,
                                                 const Vector3d &nucleusPosition3d){

            unsigned long nearestElectronIndex=indicesOfElectronsNotAtCores.front();
            assert(electronsPositions.rows() == numberOfElectrons*3);
            Vector3d nearestElectron(3), electronReference(3);
            Scalar distanceReference;

            nearestElectron = electronsPositions.segment(indicesOfElectronsNotAtCores.front()*3,3);

            smallestDistance = (nearestElectron-nucleusPosition3d).norm();

            //search nearest electron and save correspondent electron number
            for(const auto& electronIndex : indicesOfElectronsNotAtCores){
                electronReference = electronsPositions.segment(electronIndex*3,3);
                distanceReference = (electronReference-nucleusPosition3d).norm();
                if (distanceReference<smallestDistance){
                    smallestDistance = distanceReference;
                    nearestElectronIndex = electronIndex;
                }
            }

            return nearestElectronIndex;
        }

        Scalar nearestElectronDistanceReturn(){
            return smallestDistance;
        }

        Vector3d correctStepMod(const Vector3d &electronPosition3d,
                                const Vector3d &nucleusPosition3d,
                                const Vector3d &gradient3d,
                                const Vector3d &stepLength3d,
                                unsigned long electronIndex) {

            Vector3d electronPosition3dNew;
            Vector3d velocityElectron = -gradient3d;
            Vector3d distance = electronPosition3d- nucleusPosition3d;

            //if threshold to close, move the electron exactly in the core and stop doing the correctStep
            //(if all electron are moved, this method will only return the nuleus position)
            if (distance.norm() < threshhold) {
                electronPosition3dNew = nucleusPosition3d;
                //erase-remove idiom to delete the element of the vector which has the value electronIndex
                indicesOfElectronsNotAtCores.erase(
                        std::remove(indicesOfElectronsNotAtCores.begin(),
                                    indicesOfElectronsNotAtCores.end(), electronIndex),
                        indicesOfElectronsNotAtCores.end());
                return electronPosition3dNew;
            }

            Vector3d distanceNormalized = distance/(distance.norm());

            Scalar vz = velocityElectron.dot(distanceNormalized);

            Vector3d orthogonalToDistance = velocityElectron-(vz*distanceNormalized);
            Scalar orthogonalToDistanceNorm = (orthogonalToDistance).norm();
            //important to prevent nan as a result
            if(orthogonalToDistanceNorm==0.0){
                return correctStepCut(electronPosition3d,nucleusPosition3d,gradient3d,stepLength3d);
            }

            Vector3d orthogonalToDistanceNormalized = orthogonalToDistance/orthogonalToDistanceNorm;

            Scalar rate = stepLength3d.norm()/velocityElectron.norm();
            
            //Parameters to calculate the new position
            Scalar zNew = std::max(distance.norm()+vz*rate,0.0);
            Scalar rhoNew = 2*orthogonalToDistanceNorm*rate*zNew/(distance.norm()+zNew);

            electronPosition3dNew = nucleusPosition3d + rhoNew*orthogonalToDistanceNormalized + zNew*distanceNormalized;
            return electronPosition3dNew;
        }

        Vector3d correctStepCut(const Vector3d &electronPosition3d,
                                const Vector3d &nucleusPosition3d,
                                const Vector3d &gradient3d,
                                const Vector3d &stepLength3d){

            Vector3d distance = electronPosition3d- nucleusPosition3d;
            Vector3d velocityElectronNormalized = (-gradient3d)/(-gradient3d).norm();

            //orthogonal projection of distVec on to velocityElNormalized, when the length of StepLength3D is bigger
            Vector3d orthogonalProjection = -distance.dot(velocityElectronNormalized)*velocityElectronNormalized;
            if (std::min(stepLength3d.norm(),(-distance.dot(velocityElectronNormalized)*velocityElectronNormalized).norm())==stepLength3d.norm()){
                orthogonalProjection = stepLength3d;
            }

            Vector3d electronPosition3dNew = electronPosition3d + orthogonalProjection;
            return electronPosition3dNew;
        }

        std::vector<unsigned long> getIndicesOfElectronsNotAtCores (){
            return indicesOfElectronsNotAtCores;
        }

    private:
        unsigned long numberOfElectrons;
        Scalar smallestDistance;
        Scalar threshhold;
        std::vector<unsigned long> indicesOfElectronsNotAtCores;
    };
} /* namespace cppoptlib */

#endif //AMOLQCGUI_UMRIGARCORRECTION_H
