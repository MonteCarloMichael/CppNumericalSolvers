//
// Created by Michael Heuer on 13.02.17.
//

#include <gmock/gmock.h>
//#include "./../../linesearch/smallestvectorinconvexhullfinder.h
#include "linesearch/smallestvectorinconvexhullfinder.h"
#include "TestProblems.h"

using namespace testing;

class ASmallestVectorInConvexHullFinderTest : public Test {};

TEST_F(ASmallestVectorInConvexHullFinderTest , Example1) {

  static const int Dim = 2;
  static const int SetSize = 4;
  typedef double Scalar;
  using TVector   = Eigen::Matrix<Scalar, Dim, 1>;
  using TSetMatrix  = Eigen::Matrix<Scalar, SetSize, Dim>;

  TSetMatrix G;
  G(0,0) = +1.0; G(0,1) = +0.0;
  G(1,0) = +0.0; G(1,1) = +1.0;
  G(2,0) = -2.0; G(2,1) = +0.0;
  G(3,0) = +0.0; G(3,1) = -2.0;

  cppoptlib::SmallestVectorInConvexHullFinder<Scalar, Dim, SetSize> finder;

  TVector resultVector = finder.findSmallestVectorInConvexHull(G);

  Eigen::Vector2d referenceVector;
  referenceVector << 0.5,0.5;

  ASSERT_TRUE(resultVector.isApprox(referenceVector));
}

