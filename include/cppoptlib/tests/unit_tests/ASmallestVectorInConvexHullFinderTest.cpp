//
// Created by Michael Heuer on 13.02.17.
//

#include <gmock/gmock.h>
//#include "./../../linesearch/smallestvectorinconvexhullfinder.h
#include "linesearch/smallestvectorinconvexhullfinder.h"
#include "TestProblems.cpp"

using namespace testing;

class ASmallestVectorInConvexHullFinderTest : public Test {};

TEST_F(ASmallestVectorInConvexHullFinderTest , Example1) {

  static const int Dim = 2;
  static const int SetSize = 4;
  typedef double Scalar;
  using TSetVector   = Eigen::Matrix<Scalar, SetSize, 1>;
  using TSetMatrix  = Eigen::Matrix<Scalar, Dim, SetSize>;

  TSetMatrix G;
  G(0,0) = +1.0; G(1,0) = +0.0;
  G(0,1) = +0.0; G(1,1) = +1.0;
  G(0,2) = -2.0; G(1,2) = +0.0;
  G(0,3) = +0.0; G(1,3) = -2.0;

  cppoptlib::SmallestVectorInConvexHullFinder<Scalar, Dim, SetSize> finder;

  TSetVector resultVector = finder.findSmallestVectorInConvexHull(G).first;

  Eigen::Vector4d referenceVector;
  referenceVector << 1.0,2.0,3.0,4.0;

  ASSERT_TRUE(resultVector.isApprox(referenceVector));
}


