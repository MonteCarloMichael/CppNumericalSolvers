//
// Created by Michael Heuer on 13.02.17.
//

#include <gmock/gmock.h>
#include "linesearch/smallestvectorinconvexhullfinder.h"
#include "TestProblems.cpp"
#include <iomanip>

using namespace testing;

/* Examples are obtained by comparison with the results of the pyHANSO package*/

class ASmallestVectorInConvexHullFinderTest : public Test {};

TEST_F(ASmallestVectorInConvexHullFinderTest , Example1) {

  static const int Dim = 2;
  static const int SetSize = 4;
  typedef double Scalar;
  using TVector = Eigen::Matrix<Scalar, Dim, 1>;
  using TSetVector = Eigen::Matrix<Scalar, SetSize, 1>;
  using TSetMatrix = Eigen::Matrix<Scalar, Dim, SetSize>;

  TSetMatrix G;
  G(0,0) = +1.0; G(0,1) = +0.0; G(0,2) = -2.0; G(0,3) = +0.0;
  G(1,0) = +0.0; G(1,1) = +1.0; G(1,2) = +0.0; G(1,3) = -2.0;

  cppoptlib::SmallestVectorInConvexHullFinder<Scalar, Dim, SetSize> finder;

  auto result = finder.findSmallestVectorInConvexHull(G);
  TSetVector resultSetPoint = result.first;
  TVector resultVector = result.second;

  Eigen::Vector4d optimalPointReference;
  optimalPointReference << 0.3333297389712406, 0.3333297389712406, 0.1666702610287593, 0.1666702610287593;

  Eigen::Vector2d smallestVectorReference;
  smallestVectorReference << -1.078308627805447e-05, -1.078308627805447e-05;

  ASSERT_TRUE(resultSetPoint.isApprox(optimalPointReference));
  ASSERT_TRUE(resultVector.isApprox(smallestVectorReference));
}


TEST_F(ASmallestVectorInConvexHullFinderTest , Example2) {

  static const int Dim = 6;
  static const int SetSize = 12;
  typedef double Scalar;
  using TVector = Eigen::Matrix<Scalar, Dim, 1>;
  using TSetVector = Eigen::Matrix<Scalar, SetSize, 1>;
  using TSetMatrix = Eigen::Matrix<Scalar, Dim, SetSize>;

  TSetMatrix G;
  G <<
    0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,\
    0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,\
    0.0000000000000000, 0.0000000000000000,\
    0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,\
    0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,\
    0.0000000000000000, 0.0000000000000000,\
    0.0987159457723255, -0.1136355796054624, -0.1136355793648324, -0.1136355793617073, -0.1136355793804577,\
    0.0738274229095043, 0.0738274099936114, 0.0738274098811091, 0.0738274098873592, 0.0738274040466131,\
    0.0738262610916597, 0.0738262610854095,\
    0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,\
    0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,\
    0.0000000000000000, 0.0000000000000000,\
    0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,\
    0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,\
    0.0000000000000000, 0.0000000000000000,\
    -0.0502083535235470, -0.0165720830934752, -0.0134187550568042, -0.0134187131965611, -0.0134187130684334,\
    0.0116432076935665, 0.0116433015736384, 0.0116433021642757, 0.0116433021580255, 0.0116434456360011,\
    0.0116493401363706, 0.0116493401676212;

  cppoptlib::SmallestVectorInConvexHullFinder<Scalar, Dim, SetSize> finder;

  auto result = finder.findSmallestVectorInConvexHull(G);
  TSetVector resultSetPoint = result.first;
  TVector resultVector = result.second;

  Eigen::VectorXd optimalPointReference(12);
  optimalPointReference <<
                        0.06734452204453059,0.096596899087610070, 0.10108530963040750,\
                        0.10108537085788600, 0.10108537106860440, 0.07611435128054315,\
                        0.07611436628298404, 0.07611436637863113, 0.07611436637741484,\
                        0.07611438874502940, 0.07611534412064290, 0.07611534412571590;

  Eigen::VectorXd smallestVectorReference(6);
  smallestVectorReference <<
                          0., 0., 0.0005457126990673746,\
                          0., 0.,-0.0028468759377535980;

  ASSERT_TRUE(resultSetPoint.isApprox(optimalPointReference));
  ASSERT_TRUE(resultVector.isApprox(smallestVectorReference));
}

