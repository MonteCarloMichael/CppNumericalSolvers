//
// Created by dahl on 11.10.17.
//

#include <gmock/gmock.h>
#include <Eigen/Core>
#include <iostream>
#include <solver/gradientdescentsimplesolver.h>
#include "solver/gradientdescentumrigarlimitedsteplength.h"
#include "TestProblems.cpp"

using namespace testing;

class AGradientDescentUmrigarLimitedStepLengthSolverTest : public Test {};

TEST_F(AGradientDescentUmrigarLimitedStepLengthSolverTest , twoCuspsOneNucleusSmoothQuadraticTest) {
    Eigen::VectorXd x(9);
    x << 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5;
    UmrigarTwoCuspsOneNucleusSmoothQuadradicProblem9D f;

    cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
    crit.iterations = 1000;
    crit.gradNorm = 1e-5;
    cppoptlib::GradientDescentUmrigarLimitedSteplength<UmrigarTwoCuspsOneNucleusSmoothQuadradicProblem9D> solver;
    solver.setDebug(cppoptlib::DebugLevel::High);
    solver.setStopCriteria(crit);
    solver.setMaxStepLength(1e-1);
    solver.setSteepestDescentRate(1.0);
    solver.setDistanceCriteriaUmrigar(0.5);
    solver.setThreshholdUmrigar(1e-5);
    solver.minimize(f,x);

    Eigen::VectorXd xref(9);
    xref << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    ASSERT_TRUE((x-xref).norm() < 1e-2);
}

TEST_F(AGradientDescentUmrigarLimitedStepLengthSolverTest , cuspSmoothQuadraticTest) {
    Eigen::VectorXd x(6);
    x << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
    UmrigarCuspSmoothQuadradicProblem6D f;

    cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
    crit.iterations = 1000;
    crit.gradNorm = 1e-5;
    cppoptlib::GradientDescentUmrigarLimitedSteplength<UmrigarCuspSmoothQuadradicProblem6D> solver;
    solver.setDebug(cppoptlib::DebugLevel::High);
    solver.setStopCriteria(crit);
    solver.setMaxStepLength(1e-2);
    solver.setSteepestDescentRate(1e-3);
    solver.setDistanceCriteriaUmrigar(0.5);
    solver.setThreshholdUmrigar(1e-5);
    solver.minimize(f,x);

    Eigen::VectorXd xref(6);
    xref << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    ASSERT_TRUE((x-xref).norm() < 1e-2);
}

TEST_F(AGradientDescentUmrigarLimitedStepLengthSolverTest , cuspUmrigarTest) {
    Eigen::VectorXd x(3);
    x << 1.0, 1.0, 1.0;
    UmrigarCuspProblem3D f;

    cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
    crit.iterations = 1000;
    crit.gradNorm = 1e-5;
    cppoptlib::GradientDescentUmrigarLimitedSteplength<UmrigarCuspProblem3D> solver;
    solver.setDebug(cppoptlib::DebugLevel::High);
    solver.setStopCriteria(crit);
    solver.setMaxStepLength(1e-1);
    solver.setSteepestDescentRate(1e-5);
    solver.setDistanceCriteriaUmrigar(1e-1);
    solver.setThreshholdUmrigar(1e-5);
    solver.minimize(f,x);

    Eigen::VectorXd xref(3);
    xref << 0.0, 0.0, 0.0;
    ASSERT_TRUE((x-xref).norm() < 1e-2);
}

TEST_F(AGradientDescentUmrigarLimitedStepLengthSolverTest , firstTestForDebugging) {
    Eigen::VectorXd x(3);
    x << 1.0, 1.0, 1.0;
    CuspProblem3D f;

    cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
    crit.iterations = 1000;
    crit.gradNorm = 1e-5;
    cppoptlib::GradientDescentUmrigarLimitedSteplength<CuspProblem3D> solver;
    solver.setDebug(cppoptlib::DebugLevel::High);
    solver.setStopCriteria(crit);
    solver.setMaxStepLength(0.5);
    solver.setDistanceCriteriaUmrigar(1e-2);
    solver.minimize(f,x);

    Eigen::VectorXd xref(3);
    xref << 0.0, 0.0, 0.0;
    ASSERT_TRUE((x-xref).norm() < 1e-6);
}

class AGradientDescentSimpleSolverTest : public Test {};

TEST_F(AGradientDescentSimpleSolverTest , Minimum) {
    Eigen::VectorXd  x(2);
    x << 1.0, 1.0;
    QuadraticMinimum2DProblem f;

    cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
    crit.iterations = 100000;
    crit.gradNorm = 1e-5;
    cppoptlib::GradientDescentSimpleSolver<QuadraticMinimum2DProblem> solver;
    solver.setDebug(cppoptlib::DebugLevel::High);
    solver.setStopCriteria(crit);
    solver.minimize(f, x);

    Eigen::VectorXd xref(2);
    xref << 0.0, 0.0;
    ASSERT_TRUE((x-xref).norm() < 1e-6);
}



class CalculationsWithEigen : public Test {};

TEST_F(CalculationsWithEigen, AddTwoVectors) {

    Eigen::Vector3d vec1(1,2,3.3);
    //std::cout << vec1 << std::endl;
    Eigen::Vector3d vecReference(2,4,6.6);

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;

    VectorXd a(3);
    a=vec1;

    //std::cout << a << std::endl;

    MatrixXd ma1(3,3);

    for(int i=0;i<=2;++i){
        for(int j=0;j<=2;++j){
            ma1(i,j)=i*j;
        }
    }

    ma1.resize(4,3);

    for(int i=0;i<=3;++i){
        for(int j=0;j<=2;++j){
            if(i*j==0){
                ma1(i,j)= 203;
            }else{
                ma1(i,j)=i*j;
            }
        }
    }

    MatrixXd ma2(2,2);
    ma2 << 2.2,2.3,2.4,2.5;

    //Skalarmult mit reellen Zahlen
    ma1*=2;

    //Skalarprodukt (Vektor,Vektor)
    double sc1=vec1.dot(vec1);

    //Norm bilden + normieren
    double norm1=vec1.norm();
    Eigen::Vector3d vec3=vec1.normalized();

    //std::cout<< norm1 << "\n" << vec3.norm() << std::endl;

    //Maximumssuche
    double sc2=3.0;
    double sc3=std::max(sc1,sc2);

    std::cout<< ma1 <<std::endl;

    ASSERT_EQ(vec1+vec1,vecReference);

    /*dynamische vektoren und matrizen testen. templatisierung und
     * eigenlibrary testen und vertraut machen. ggf. mit anderen optimieren
     * arbeiten zum vergleichen.*/
}
