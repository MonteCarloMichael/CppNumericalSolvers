//
// Created by Michael Heuer on 13.02.17.
//

#ifndef AMOLQCGUI_NONSMOOTHTERMINATION_H
#define AMOLQCGUI_NONSMOOTHTERMINATION_H

#include <Eigen/Cholesky>
//#include <Eigen/Householder>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace cppoptlib {

  template<typename Scalar_, int Dim_ = Eigen::Dynamic, int SetSize_ = Eigen::Dynamic>
  class SmallestVectorInConvexHullFinder {
  public:
    static const int Dim = Dim_;
    static const int SetSize = SetSize_;
    typedef Scalar_ Scalar;
    using TVector   = Eigen::Matrix<Scalar, Dim, 1>;
    using TSetVector = Eigen::Matrix<Scalar, SetSize, 1>;
    using TSetMatrix  = Eigen::Matrix<Scalar, SetSize, Dim>;
    using TSquareMatrix  = Eigen::Matrix<Scalar, Dim, Dim>;
    using TVectorPlusOne = Eigen::Matrix<Scalar,Dim +1,1>;
    using TFlattenedSquareMatrix = Eigen::Matrix<Scalar, Dim*Dim,1>;
    //SmallestVectorInConvexHullSetProblem() {};

    TVector findSmallestVectorInConvexHull(const TSetMatrix &G) {
    //TODO REPLACE LLT SOLVE BECAUSE MATRICES ARE NOT GURANTEED TO BE POSITIVE DEFINIT
      // x: primal variables
      // y: dual lagrange mutlipliers
      // z: dual slack variables

      e = Eigen::Matrix<Scalar,Dim,1>::Constant(1.0);
      x = e;

      z = x; // initialize
      y = static_cast<Scalar>(0.0);

      const Scalar mu0 = x.dot(z) / static_cast<Scalar>(Dim);
      // nicht x.transpose()*x = 1
      const Scalar muTolerance = 1e-5;
      const Scalar residualNormTolerance = 1e-5;

      Q = G.transpose() * G;

      //Parameters
      Scalar stepSizeDamping = 0.9995;
      int deltaSigmaHeuristic = 3;

      //const Scalar infinityNormedQ = Q.lpNorm<Eigen::Infinity>() + static_cast<Scalar>(2.0);
      const Scalar infinityNormedQ = Q.cwiseAbs().rowwise().sum().maxCoeff() + static_cast<Scalar>(2.0);


      const Scalar muStoppingConstant = muTolerance * mu0;
      const Scalar residualStoppingConstant = residualNormTolerance * infinityNormedQ;

      TVectorPlusOne tempVec;


      int maxit = 100;
      int k;
      for (k=0; k < maxit; ++k) {

        r1 = -Q * x + e * y + z;
        r2 = -1 + x.sum();
        r3 = -x.array() * z.array();

        tempVec.head(Dim) = r1.head(Dim);
        tempVec.tail(1)(0) = r2;

        // infinity norm for a vector because below does not work:
        //rs = tempVec.lpNorm<Eigen::Infinity>();
        rs = tempVec.cwiseAbs().maxCoeff();

        mu = -r3.sum() / static_cast<Scalar>(Dim);

        // Stop condition
        if (rs < residualStoppingConstant) {
          if (mu < muStoppingConstant) break;
        }

        zdx = z.array() / x.array(); //factorization??
        QD = Q;
        // treat diagonal elements specially
        QD.diagonal().array() += zdx.array();

        // Do cholesky decomposition
        llt.compute(QD);
        C = llt.matrixU();

        //KT = C.transpose().colPivHouseholderQr().solve(e);//KT = C.transpose().inverse() * e;
        //KT = C.inverse() * e; //TODO CHECK and COMPARE!
        llt.compute(C);
        KT = llt.solve(e);
        //hhQR.compute(C);
        //KT = hhQR.solve(e);

        M = KT.dot(KT);

        // compute the approx tangent direction
        computeTangentDirection();

        //Determine maximal step possible in the new direction
        // primal step size
        TVector p = -x.array() / dx.array();
        ap = calculateStepSize(p);
        // dual step size
        p = -z.array() / dz.array();
        ad = calculateStepSize(p);

        muaff = ((x + dx*ap).dot(z + dz * ad)) / static_cast<Scalar>(Dim);
        sig = std::pow(muaff / mu, deltaSigmaHeuristic);
        // compute the new corrected search direction taht now includes appropriate amount of centering and mehrotras
        // second order correction term ( see r3 ). We of course reuse the factorization from above
        r3.array() += sig * mu;
        r3.array() -= (dx.array() * dz.array());

        computeTangentDirection();

        p = -x.array() / dx.array();
        ap = calculateStepSize(p);
        // dual step size
        p = -z.array() / dz.array();
        ad = calculateStepSize(p);

        x = x + (dx * (stepSizeDamping * ap));
        y = y + dy * ad * stepSizeDamping;
        z = z + (dz * (stepSizeDamping * ad));
       //TODO rs never gets smaller than residualstoppingconstant, ES GEHT FÜR k=2 schief, in k=1 wird etwas falsch
        // gesettet!!
      };

      if (k == maxit) std::cout << "max it reached" << std::endl;

      replaceNegativeElementsByZero(x);// Project x onto R+

      x /= x.sum(); // normalize

      //set other output variables using best found x
      d = G * x;
      q = d.dot(d);

      return x;
    };

    Scalar calculateStepSize(const TVector &x) {
      Scalar min;
      if( x[0] < 0 ) min = std::numeric_limits<Scalar>::max();
      else min = x[0];

      for (int i = 1; i < x.rows(); ++i) {
        if ((x[i] < x[i - 1]) && (x[i] > static_cast<Scalar>(0.0))) min = x[i];
      }
      if (min > static_cast<Scalar>(1.0)) return static_cast<Scalar>(1.0);
      else return min;
    }

    void replaceNegativeElementsByZero(TVector &x) {
      for (int i = 0; i < x.size(); ++i) {
        if (x[i] < 0) x[i] = static_cast<Scalar>(0.0);
      }
    };

    void computeTangentDirection(){
      r4 = r3.array() / x.array();
      r4 += r1;
      //svd.compute(C.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
      //r5 = KT.dot(svd.solve(r4));
      llt.compute(C.transpose());
      r5 = KT.dot(llt.solve(r4));
      r6 = r2 + r5;
      dy = -r6 / M; //CORRECT
      r7 = r4 + (e * dy).eval();
      dx = QD.inverse() * r7; // TODO is inverse safe?  //CORRECT
      //llt.compute(QD);
      //dx = KT.dot(llt.solve(r7)); // TODO error: no match for 'operator='....

      dz = (r3.array() - (z.array()*dx.array()).array() ) / x.array(); //CORRECT

    };

  private:
    Scalar r2, rs, mu, muaff, ap, ad, sig, r5, r6, M, dy,\
      y, q;
    TVector r1, r3, r4, r7, zdx, KT, dx,  dz,\
      e,x,z;

    TSetVector d;

    TSquareMatrix C, Q, QD;
    Eigen::JacobiSVD<TSquareMatrix> svd;
    Eigen::LLT<TSquareMatrix,Eigen::UpLoType::Upper> llt;
    //Eigen::HouseholderQR<TSquareMatrix> hhQR = Eigen::HouseholderQR;
  };

}

#endif //AMOLQCGUI_NONSMOOTHTERMINATION_H
