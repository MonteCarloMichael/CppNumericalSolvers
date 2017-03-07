//
// Created by Michael Heuer on 13.02.17.
//

#ifndef AMOLQCGUI_NONSMOOTHTERMINATION_H
#define AMOLQCGUI_NONSMOOTHTERMINATION_H

#include <Eigen/Cholesky>
#include <Eigen/LU>

namespace cppoptlib {


  template<typename Scalar_, long int Dim_ = Eigen::Dynamic, long int SetSize_ = Eigen::Dynamic>
  class SmallestVectorInConvexHullFinder {
  public:
    static const long int Dim = Dim_;
    static const long int SetSize = SetSize_;
    typedef Scalar_ Scalar;
    using TVector   = Eigen::Matrix<Scalar, Dim, 1>;
    using TSetVector = Eigen::Matrix<Scalar, SetSize, 1>;
    using TSetMatrix  = Eigen::Matrix<Scalar, Dim, SetSize>;
    using TSquareSetMatrix  = Eigen::Matrix<Scalar, SetSize, SetSize>;
    //using TSetVectorPlusOne = Eigen::Matrix<Scalar, SetSize+1, 1>;
    using TFlattenedSquareMatrix = Eigen::Matrix<Scalar, SetSize*SetSize, 1>;

    void resizeFinder(const long int newDim, const long int newSetSize){
      // resize TVectors members
      r1.resize(newSetSize,1);
      r3.resize(newSetSize,1);
      r4.resize(newSetSize,1);
      r7.resize(newSetSize,1);
      zdx.resize(newSetSize,1);
      KT.resize(newSetSize,1);
      dx.resize(newSetSize,1);
      dz.resize(newSetSize,1);
      e.resize(newSetSize,1);
      x.resize(newSetSize,1);
      z.resize(newSetSize,1);

      //resize TVectorPlusOne
      //tempVec.resize(newSetSize+1,1);

      //resize TSetVector members
      d.resize(newDim,1);

      //resize TSquareMatrix members
      C.resize(newSetSize,newSetSize);
      Q.resize(newSetSize,newSetSize);
      QD.resize(newSetSize,newSetSize);
    };

    /* Computing shortest l2-norm vector in convex hull of cached gradients:
     */
    std::pair<TSetVector,TVector> findSmallestVectorInConvexHull(const TSetMatrix &G) {
      // x: primal variables
      // y: dual lagrange mutlipliers
      // z: dual slack variables

      e.setOnes(G.cols(),1);
      x = e;

      tempVec.resize(G.cols()+1,1);

      z = x; // initialize
      y = static_cast<Scalar>(0.0);

      //TODO/const Scalar mu0 = x.transpose().dot(z) / static_cast<Scalar>(SetSize);
      const Scalar mu0 = 1.0;
      const Scalar muTolerance = 1e-5;
      const Scalar residualNormTolerance = 1e-5;

      Q = G.transpose() * G;

      //Parameters
      Scalar stepSizeDamping = 0.9995;
      int deltaSigmaHeuristic = 3;

      const Scalar infinityNormedQ = Q.cwiseAbs().rowwise().sum().maxCoeff() + static_cast<Scalar>(2.0);


      const Scalar muStoppingConstant = muTolerance * mu0;
      const Scalar residualStoppingConstant = residualNormTolerance * infinityNormedQ;


      int maxit = 100;
      int k;
      for (k=0; k < maxit; ++k) {

        r1 = -Q*x + e * y + z;
        r2 = -1 + x.sum();
        r3 = -x.array() * z.array();

        tempVec.head(x.rows()) = r1.head(x.rows());
        tempVec.tail(1)(0) = r2;

        // infinity norm for a vector because below does not work:
        rs = tempVec.cwiseAbs().maxCoeff();//rs = tempVec.lpNorm<Eigen::Infinity>();

        mu = -(r3.sum()) / static_cast<Scalar>(G.cols());

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
        lu.compute(C.transpose());
        KT = lu.solve(e);

        M = KT.transpose().dot(KT);

        // compute the approx tangent direction
        computeTangentDirection();

        //Determine maximal step possible in the new direction
        // primal step size
        TSetVector p = -x.array() / dx.array();
        ap = calculateStepSize(p);
        // dual step size
        p = -z.array() / dz.array();
        ad = calculateStepSize(p);

        muaff = ((x + dx*ap).dot(z + dz * ad)) / static_cast<Scalar>(G.cols());
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
      };

      //TODO - use log from cppoptlib
      //if (k == maxit) std::cout << "max it reached" << std::endl;
      //else std::cout << "optimal convex hull vector found after " << k << " iterations"  << std::endl;

      replaceNegativeElementsByZero(x);// Project x onto R+

      x /= x.sum(); // normalize

      //set other output variables using best found x
      d = G * x;
      //q = d.norm();//d.dot(d); // TODO NEEDED?

      return std::make_pair(x,d);
    };

    Scalar calculateStepSize(const TSetVector &x) {
      Scalar min;
      if( x[0] < 0 ) min = std::numeric_limits<Scalar>::max();
      else min = x[0];

      for (int i = 1; i < x.rows(); ++i) {
        if ((x[i] < x[i - 1]) && (x[i] > static_cast<Scalar>(0.0))) min = x[i];
      }
      if (min > static_cast<Scalar>(1.0)) return static_cast<Scalar>(1.0);
      else return min;
    }

    void replaceNegativeElementsByZero(TSetVector &x) {
      for (int i = 0; i < x.size(); ++i) {
        if (x[i] < 0) x[i] = static_cast<Scalar>(0.0);
      }
    };

    void computeTangentDirection(){
      r4 = r3.array() / x.array();
      r4 += r1;
      lu.compute(C.transpose());
      r5 = KT.dot(lu.solve(r4));
      r6 = r2 + r5;
      dy = -r6 / M;
      r7 = r4 + (e * dy).eval();
      dx = QD.inverse() * r7;

      dz = (r3.array() - (z.array()*dx.array()).array() ) / x.array();
    };

  private:
    Scalar r2, rs, mu, muaff, ap, ad, sig, r5, r6, M, dy,\
      y, q;
    TSetVector r1, r3, r4, r7, zdx, KT, dx,  dz,\
      e,x,z;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tempVec;

    TVector d;

    TSquareSetMatrix C, Q, QD;
    Eigen::LLT<TSquareSetMatrix,Eigen::UpLoType::Upper> llt;
    Eigen::FullPivLU<TSquareSetMatrix> lu;
  };

}

#endif //AMOLQCGUI_NONSMOOTHTERMINATION_H
