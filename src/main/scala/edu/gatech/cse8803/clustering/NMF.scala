package edu.gatech.cse8803.clustering

/**
  * @author Hang Su <hangsu@gatech.edu>
  */


import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix


object NMF {

  /**
   * Run NMF clustering 
   * @param V The original non-negative matrix 
   * @param k The number of clusters to be formed, also the number of cols in W and number of rows in H
   * @param maxIterations The maximum number of iterations to perform
   * @param convergenceTol The maximum change in error at which convergence occurs.
   * @return two matrixes W and H in RowMatrix and DenseMatrix format respectively 
   */
  def run(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): (RowMatrix, BDM[Double]) = {

    /**
     * TODO 1: Implement your code here
     * Initialize W, H randomly 
     * Calculate the initial error (Euclidean distance between V and W * H)
     */
    //(new RowMatrix(V.rows.map(_ => BDV.rand[Double](k)).map(fromBreeze).cache), BDM.rand[Double](k, V.numCols().toInt))
    var W = new RowMatrix(V.rows.map(_ => BDV.rand[Double](k)).map(fromBreeze).cache)
    var H = BDM.rand[Double](k, V.numCols().toInt)

    /**
     * TODO 2: Implement your code here
     * Iteratively update W, H in a parallel fashion until error falls below the tolerance value 
     * The updating equations are, 
     * H = H.* W^T^V ./ (W^T^W H)
     * W = W.* VH^T^ ./ (W H H^T^)
     */
    var prevErr = 0.0
    var err = getError(V,multiply(W,H))
    var i = 0

    while((abs(err-prevErr) > convergenceTol) && (i < maxIterations)){
      W.rows.cache()
      V.rows.cache()

      val Hs = H * H.t

      W = dotDiv(dotProd(W,multiply(V,H.t)),multiply(W,Hs))
      val Ws = computeWTV(W,W)

      H = (H :* computeWTV(W,V)) :/ (Ws * H)

      prevErr = err
      err = getError(V,multiply(W,H))

      W.rows.unpersist(false)
      V.rows.unpersist(false)

      i = i + 1
      //println(err-prevErr)
    }

    (W,H)
  }


  /**  
  * RECOMMENDED: Implement the helper functions if you needed
  * Below are recommended helper functions for matrix manipulation
  * For the implementation of the first three helper functions (with a null return), 
  * you can refer to dotProd and dotDiv whose implementation are provided
  */
  /**
  * Note:You can find some helper functions to convert vectors and matrices
  * from breeze library to mllib library and vice versa in package.scala
  */

  def getError(V: RowMatrix,WH: RowMatrix): Double = {
    val err = V.rows.zip(WH.rows).map(s => toBreezeVector(s._1) :- toBreezeVector(s._2)).map(s => s :* s).map(s => sum(s)).reduce((x,y) => x+y)/2
    err
  }

  /** compute the mutiplication of a RowMatrix and a dense matrix */
  def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {
    val rows = X.multiply(fromBreeze(d))

    rows
  }

 /** get the dense matrix representation for a RowMatrix */
 def getDenseMatrix(X: RowMatrix): BDM[Double] = {
   val rows = X.rows.map{s =>
     val V =  new BDM[Double](1, s.size, s.toArray)
     V
   }

   rows.reduce((x, y) => DenseMatrix.vertcat(x,y))
 }

  /** matrix multiplication of W.t and V */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {
    val rows = W.rows.zip(V.rows).map{s =>
        val Wt = new BDM[Double](s._1.size,1, s._1.toArray)
        val V =  new BDM[Double](1, s._2.size, s._2.toArray)
        val result = Wt * V
        result
    }

    rows.reduce(_+_)
  }

  /** dot product of two RowMatrixes */
  def dotProd(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :* toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot division of two RowMatrixes */
  def dotDiv(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :/ toBreezeVector(v2).mapValues(_ + 2.0e-15)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }
}
