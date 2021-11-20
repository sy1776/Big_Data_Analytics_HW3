/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse8803.clustering

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

//import java.io.File

object Metrics {
  /**
   * Given input RDD with tuples of assigned cluster id by clustering,
   * and corresponding real class. Calculate the purity of clustering.
   * Purity is defined as
   *             \fract{1}{N}\sum_K max_j |w_k \cap c_j|
   * where N is the number of samples, K is number of clusters and j
   * is index of class. w_k denotes the set of samples in k-th cluster
   * and c_j denotes set of samples of class j.
   * @param clusterAssignmentAndLabel RDD in the tuple format
   *                                  (assigned_cluster_id, class)
   * @return purity
   */
  def purity(clusterAssignmentAndLabel: RDD[(Int, Int)]): Double = {
    /**
     * TODO: Remove the placeholder and implement your code here
     */
    val purityIntersection = clusterAssignmentAndLabel.map(s => ((s._1,s._2), 1)).reduceByKey(_ + _)
    val purityMax = purityIntersection.map(s => (s._1._1, s._2)).reduceByKey(math.max)
    val puritySum = purityMax.map(s => s._2).sum()
    val purity = (puritySum / clusterAssignmentAndLabel.count().toDouble)

    //println("PuritySum: ", puritySum)

    /*
    val file = new File("temp/purityIntersection")
    if (file.exists) {
      purityIntersection.repartition(1).saveAsTextFile("temp/purityIntersection1")
      purityMax.repartition(1).saveAsTextFile("temp/purityMax1")
    } else {
      purityIntersection.repartition(1).saveAsTextFile("temp/purityIntersection")
      purityMax.repartition(1).saveAsTextFile("temp/purityMax")
    }*/

    purity
  }
}
