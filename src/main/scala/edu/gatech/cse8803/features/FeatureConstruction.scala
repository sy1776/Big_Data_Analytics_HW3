/**
 * @author Hang Su
 */
package edu.gatech.cse8803.features

import edu.gatech.cse8803.model.{LabResult, Medication, Diagnostic}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import java.io.File

object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    //diagnostic.sparkContext.parallelize(List((("patient", "diagnostics"), 1.0)))
    val diagFeature = diagnostic.map(s => ((s.patientID, s.code), 1.0)).reduceByKey(_ + _)
    //diagFeature.repartition(1).saveAsTextFile("temp/diagFeature")

    diagFeature
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
   * @param medication RDD of medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove exist ing
     * placeholder code
     */
    //medication.sparkContext.parallelize(List((("patient", "med"), 1.0)))
    val medFeature = medication.map(s => ((s.patientID, s.medicine), 1.0)).reduceByKey(_ + _)

    medFeature
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */

    val labSumCount = labResult.map(s => ((s.patientID, s.testName), s.value, 1.0)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2,x._3+y._3))
    val labFeature = labSumCount.map(s => (s._1, s._2._2 / s._2._3))
    //labFeature.repartition(1).saveAsTextFile("temp/labFeature")

    labFeature
  }

  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
   * @param diagnostic RDD of diagnostics
   * @param candiateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candiateCode: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    //diagnostic.sparkContext.parallelize(List((("patient", "diagnostics"), 1.0)))
    val diagFiltered = diagnostic.filter(s => candiateCode.contains(s.code))
    val diagFeature = diagFiltered.map(s => ((s.patientID, s.code), 1.0)).reduceByKey(_ + _)
    //diagFeature.repartition(1).saveAsTextFile("temp/diagFeature")

    diagFeature
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation, use medications from
   * given set only and drop all others.
   * @param medication RDD of diagnostics
   * @param candidateMedication set of candidate medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    //medication.sparkContext.parallelize(List((("patient", "med"), 1.0)))
    val medFiltered = medication.filter(s => candidateMedication.contains(s.medicine))
    val medFeature = medFiltered.map(s => ((s.patientID, s.medicine), 1.0)).reduceByKey(_ + _)

    medFeature
  }


  /**
   * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
   * given set of lab test names only and drop all others.
   * @param labResult RDD of lab result
   * @param candidateLab set of candidate lab test name
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    //labResult.sparkContext.parallelize(List((("patient", "lab"), 1.0)))
    val labFiltered = labResult.filter(s => candidateLab.contains(s.testName))
    val labSumCount = labFiltered.map(s => ((s.patientID, s.testName), s.value, 1.0)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2,x._3+y._3))
    val labFeature = labSumCount.map(s => (s._1, s._2._2 / s._2._3))

    labFeature
  }


  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
   * @param sc SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    /** create a feature name to id map*/
    val featureMap = feature.map(_._1._2).
                            distinct.
                            collect.
                            zipWithIndex.
                            toMap

    /** transform input feature */

    /**
     * Functions maybe helpful:
     *    collect
     *    groupByKey
     */

    val scFeatureMap = sc.broadcast(featureMap)
    val featureGroupedBy = feature.map{case((patientID, feature), value) => (patientID, scFeatureMap.value(feature), value)}.
                        groupBy(_._1)

    val result = featureGroupedBy.map{case (patientID, features) =>
        val numFeature = scFeatureMap.value.size
        val list = features.toList.
                              map{case(patientID1, featureIndex, featureValue) => (featureIndex, featureValue)}
        val featureVector = Vectors.sparse(numFeature, list)
        val labeledPoint = (patientID, featureVector)
        labeledPoint
    }

    /*
    val file = new File("temp/result")

    if (file.exists) {
      result.repartition(1).saveAsTextFile("temp/result1")
    } else {
      result.repartition(1).saveAsTextFile("temp/result")
    }
    */
    result
    /** The feature vectors returned can be sparse or dense. It is advisable to use sparse */

  }
}


