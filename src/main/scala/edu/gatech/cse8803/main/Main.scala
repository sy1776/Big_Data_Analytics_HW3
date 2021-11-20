/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse8803.main

import java.text.SimpleDateFormat

import edu.gatech.cse8803.clustering.{NMF, Metrics}
import edu.gatech.cse8803.features.FeatureConstruction
import edu.gatech.cse8803.ioutils.CSVUtils
import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import edu.gatech.cse8803.phenotyping.T2dmPhenotype
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans, StreamingKMeans}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vectors, Vector}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source
//import java.io.File

object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = createContext
    val sqlContext = new SQLContext(sc)

    /** initialize loading of data */
    val (medication, labResult, diagnostic) = loadRddRawData(sqlContext)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData

    /** conduct phenotyping */
    val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication)
    )

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples)

    val (kMeansPurity, gaussianMixturePurity, streamKmeansPurity, nmfPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of StreamingKMeans is: $streamKmeansPurity%.5f")
    println(f"[All feature] purity of NMF is: $nmfPurity%.5f")

    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication)
    )

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples)

    val (kMeansPurity2, gaussianMixturePurity2, streamKmeansPurity2, nmfPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of StreamingKMeans is: $streamKmeansPurity2%.5f")
    println(f"[Filtered feature] purity of NMF is: $nmfPurity2%.5f")
    sc.stop 
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures:RDD[(String, Vector)]): (Double, Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix


    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray)))})
    val rawFeatureVectors = features.map(_._2).cache()

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]
    /** transform a feature into its reduced dimension representation */
    def transform(feature: Vector): Vector = {
      Vectors.dense(Matrices.dense(1, feature.size, feature.toArray).multiply(densePc).toArray)
    }

    val patientList = features.map(_._1)

    /*
    val file = new File("temp/featureVectors")

    if (file.exists) {
      featureVectors.repartition(1).saveAsTextFile("temp/featureVectors1")
    } else {
      featureVectors.repartition(1).saveAsTextFile("temp/featureVectors")
    } */
    /** TODO: K Means Clustering using spark mllib
      *  Train a k means model using the variabe featureVectors as input
      *  Set maxIterations =20 and seed as 8803L
      *  Assign each feature vector to a cluster(predicted Class)
      *  Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
      *  Find Purity using that RDD as an input to Metrics.purity
      *  Remove the placeholder below after your implementation
      **/
    val noOfIteration = 20
    val seed = 8803L
    val noOfClusters = 3
    //val transformedFeatures = transform(rawFeatureVectors)

    val kMeansModel = KMeans.train(featureVectors, noOfClusters, noOfIteration, 1, "k-means||", seed)
    val KMeansPredictedCluster = kMeansModel.predict(featureVectors)
    //Cluster index starts at 0. Thus, adding 1 to each clusters in order to compare with phenotype (1, 2, 3)
    val KMeansPredictedCl = KMeansPredictedCluster.map(s => s+1)
    val kMeansPhenoType = patientList.zip(KMeansPredictedCl)
    val kMeansComparison = kMeansPhenoType.join(phenotypeLabel)

    /*
    val file2 = new File("temp/phenotypeLabel")
    if (!file2.exists) {
      phenotypeLabel.repartition(1).saveAsTextFile("temp/phenotypeLabel")
    }

    val file1 = new File("temp/KMeansPredictedCluster")

    if (file1.exists) {
      KMeansPredictedCluster.repartition(1).saveAsTextFile("temp/KMeansPredictedCluster1")
      KMeansPredictedCl.repartition(1).saveAsTextFile("temp/KMeansPredictedCl1")
      //patientList.repartition(1).saveAsTextFile("temp/PatientList1")
      kMeansPhenoType.repartition(1).saveAsTextFile("temp/KMeansPhenoType1")
      kMeansComparison.repartition(1).saveAsTextFile("temp/KMeansComparison1")
    } else {
      KMeansPredictedCluster.repartition(1).saveAsTextFile("temp/KMeansPredictedCluster")
      KMeansPredictedCl.repartition(1).saveAsTextFile("temp/KMeansPredictedCl")
      //patientList.repartition(1).saveAsTextFile("temp/PatientList")
      kMeansPhenoType.repartition(1).saveAsTextFile("temp/KMeansPhenoType")
      kMeansComparison.repartition(1).saveAsTextFile("temp/KMeansComparison")
    }
    */
    compareClustering(kMeansComparison.map(_._2), "K-Means")
    val kMeansPurity = Metrics.purity(kMeansComparison.map(_._2))

    /** TODO: GMMM Clustering using spark mllib
      *  Train a Gaussian Mixture model using the variabe featureVectors as input
      *  Set maxIterations =20 and seed as 8803L
      *  Assign each feature vector to a cluster(predicted Class)
      *  Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
      *  Find Purity using that RDD as an input to Metrics.purity
      *  Remove the placeholder below after your implementation
      **/
    val GMMModel= new GaussianMixture().setK(noOfClusters).setMaxIterations(noOfIteration).setSeed(seed).run(featureVectors)
    val GMMPredictedCluster = GMMModel.predict(featureVectors)
    //Cluster index starts at 0. Thus, adding 1 to each clusters in order to compare with phenotype (1, 2, 3)
    val GMMPredictedCl = GMMPredictedCluster.map(s => s+1)
    val GMMPhenoType = patientList.zip(GMMPredictedCl)
    val GMMComparison = GMMPhenoType.join(phenotypeLabel)

    compareClustering(GMMComparison.map(_._2), "GMM")
    val gaussianMixturePurity = Metrics.purity(GMMComparison.map(_._2))

    /** TODO: StreamingKMeans Clustering using spark mllib
      *  Train a StreamingKMeans model using the variabe featureVectors as input
      *  Set the number of cluster K = 3 and DecayFactor = 1.0 and weight as 0.0
      *  please pay attention to the input type
      *  Assign each feature vector to a cluster(predicted Class)
      *  Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
      *  Find Purity using that RDD as an input to Metrics.purity
      *  Remove the placeholder below after your implementation
      **/
    val streamKModel = new StreamingKMeans().setK(noOfClusters).setDecayFactor(1.0).setRandomCenters(10, 0.0, seed)
    val streamKLatestModel = streamKModel.latestModel()
    val streamKPredictedCluster = streamKLatestModel.predict(featureVectors)
    val streamKPredictedCl = streamKPredictedCluster.map(s => s+1)
    val streamKPhenoType = patientList.zip(streamKPredictedCl)
    val streamKComparison = streamKPhenoType.join(phenotypeLabel)

    /*
    val file1 = new File("temp/streamKComparison")

    if (file1.exists) {
      streamKComparison.repartition(1).saveAsTextFile("temp/streamKComparison1")
      streamKPredictedCluster.repartition(1).saveAsTextFile("temp/streamKPredictedCluster1")
    } else {
      streamKComparison.repartition(1).saveAsTextFile("temp/streamKComparison")
      streamKPredictedCluster.repartition(1).saveAsTextFile("temp/streamKPredictedCluster")
    }*/
    compareClustering(streamKComparison.map(_._2), "Streaming K")
    val streamKmeansPurity = Metrics.purity(streamKComparison.map(_._2))

    /** NMF */
    val rawFeaturesNonnegative = rawFeatures.map({ case (patientID, f)=> Vectors.dense(f.toArray.map(v=>Math.abs(v)))})
    val (w, _) = NMF.run(new RowMatrix(rawFeaturesNonnegative), noOfClusters, 100)
    // for each row (patient) in W matrix, the index with the max value should be assigned as its cluster type
    val assignments = w.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
    // zip patientIDs with their corresponding cluster assignments
    // Note that map doesn't change the order of rows
    val assignmentsWithPatientIds=features.map({case (patientId,f)=>patientId}).zip(assignments) 
    // join your cluster assignments and phenotypeLabel on the patientID and obtain a RDD[(Int,Int)]
    // which is a RDD of (clusterNumber, phenotypeLabel) pairs 
    val nmfClusterAssignmentAndLabel = assignmentsWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value})
    // Obtain purity value
    val nmfPurity = Metrics.purity(nmfClusterAssignmentAndLabel)

    //In order to compare, need to add 1 to the clusters. To make 1,2,3 instead of 0,1,2
    val newNMFClusterAssignmentAndLabel = nmfClusterAssignmentAndLabel.map(s => (s._1 + 1, s._2))
    compareClustering(newNMFClusterAssignmentAndLabel, "NMF")

    (kMeansPurity, gaussianMixturePurity, streamKmeansPurity, nmfPurity)
  }

  def compareClustering(clusterAssignmentAndLabel: RDD[(Int, Int)], clusteringName:String) = {
    val casePatients = clusterAssignmentAndLabel.filter(s => s._2 == 1)
    val totalCasePatients = casePatients.count().toDouble
    val caseCluster1 = casePatients.filter(s => s._1 == 1)
    val totalCaseCluster1 = caseCluster1.count().toDouble
    val caseCluster2 = casePatients.filter(s => s._1 == 2)
    val totalCaseCluster2 = caseCluster2.count().toDouble
    val caseCluster3 = casePatients.filter(s => s._1 == 3)
    val totalCaseCluster3 = caseCluster3.count().toDouble

    val caseCluster1Percentage = (totalCaseCluster1 / totalCasePatients) *  100
    val caseCluster2Percentage = (totalCaseCluster2 / totalCasePatients) *  100
    val caseCluster3Percentage = (totalCaseCluster3 / totalCasePatients) *  100

    val controlPatients = clusterAssignmentAndLabel.filter(s => s._2 == 2)
    val totalControlPatients = controlPatients.count().toDouble
    val controlCluster1 = controlPatients.filter(s => s._1 == 1)
    val totalControlCluster1 = controlCluster1.count().toDouble
    val controlCluster2 = controlPatients.filter(s => s._1 == 2)
    val totalControlCluster2 = controlCluster2.count().toDouble
    val controlCluster3 = controlPatients.filter(s => s._1 == 3)
    val totalControlCluster3 = controlCluster3.count().toDouble

    val controlCluster1Percentage = (totalControlCluster1 / totalControlPatients) *  100
    val controlCluster2Percentage = (totalControlCluster2 / totalControlPatients) *  100
    val controlCluster3Percentage = (totalControlCluster3 / totalControlPatients) *  100

    val unknownPatients = clusterAssignmentAndLabel.filter(s => s._2 == 3)
    val totalUnknownPatients = unknownPatients.count().toDouble
    val unknownCluster1 = unknownPatients.filter(s => s._1 == 1)
    val totalUnknownCluster1 = unknownCluster1.count().toDouble
    val unknownCluster2 = unknownPatients.filter(s => s._1 == 2)
    val totalUnknownCluster2 = unknownCluster2.count().toDouble
    val unknownCluster3 = unknownPatients.filter(s => s._1 == 3)
    val totalUnknownCluster3 = unknownCluster3.count().toDouble

    val unknownCluster1Percentage = (totalUnknownCluster1 / totalUnknownPatients) *  100
    val unknownCluster2Percentage = (totalUnknownCluster2 / totalUnknownPatients) *  100
    val unknownCluster3Percentage = (totalUnknownCluster3 / totalUnknownPatients) *  100

    println(f"--------------------$clusteringName---------------------")
    println(f"Percentage Cluster | Case     | Control  | Unknown      ")
    println(f"Cluster 1          | $caseCluster1Percentage%.5f | $controlCluster1Percentage%.5f | $unknownCluster1Percentage%.5f")
    println(f"Cluster 2          | $caseCluster2Percentage%.5f | $controlCluster2Percentage%.5f | $unknownCluster2Percentage%.5f")
    println(f"Cluster 3          | $caseCluster3Percentage%.5f | $controlCluster3Percentage%.5f | $unknownCluster3Percentage%.5f")

  }
  /**
   * load the sets of string for filtering of medication
   * lab result and diagnostics
    *
    * @return
   */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }

  def loadRddRawData(sqlContext: SQLContext): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {
    /** You may need to use this date format. */
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")

    /** load data using Spark SQL into three RDDs and return them
      * Hint: You can utilize edu.gatech.cse8803.ioutils.CSVUtils and SQLContext.
      *
      * Notes:Refer to model/models.scala for the shape of Medication, LabResult, Diagnostic data type.
      *       Be careful when you deal with String and numbers in String type.
      *       Ignore lab results with missing (empty or NaN) values when these are read in.
      *       For dates, use Date_Resulted for labResults and Order_Date for medication.
      * */

    /** TODO: implement your own code here and remove existing placeholder code below */

    val mediation_orders_INPUT = CSVUtils.loadCSVAsTable(sqlContext, "data/medication_orders_INPUT.csv")
    val lab_results_INPUT = CSVUtils.loadCSVAsTable(sqlContext, "data/lab_results_INPUT.csv", "lab_results")
    val encounter_INPUT = CSVUtils.loadCSVAsTable(sqlContext, "data/encounter_INPUT.csv", "encounter")
    val encounter_dx_INPUT = CSVUtils.loadCSVAsTable(sqlContext, "data/encounter_dx_INPUT.csv", "encounter_dx")

    val lab_results = sqlContext.sql("select Member_ID, Date_Resulted, Result_Name, Numeric_Result from lab_results where Numeric_Result != ''" +
                                     " AND Numeric_Result not like '%,%'")
    val encounter = sqlContext.sql("select a.Member_ID, a.Encounter_DateTime, b.code from encounter a, encounter_dx b where a.Encounter_ID = b.Encounter_ID")
    //val encounter = sqlContext.sql("select a.Member_ID, a.Encounter_DateTime, b.code from encounter a inner join encounter_dx b on a.Encounter_ID = b.Encounter_ID")
    val medication: RDD[Medication] =  mediation_orders_INPUT.map(s =>Medication(s(1).asInstanceOf[String],
                                       dateFormat.parse(s(11).asInstanceOf[String]), s(3).asInstanceOf[String].toLowerCase()))
    val labResult: RDD[LabResult] =  lab_results.map(s => LabResult(s(0).asInstanceOf[String], dateFormat.parse(s(1).asInstanceOf[String]),
                                     s(2).asInstanceOf[String].toLowerCase, s(3).asInstanceOf[String].toDouble))
    //labResult.saveAsTextFile("labResult")
    val diagnostic: RDD[Diagnostic] =  encounter.map(s => Diagnostic(s(0).asInstanceOf[String], dateFormat.parse(s(1).asInstanceOf[String]),
                                       s(2).asInstanceOf[String]))


    //println("distinct patients in medication: ", medication.map(s => s.patientID).distinct.count())

    //println("distinct patients in dianostics: ",diagnostic.map(s => s.patientID).distinct.count())
    //println("dianostic: ",diagnostic.take(5).foreach(println) )

    //println("distinct patients in labresult: ", labResult.map(s => s.patientID).distinct.count())

    (medication, labResult, diagnostic)
  }

  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
    new SparkContext(conf)
  }

  def createContext(appName: String): SparkContext = createContext(appName, "local")

  def createContext: SparkContext = createContext("CSE 8803 Homework Two Application", "local")
}
