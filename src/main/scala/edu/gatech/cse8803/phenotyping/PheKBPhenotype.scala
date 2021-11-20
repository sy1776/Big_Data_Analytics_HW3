/**
  * @author Hang Su <hangsu@gatech.edu>,
  * @author Sungtae An <stan84@gatech.edu>,
  */

package edu.gatech.cse8803.phenotyping

import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import org.apache.spark.rdd.RDD

object T2dmPhenotype {
  
  // criteria codes given
  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
      "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")

  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
      "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")

  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
      "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
      "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
      "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
      "avandia", "actos", "actos", "glipizide")

  val DM_RELATED_DX = Set("250.", "256.4", "277.7", "648.0", "648.01", "648.02", "648.03", "648.04", "648.81", "648.82", "648.83", "648.84",
                         "790.2", "790.21", "790.22", "790.29", "791.5", "V77.1")
  /**
    * Transform given data set to a RDD of patients and corresponding phenotype
    *
    * @param medication medication RDD
    * @param labResult lab result RDD
    * @param diagnostic diagnostic code RDD
    * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
    */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
      * Remove the place holder and implement your code here.
      * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
      * When testing your code, we expect your function to have no side effect,
      * i.e. do NOT read from file or write file
      *
      * You don't need to follow the example placeholder code below exactly, but do have the same return type.
      *
      * Hint: Consider case sensitivity when doing string comparisons.
      */

    val sc = medication.sparkContext

    /** Hard code the criteria */
    val type1_dm_dx = Set("code1", "250.03")
    val type1_dm_med = Set("med1", "insulin nph")

    /** Find CASE Patients */
    val med = medication.map(s => (s.patientID, s.medicine))
    val diag = diagnostic.map(s => (s.patientID, s.code) )
    //val diagMed = diag.leftOuterJoin(med).map(s => (s._1 -> "patientID", s._2._1 -> "code", s._2._2 -> "medicine"))
    //val diagMed = diag.leftOuterJoin(med).filter{case (patientId, (code, medicine)) => medicine != None}.map(s => (s._1, s._2._1, s._2._2))
    val diagMed = diag.leftOuterJoin(med).map(s => (s._1, s._2._1, s._2._2))

    val type2DMDiag = diagMed.filter(s => T2DM_DX.contains(s._2))
    //diagMed.saveAsTextFile("diagMed")
    //type2DMDiag.saveAsTextFile("type2DMDiag")
    //filter out tuple having 'None' and get the value of Some class using 'get' for s._3
    val filteredSome = type2DMDiag.filter(s => s._3 != None).map(s => (s._1, s._2, s._3.get))
    //below has 4 patient ids
    val casePatient1 = type2DMDiag.filter(s => s._3 == None).map(s => s._1).distinct()
    val type1DMMed = filteredSome.filter(s => T1DM_MED.contains(s._3))
    //below has 423 patient ids
    val casePatient2 = filteredSome.map(_._1).distinct.subtract(type1DMMed.map(_._1).distinct)

    //println("PhenoType - distinct patient counts of Order for Type 1 DM Medicine: ", type1DMMed.map(_._1).distinct.count())

    val type2DMMed = filteredSome.filter(s => T2DM_MED.contains(s._3))
    //type2DMMed.saveAsTextFile("type2DMMed")
    //below has 255 patient ids
    val casePatient3 = type1DMMed.map(_._1).distinct.subtract(type2DMMed.map(_._1).distinct)

    //println("PhenoType - distinct patient counts of No Order for Type 2 DM Medicine: ", casePatient3.count())

    val uniquePatients_type2DMMed = type1DMMed.map(_._1).distinct.subtract(casePatient3)
    val set = uniquePatients_type2DMMed.collect().toSet

    val filteredMed = medication.filter(s => set.contains(s.patientID))
    val groupedType2DMMed = filteredMed.filter(s => T2DM_MED.contains(s.medicine)).groupBy(s => s.patientID)
    val groupedType1DMMed = filteredMed.filter(s => T1DM_MED.contains(s.medicine)).groupBy(s => s.patientID)
    val minType2DMMed = groupedType2DMMed.map(s => (s._1, s._2.minBy(s => s.date).date))
    val minType1DMMed = groupedType1DMMed.map(s => (s._1, s._2.minBy(s => s.date).date))
    val casePatient4 = minType2DMMed.join(minType1DMMed).filter(s => s._2._1.before(s._2._2)).map(s => s._1).distinct

    //med_type2DMMed.saveAsTextFile("med_type2DMMed")
    //println("PhenoType - distinct patient counts of Order for Type2 DM Med: ", filteredMed.map(_.patientID).distinct.count())
    //println("PhenoType - distinct patient counts of Type 2 DM med before Type 1 PM Med: ", casePatient4.count())

    //val casePatients = sc.parallelize(Seq(("controlPatients-one", 2), ("controlPatients-two", 2), ("controlPatients-three", 2)))
    val casePatients = casePatient1.union(casePatient2).union(casePatient3).union(casePatient4).distinct
    //println("PhenoType - Total distinct patient counts: ", casePatients.count())


    /** Find CONTROL Patients */
    val lab_glucose = labResult.filter(s => s.testName.contains("glucose"))
    val lab_abnormalRecs = labResult.filter(s => checkResult(s))
    val noAbnormalPatients = lab_glucose.map(s => s.patientID).distinct.subtract(lab_abnormalRecs.map(s => s.patientID).distinct)

    //println("PhenoType - distinct patient counts of Any Type of Glucose Measure: ", lab_glucose.map(s => s.patientID).distinct.count())
    //println("PhenoType - distinct patient counts of Abnormal lab value: ", lab_abnormalRecs.map(s => s.patientID).distinct.count())
    //println("PhenoType - distinct patient counts of No Abnormal lab value: ", noAbnormalPatients.count())

    //Only code that starts with '250.' exists in original file, encounter_dx_INPUT.csv. Rest of codes in DM_RELATED_DX don't exsit
    val diagDM = diagnostic.filter(s => s.code.contains("250."))
    //diagDM.saveAsTextFile("diagDM")
    //println("PhenoType - distinct patient counts of Diag DM: ", diagDM.map(s => s.patientID).distinct.count())

    val controlPatients = noAbnormalPatients.subtract(diagDM.map(s => s.patientID).distinct)
    //println("PhenoType - distinct patient counts of Control: ", controlPatients.count())

    /** Find OTHER Patients */
    val allUniquePatients = diagnostic.map(s => s.patientID).distinct()
    val others = allUniquePatients.subtract(casePatients).subtract(controlPatients)

    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients.map(s => (s, 1)), controlPatients.map(s => (s, 2)), others.map(s => (s, 3)))
    //println("PhenoType - total # of unique patients: ", phenotypeLabel.count())
    //phenotypeLabel.saveAsTextFile("phenotypeLabel")

    /** Return */
    phenotypeLabel
  }

  def checkResult(x: LabResult): Boolean = x.testName match {
    case "hba1c" if x.value >= 6.0 => true
    case "hemoglobin a1c" if x.value >= 6.0 => true //x.value >= 6
    case "fasting glucose" if x.value >= 110.0 => true //x.value >= 110
    case "fasting blood glucose" if x.value >= 110.0 => true //x.value >= 110
    case "fasting plasma glucose" if x.value >= 110.0 => true //x.value >= 110
    case "glucose" if x.value > 110.0 => true
    case "glucose, Serum" if x.value > 110.0 => true //x.value > 110
    case _ => false
  }
}
