import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.functions.{min, max}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, CrossValidatorModel}

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
/**
  * Created by edwardcannon on 26/06/2017.
  * Simple example of building a logistic regression model
  * to predict Credit card fraud cases from Kaggle competition
  * using spark ml
  */
object CreditCardMainEntry {

  /**
    * Reads in a csv to sql dataframe
    * @param spark - Spark session
    * @param ifile - input .csv file path
    * @return
    */
  def read_csv(spark: org.apache.spark.sql.SparkSession, ifile: String) : org.apache.spark.sql.DataFrame ={
    spark.read.option("header","true").option("inferSchema", "true").csv(ifile)
  }

  /**
    * Creates logistic regression model and cross-validates with 5-fold
    * @param df - Input data frame
    * @return
    */
  def buildLogisticRegression(df : org.apache.spark.sql.DataFrame): (org.apache.spark.ml.tuning.CrossValidatorModel, org.apache
      .spark.sql.DataFrame) = {
    val lr = new LogisticRegression()
    // Print out the parameters, documentation, and any default values.
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")
    lr.setMaxIter(10)
      .setRegParam(0.01)
    val dfx = df.withColumnRenamed("Class", "label")
    val columnNames = dfx.columns.slice(0,dfx.columns.size-1)
    //all data types are string and should be float
    val assembler = new VectorAssembler()
      .setInputCols(columnNames)
      .setOutputCol("features")
    val output = assembler.transform(dfx)
    val trainDf = output.select("features","label")
    val testerDF = trainDf.select("features")
    val model1 = lr.fit(trainDf)
    val trainingSummary = model1.summary
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
    val roc = binarySummary.roc
    roc.show()
    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")
    val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).build()
    val cv = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
    val cvModel = cv.fit(trainDf)
    (cvModel, testerDF)
  }

  def main(args: Array[String]) : Unit = {
    val spark = SparkSession.builder
      .master("local[2]")
      .appName("Fraud detection")
      .getOrCreate()

    val df = CreditCardMainEntry.read_csv(spark,args(0))
    val cvModel, testerDF = CreditCardMainEntry.buildLogisticRegression(df)
    cvModel._1.transform(testerDF._2).groupBy("prediction").count().show()
  }
}
