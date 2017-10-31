package com.sysu.nlp
import scala.math._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._


class Predicter(
  wordVec: Array[Array[Double]],
  topicModel: RDD[Array[Tuple2[Int, Double]]]
  ) extends java.io.Serializable{

  private val vecDim = wordVec(0).length
  private val topicVec = topicModel.map(topic => this.calTopicVec(topic)).collect()

  def predict(corpus: RDD[List[Int]]): Array[Int] = corpus.map(doc => this.predictDoc(doc)).collect()

  /*
   * calculate each topic's vector according to the result of LDA using word2vec
  */
  private def calTopicVec(topic: Array[Tuple2[Int, Double]]): Array[Double] = {
    var vec = new Array[Double](vecDim)
    for (i <- 0 until topic.length) {
      for (j <- 0 until vecDim) vec(j) += wordVec(topic(i)._1)(j)*topic(i)._2
    }
    return vec
  }

  /*
   * calculate the vector of a document
   * find the topic whose vector has highest cosine similarity
  */
  private def predictDoc(doc: List[Int]): Int = {
    var docVec = new Array[Double](vecDim)
    for (w <- doc) {
      for (i <- 0 until vecDim) docVec(i) += wordVec(w)(i)
    }

    var mostLikelyTopic: Int = 0
    var maxSimilarity: Double = 0
    for (i <- 0 until topicVec.length) {
      var numerator: Double = 0
      var denominator: Double = 0
      for (j <- 0 until vecDim) {
        numerator += topicVec(i)(j)*docVec(j)
        denominator += pow(topicVec(i)(j)-docVec(j), 2)
      }
      var sim = numerator/sqrt(denominator)
      if (sim > maxSimilarity) {
        mostLikelyTopic = i
        maxSimilarity = sim
      }
    }

    return mostLikelyTopic
  }
}