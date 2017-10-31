package com.sysu.nlp
import scala.collection.mutable.ArrayBuffer 
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD


object Lda {

  def train(
    sc: SparkContext,
    corpus: Array[List[Int]],
    numTopics: Int,
    wordCount: Int,
    iterations: Int=1,
    optimizer: String="em",
    numTopicWords: Int=10): RDD[Array[Tuple2[Int, Double]]] = {

    // convert the corpus to 'bag of words' model
    var bow = new Array[Tuple2[Long, Vector]](corpus.length)
    for (i <- 0 until corpus.length) {
      var doc = corpus(i)
      var bag = ArrayBuffer[Tuple2[Int, Double]]()
      for (w <- doc) {
        var flag = false
        for (j <- 0 until bag.length) {
          if (bag(j)._1 == w) {
            bag(j) = (bag(j)._1, bag(j)._2+1)
            flag = true
          }
        }
        if (!flag) bag ++= Array((w, 1.0))
      }
      bow(i) = (i.toLong, Vectors.sparse(wordCount, bag))
    }
    println("finsih converting corpus!")

    //train LDA model
    var rdd = sc.parallelize(bow)
    var lda = new LDA()
    var result = lda
      .setK(numTopics)
      .setMaxIterations(iterations)
      .setOptimizer(optimizer)
      .run(rdd)
      .describeTopics(numTopicWords)
    println("finish training LDA!")

    // convert the result to the form we need
    var topics = new Array[Array[Tuple2[Int, Double]]](numTopics)
    for (i <- 0 until numTopics) {
      topics(i) = new Array[Tuple2[Int, Double]](numTopicWords)
      for (j <- 0 until numTopicWords) topics(i)(j) = (result(i)._1(j), result(i)._2(j))
    }
    return sc.parallelize(topics)
  }

}