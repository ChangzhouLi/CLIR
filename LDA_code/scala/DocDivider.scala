package com.sysu.nlp

import scala.math._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


class DocDivider(
  wordVec: Array[Array[Double]], 
  pattern: Array[Array[Int]], 
  signwordsQues: Array[Tuple2[Int, Double]],
  signwordsRes: Array[Int],
  stopwords: List[Int]) extends java.io.Serializable {

  val vecDim = wordVec(0).length
  val divider = new Divider(wordVec, pattern, signwordsQues, signwordsRes, stopwords)
  var questionText: Array[List[Int]] = new Array[List[Int]](1)
  var methodText: Array[List[Int]] = new Array[List[Int]](1)
  var resultText: Array[List[Int]] = new Array[List[Int]](1)

  // divide each doc in corpus to three parts 
  def divide(corpus: RDD[List[List[Int]]]): Unit = {
    var result = corpus.map(doc => divider.divideDoc(doc)).collect()
    questionText = new Array[List[Int]](result.length)
    methodText = new Array[List[Int]](result.length)
    resultText = new Array[List[Int]](result.length)

    for (i <- 0 until result.length) {
      questionText(i) = result(i)(0)
      methodText(i) = result(i)(1)
      resultText(i) = result(i)(2)
    }
  }
  
  class Divider(
    wordVec: Array[Array[Double]], 
    pattern: Array[Array[Int]], 
    signwordsQues: Array[Tuple2[Int, Double]],
    signwordsRes: Array[Int],
    stopwords: List[Int]) extends java.io.Serializable {
    
    // divide a doc into question, method and result
    def divideDoc(doc: List[List[Int]]): Array[List[Int]] = {
      var pos = Array(0, 0)
      var flag = 0
      var questionSent = List[Int]()
      var methodSent = List[Int]()
      var resSent = List[Int]()
      
      // find whether the abstract is in standard form
      for (sent <- doc) {
        for (w <- sent) {
          if (pattern(flag).exists(x => x == w)) {
            pos(flag) = doc.indexOf(sent)
            flag += 1
            if (flag == pattern.length) {
              for (i <- 0 until pos(0)) questionSent = questionSent ::: doc(i)
              for (i <- pos(0) until pos(1)) methodSent = methodSent ::: doc(i)
              for (i <- pos(1) until doc.length) resSent = resSent ::: doc(i)
              return Array(filter(questionSent), filter(methodSent), filter(resSent))
            }
          }
        }
      }
  
      // divide the doc using word2vec
      // calculate the vector of each sentence
      var sentVec = new Array[Array[Double]](doc.length)

      for (i <- 0 until doc.length) sentVec(i) = calSentVec(doc(i))
      // search the turning point
      var dis = new Array[Double](doc.length-1)
      for (i <- 0 until sentVec.length-1) {
        for (j <- 0 until vecDim) 
          dis(i) += pow(sentVec(i)(j) - sentVec(i + 1)(j), 2)
        dis(i) = sqrt(dis(i))
      }
      var quesEnd = 1
      if (dis.length != 0) {
        var half = (dis.max - dis.min) / 2
        for (i <- 1 until dis.length) {
          if (dis(i) > dis.min + half && quesEnd != 0) quesEnd = i
        }
      }
      // search the result sentence
      for (i <- quesEnd until doc.length) {
        var isResult = false
        for (w <- doc(i)) {
          if (signwordsRes.exists(x => x == w)) isResult = true
        }
        if (isResult) resSent = doc(i) ::: resSent
        else methodSent = doc(i) ::: methodSent
      }
      
      for (i <- 0 until quesEnd) questionSent = questionSent ::: doc(i)
      return Array(filter(questionSent), filter(methodSent), filter(resSent))
    }
  
    def calSentVec(sent: List[Int]): Array[Double] = {
      var sentVec = new Array[Double](vecDim)
      for (w <- sent) {
        var j: Int = -1
        for (i <- 0 until signwordsQues.length) {
          if (signwordsQues(i)._1 == w) j = i
        }
        if (j != -1) {
          for (i <- 0 until vecDim) 
            sentVec(i) += (vecDim + 5 * signwordsQues(j)._2) * wordVec(w)(i)
        } else {
          for (i <- 0 until vecDim) sentVec(i) += wordVec(w)(i)
        }
      }
      for (i <- 0 until sentVec.length) sentVec(i) /= sent.length
      return sentVec
    }

    def filter(doc: List[Int]): List[Int] = {
      var newDoc: List[Int] = List()
      for (w <- doc) {
        if (!stopwords.exists(sw => sw == w)) newDoc = w :: newDoc
      }
      return newDoc
    }
  }
}
