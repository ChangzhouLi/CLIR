package com.sysu.nlp

import scala.collection.mutable.HashMap
import scala.io.Source
import scala.util.Sorting.stableSort

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD


class KGenerator(wordCount: Int, coff: Int) extends java.io.Serializable {

  def generate(corpus: RDD[List[Int]]): Int = {
    val tf = corpus.map(doc => count(doc))
    val idf = new IDF().fit(tf)
    val tfidf = idf.transform(tf)
    val k = tfidf.flatMap{
      case SparseVector(size, indices, values) => 
        indices.zip(values).sortBy(-_._2).take(5).toSeq
    }.distinct().count()
    return k.toInt / coff
  }
  
  def count(doc: List[Int]): Vector = {
    val termFrequencies = HashMap.empty[Int, Double]
    doc.foreach { i => termFrequencies.put(i, termFrequencies.getOrElse(i, 0.0) + 1.0) }
    return Vectors.sparse(wordCount, termFrequencies.toSeq)
  }
}