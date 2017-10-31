package com.sysu.nlp

import Array._
import java.util.Date
import scala.io.Source
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object Main {
    def main(args: Array[String]) {
        var times = 62
        var cores = "1488"
        if (args.length == 1) cores = args(0)
        println("--------------loading corpus!!---------------")
        val txt = new TxtLoader(corpus_times=times)
        println("corpus size: " + txt.corpus.length)
        
        val conf = new SparkConf()
            .setAppName("PaperRank")
            .setMaster("spark://cn7839:7077")
            .set("spark.executor.memory", "20g")
            .set("spark.driver.maxResultSize", "0")
            .set("spark.executor.cores", "24")
            .set("spark.cores.max", cores)
            .set("spark.default.parallelism", "4464")

        val sc = new SparkContext(conf)
        println(conf.getAll)
        
        val start = new Date()
        
        val divideStart = new Date()
        val divider = new DocDivider(txt.wordVec, txt.pattern, txt.ques, txt.res, txt.stopwords)
        println("-------------dividing corpus!!---------------")
        divider.divide(sc.parallelize(txt.corpus))
        val divideEnd = new Date()
        println("divide time: " + (divideEnd.getTime()-divideStart.getTime())/1000.0)
        
        println("-------------calculating topic number--------")
        val classNumStart = new Date()
        var generator = new KGenerator(txt.wordVec.length, 200)
        val k = generator.generate(sc.parallelize(divider.questionText))
        println("topics: "+k)
        val classNumEnd = new Date()
        println("classNum time: " + (classNumEnd.getTime()-classNumStart.getTime())/1000.0)
        
        println("-------------training LDA---------------------")
        val ldaStart = new Date()
        var quesTopicModel = Lda.train(sc, divider.questionText, k, txt.wordVec.length)
        // var methodTopicModel = Lda.train(sc, divider.methodText, 380, txt.wordVec.length, optimizer="em")
        val ldaEnd = new Date()
        println("lda time: " + (ldaEnd.getTime()-ldaStart.getTime())/1000.0)
        
        // val predictStart = new Date() 
        // println("-------------predicting documents-------------")
        // var predicter = new Predicter(txt.wordVec, quesTopicModel)
        // var result = predicter.predict(sc.parallelize(divider.questionText))
        // val predictEnd = new Date()
        // println("predict time: " + (predictEnd.getTime()-predictStart.getTime())/1000.0)
        
        val end = new Date()
        println("spent time: "+(end.getTime()-start.getTime())/1000.0+"s")
    }
}