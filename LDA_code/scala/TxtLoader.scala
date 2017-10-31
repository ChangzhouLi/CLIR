package com.sysu.nlp
import scala.io.Source
import Array._

class TxtLoader(
    path_corpus: String="/cor_index_neway.txt",
    path_wv: String="/pca_index.txt",
    path_ques: String="/ques_index.txt",
    path_res: String="/result_index.txt",
    path_pattern: String="/patterns_index.txt",
    path_stopwords: String="/stop_index.txt",
    path_dic: String="/all_dict.txt",
    corpus_times: Int = 20) {
        
    def readCorpus(path: String, stopwords: List[Int]) : List[List[List[Int]]] = {
        val corpus_ = Source.fromURL(getClass.getResource(path)).mkString("").split("\n")
        val length = corpus_.length
        
        var corpus: List[List[List[Int]]] = List()
        for ( i <- 0 until 10000 ) {
            var doc: List[List[Int]] = List()
            var tem  = corpus_(i).replace("[[", "").replace("]]", "").split("\\], \\[")
            for ( j <- tem ) {
                var doc_divide: List[Int] = List()
                for ( p <- j.split(", ") ) {
                    // 去停用词，“”
                    if (p != "" && !stopwords.contains(p.toInt)) {
                        doc_divide = doc_divide ::: List(p.toInt)
                    }
                    // 去停用词
                    // if ( !stopwords.contains(p.toInt) ) {
                        // doc_divide = doc_divide ::: List(p.toInt)
                    // }
                }
                doc = doc ::: List(doc_divide)
            }
            corpus = corpus ::: List(doc)
        }
        var copyCorpus: List[List[List[Int]]] = List()
        for (i <- 0 until corpus_times) copyCorpus = copyCorpus ::: corpus
        return copyCorpus
    }
    
    def readWordVec(path: String, dim: Int = 10) : Array[Array[Double]] = {
        // val wv = Source.fromFile(path).mkString("").split("\n")
        val wv = Source.fromURL(getClass.getResource(path)).mkString("").split("\n")
        val length = wv.length
        // println("length wv: "+length)
        var wordVec = ofDim[Double](length, dim)
        for ( i <- 0 to length - 1 ) {
            var q = 0
            for ( j <- wv(i).replace(" ", "").split(",") ) {
                // 排除向量为空的情况
                if ( j == "" ){
                    wordVec(i)(q) = 0
                }
                else {
                    wordVec(i)(q) = j.toDouble
                }
                q += 1
            }
        }
        return wordVec
    }
    
    def readSignQues(path: String) : Array[Tuple2[Int, Double]] = {
        // val sign_ques = Source.fromFile(path).mkString("").split("\n")
        val sign_ques = Source.fromURL(getClass.getResource(path)).mkString("").split("\n")
        val length = sign_ques.length
        var signQues = new Array[Tuple2[Int, Double]](length)
        
        for ( i <- 0 to length - 1 ) {
            val x = sign_ques(i).split(" ")
            signQues(i) = (x(0).toInt, x(1).toDouble)
        }
        return signQues
    }
    
    def readSignRes(path: String) : Array[Int] = {
        // val sign_res = Source.fromFile(path).mkString("").split("\n")
        val sign_res = Source.fromURL(getClass.getResource(path)).mkString("").split("\n")
        val length = sign_res.length
        var signRes = new Array[Int](length)
        
        for ( i <- 0 to length - 1 ) {
                signRes(i) = sign_res(i).toInt
        }
        return signRes
    }
    
    def readPattern(path: String) : Array[Array[Int]] = {
        // val pattern_ = Source.fromFile(path).mkString("").split("\n")
        val pattern_ = Source.fromURL(getClass.getResource(path)).mkString("").split("\n")
        val length = pattern_.length
        var pattern = new Array[Array[Int]](length)
        
        for ( i <- 0 to length - 1 ) {
            val tem = pattern_(i).split(',')
            pattern(i) = new Array(tem.length)
            for ( j <- 0 until tem.length ) {
                pattern(i)(j) = tem(j).toInt
            } 
        }
        return pattern
    }
    
    def readStopwords(path: String) : List[Int] = {
        // val stopwords_ = Source.fromFile(path).mkString("").split("\n")
        val stopwords_ = Source.fromURL(getClass.getResource(path)).mkString("").split("\n")
        val length = stopwords_.length
        var stopwords: List[Int] = List()
        
        for ( i <- 0 to length - 1 ) {
            stopwords = stopwords ::: List(stopwords_(i).toInt)
        }
        return stopwords
    }
    
    def readDic(path: String) : Array[String] = {
        val dic_ = Source.fromURL(getClass.getResource(path)).mkString("").split("\n")
        val length = dic_.length
        var dic = new Array[String](length)
        
        for ( i <- 0 to length - 1 ) {
            dic(i) = dic_(i)
        }
        return dic
    }
    
    val stopwords = readStopwords(path_stopwords)
    val corpus = readCorpus(path_corpus, stopwords)
    val wordVec = readWordVec(path_wv)
    val ques = readSignQues(path_ques)
    val res = readSignRes(path_res)
    val pattern = readPattern(path_pattern)
    // val dic = readDic(path_dic)
    
}