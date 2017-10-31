package com.sysu.nlp
import java.io._

class Transformer(dic: Array[String]) {
    
    def transform(result: Array[List[Int]], name: String) {
        val writer = new PrintWriter(new File(name))
        // println(result(0)(0)(0))
        // println("result len: "+result.length)
        // println("result(0): "+result(0).length)
        for ( i <- 0 to result.length - 1 ) {
            var doc = ""
            // println("i: "+i)
            for (j <- 0 to result(i).length - 1 ) {
                // println("j: "+j)
                // for (k <- 0 until result(i)(j).length) doc += (dic(result(i)(j)(k)) + ' ')
                doc += (dic(result(i)(j)) + ' ')
                // doc += ','
            }
            writer.write(doc+"\n")
            doc = ""
        }   
        writer.close()
    }
}