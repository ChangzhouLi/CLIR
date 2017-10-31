import findspark
findspark.init()
import pyspark
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SQLContext

#sc = pyspark.SparkContext()

#corpus为函数get_corpus函数得到的rdd格式语料

def train_word2vec(sc,corpus):
    
    '''
    #若语料形式为列表`
    localDoc=[]
    for i in range(len(corpus)):
        text=corpus[i][0]
        print(text)
        localDoc.append(text)
    print(localDoc)
    print(corpus)
    doc = sc.parallelize(localDoc).map(lambda line: line.split(" "))
    print(doc)
    '''
    sql_context = SQLContext(sc)
    corpus = corpus.map(lambda x: (x, 1))
    corpus = sql_context.createDataFrame(corpus, ['sentence'])
    # print(corpus.collect())
    # model = Word2Vec().setVectorSize(400).setSeed(42).setMinCount(1).fit(corpus)
    model = Word2Vec(vectorSize=400, seed=42, inputCol="sentence", outputCol="origin_vec").fit(corpus)
    # model.getVectors().show()
    # return a dataframe
    return model.getVectors()

# if __name__ == '__main__':
#     #corpus=[['a b c'],['d e f']]
#     mymodel=train_word2vec(sc,corpus)
#     #print(mymodel['a'])

        