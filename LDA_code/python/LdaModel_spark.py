# latest 8-26
import codecs
import pickle
import time
import findspark
findspark.init()
import pyspark
import numpy as np
from py4j.java_gateway import java_import
from pyspark.mllib.clustering import LDA
from pyspark.mllib.linalg import Vectors, SparseVector
from pyspark.mllib.common import callJavaFunc, _py2java, _java2py
from pyspark.serializers import PickleSerializer
from myfile import HashingTF
import os

# 去停用词
def filt(stopwords, text):
    tem = []
    for word in text:
        if word not in stopwords: tem.append(word)
    return tem

def filterStopWords(sc, f, corpus):
    stopwords = codecs.open(f, 'r', encoding='utf-8').readlines()
    stopwords = [w.strip() for w in stopwords]
    rdd = sc.parallelize(corpus)
    docs = rdd.map(lambda x: filt(stopwords, x)).collect()
    return docs


def lda_model(sc, corpus, num_topics, w_c, iters=20):
    corpus = filterStopWords(sc, 'stopwords_lda.txt', corpus)
    rdd = sc.parallelize(corpus)
    htf = HashingTF(2*w_c)
    htf_bow = htf.transform(rdd)
    # hashing对应词典
    dic = htf.getDict(rdd).collect()
    data = index(htf_bow.collect())
    rdd_data = sc.parallelize(data)
    
    print('start training LDA!')
    model = LDA.train(rdd_data, k=num_topics, maxIterations=iters, optimizer='em')
    
    print('topic is generating...')
    lda_java_model = model._java_model
    # use spark1.6 module 
    func = getattr(model._java_model, 'describeTopics')
    result = func(_py2java(sc, 10))
    topics = _java2py(sc, sc._jvm.com.sysu.sparkhelper.LdaHelper.convert(result))
    # print(topics[0])

    print('generating final result...')
    vocab_size = model.vocabSize()
    topics_list = make_data(topics, vocab_size, dic, num_topics)
    
    return topics_list
    
def index(corpus):
    # 依顺序添加序号s
    for i in range(len(corpus)):
        corpus[i] = [i, corpus[i]]
    return corpus

# for describeTopics()
def make_data(topics, vocab_size, dic, k):
    dict1 = {}
    for each in dic:
        dict1.update(each)
    # print(dict1)
    result = []
    for each in topics:
        dict2 = {}
        word = each[0]
        weight = each[1]
        for i in range(len(word)):
            if str(word[i]) in dict1: dict2[dict1[str(word[i])]] = weight[i]
        result.append(dict2)

    with open('./spark_result.txt', 'w', encoding='utf-8') as f:
        for each in result:
            f.write(str(each)+'\n')
    return result
    
    
    
    
    
    
