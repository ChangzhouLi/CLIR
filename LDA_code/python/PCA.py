# coding: utf-8
import os
import findspark
findspark.init()
import numpy as np
from pyspark.ml.feature import PCA
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.sql import SQLContext
import numpy as np
import time
from GetCorpus import MAX_DOCUMENTS

MONTH = time.localtime(time.time())[1]
DAY = time.localtime(time.time())[2]
HOUR = time.localtime(time.time())[3]
MINUTE = time.localtime(time.time())[4]

def pca(sc, df, dim=10, path='./pca.txt'):
    print('in!')
    start = time.time()
    
    # 直接读pca
    if (MAX_DOCUMENTS == 100000):
        path = '/root/final/pca_result/pca_10w.txt'
    elif (MAX_DOCUMENTS == 10000):
        path = '/root/final/pca_result/pca_1w.txt'
    elif (MAX_DOCUMENTS == 1000):
        path = '/root/final/pca_result/pca_1000.txt'
    
    pca_vecs = {}
    if not os.path.exists(path):
        # 将词向量模型转为DataFrame
        sql_context = SQLContext(sc)
        # 对词向量进行PCA降维
        words = list(df.select('word').collect())
        words = [x[0] for x in words]
        
        pca_result = PCA(k=dim, inputCol='vector', outputCol='new_vec').fit(df).transform(df).collect()
        for i in range(len(pca_result)): pca_vecs[words[i]] = np.array(pca_result[i].new_vec)
        
        # 将结果写回本地文件
        with open(path, 'w') as f:
            for word in pca_vecs:
                f.write(word+' ')
                for value in pca_vecs[word]: f.write(str(value)+',')
                f.write('\n')
    else:
        with open(path, 'r') as f:
            pca_vecs = {}
            for line in f:
                sub = line.split(' ')
                # print(sub)
                vecs = []
                for value in sub[1][0:-1].split(','):
                    if value != '': vecs.append(float(value))
                pca_vecs[sub[0]] = np.array(vecs)
    print("pca个数：", len(pca_vecs))
    return pca_vecs
