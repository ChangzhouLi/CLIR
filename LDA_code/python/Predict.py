import numpy as np
import os
import findspark
findspark.init()
from pyspark import SparkContext
import time

#calculating vectors for each paragraphs in each abstracts
def para_vec_fun(abstract, pca_word_vecs, dim):
    # 每次只传入问题或者方法
    para = abstract
    SumWordVec = np.zeros(dim)
    i = 0
    pca_word_vecs_keys = pca_word_vecs.keys()
    for item in para:
        if item in pca_word_vecs_keys:
            i = i + 1
            SumWordVec = SumWordVec + np.array(pca_word_vecs[item])
    para_vec = SumWordVec / i
    return para_vec

#calculating topic vectors
def topic_vec(topic, pca_word_vecs, dim):
    # print('calculating topic vectors...')
    tpvec = np.zeros(dim)
    j = 0
    pca_word_vecs_keys = pca_word_vecs.keys()
    for word in topic:
        if word in pca_word_vecs_keys:
            j = j + 1
            tpvec = tpvec + np.array(pca_word_vecs[word]) * topic[word]
    tpvec = tpvec / j
    return tpvec

#calculating similarity
def similarity(abstract, topic_vecs):
    # print('calculating similarity...')
    cosine_max = -1
    cosine_topic = -1
    index = 0
    for topic in topic_vecs:
        a = abstract
        b = topic
        cosine = a.dot(b)/np.sqrt(a.dot(a)*b.dot(b))
        if (cosine > cosine_max):
            cosine_max = cosine
            cosine_topic = index
        index += 1
    return (cosine_topic, cosine_max)

def predict(sc, divided_corpus, lda, pca_word_vecs, dim=10):
    print('calculating similarity......')

    all_vecs = sc.parallelize(divided_corpus).map(lambda item: para_vec_fun(item, pca_word_vecs, dim))
    print('geting documents vecs....')
    topic_vecs = sc.parallelize(lda).map(lambda item: topic_vec(item, pca_word_vecs, dim)).collect()
    print('geting topics vecs....')
    similar = all_vecs.map(lambda item:  similarity(item, topic_vecs))
    si = similar.collect()
    print('prediction done!')
    # print("si", si)
    return si
    


# Sorting topics without using Spark     
def SortTopic(result):
    # print("Sorting topics without using Spark .....")
    rank_all = []
    for abstract in result:
        ranking = []
        for para in abstract:
            rank = []
            sort = sorted(para,reverse = True)
            for item in sort:
                rank.append(int(para.index(item)+1))
            ranking.append(rank)
        rank_all.append(ranking)
    # print("Topics sorted.")
    return rank_all
    

#Sorting topics using Spark    
def SortTopicParalle(sc,result):
    # print("Sorting topics with Saprk.....")
    def RankParalle(abstract):
        ranking = []
        para = abstract
        rank = []
        sort = sorted(para,reverse = True)
        for item in sort:
            rank.append(int(para.index(item)+1))
        ranking.append(rank)
        return ranking
    ranking = sc.parallelize(result).map(RankParalle)
    ranking = ranking.collect()
    # print("Topics sorted.")
    return sum(ranking, [])


#Writing files to local    
def WriteResult(result,rank_all):
    print('writing files...')
    start = time.time()
    with open('/root/final/predict_result/output_'+str(mon)+'-'+str(day)+'.txt','a', encoding='utf-8') as file:
        for abstract in result:
            file.write('Abstract No.'+ str(result.index(abstract)) + '\n')
            for para in abstract:
                file.write(str(para)+'\n')
            file.write(' '+'\n')
        for ranking in rank_all:
            file.write('Abstract No.'+ str(rank_all.index(ranking))+'\n')
            for rank in ranking:
                file.write(str(rank)+'\n')    
            file.write(' '+'\n')
        file.write("------------------------------------------------------------------")
        file.close()
    end = time.time()
    print('Writing done.')
    print('writing spent: ', end-start)



    