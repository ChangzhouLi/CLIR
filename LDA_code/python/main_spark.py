# coding:utf-8
from pyspark import SparkConf, SparkContext
from DocDivide import doc_divide_all, mergeDividedCorpus
from GetClassNum_spark import get_class_num
from GetCorpus import get_corpus, MAX_DOCUMENTS, TIMES
from LdaModel_spark import lda_model
from PCA import pca, MONTH, DAY, HOUR, MINUTE
from Predict import predict, WriteResult, SortTopicParalle
from TrainWord2Vec import train_word2vec
from operator import itemgetter
import time
import os
from time import time
from time import clock
import numpy as np
import pickle

# *****************
# MAX_DOCUMENTS
# _DIM
# _Iters
# *****************
_Iters = 1
_DIM = 10

if __name__ == '__main__':
    # file.write("total document: "+str(MAX_DOCUMENTS)+"\n")
    
    # 初始化Spark环境
    cores_max = 24*64 - 24*1
    parallelism = 5*cores_max
    conf = SparkConf()\
        .setAppName('Testing-10-10')\
        .setMaster('spark://cn7839:7077')\
        .set('spark.driver.memory', '70g')\
	.set('spark.executor.memory', '20g')\
	.set('spark.executor.cores', '24')\
	.set('spark.cores.max', str(cores_max))\
        .set('spark.driver.maxResultSize', '0')\
	.set('spark.akka.frameSize', '512')\
        .set('spark.default.parallelism', str(parallelism))\
        .set('spark.executorEnv.PYTHONHASHSEED', '123')
    sc = SparkContext(conf=conf)
    print('-------------------------------Spark Config Attribute------------------------------------')
    print(conf.getAll())
    print('\n')
   
    # file = open('RecordTime'+str(cores_max)+'.txt', 'w', encoding = 'utf-8')
    # file.write('corpus size: ' + str(MAX_DOCUMENTS*TIMES) + '\n')   

    print("reloading corpus...")
    start_corpus = time()
    # corpus是整篇文章，corpus_sents是文章分句
    corpus, corpus_sents = get_corpus(sc, abst='abstract_17k.txt', abst_st="st_17k.txt", stopwordfile='stopwords.txt', stopw = True)
    end_corpus = time()
    
    print("loading corpus finished!")
    print('time of corpus %fs'%(end_corpus-start_corpus))
    # file.write('time of corpus: ' + str(end_corpus-start_corpus) + "s\n")
    
    print('--------------MAX_DOCUMENTS--------------: ', MAX_DOCUMENTS)
    
    # 词向量与ＰＣＡ, 如果想要跳过训练词向量部分则将path的值设定为pca.txt的路径
    path = './pca.txt'
    if not os.path.exists(path):
        print("training word2vec...")
        # 训练词向量(非RDD)
        start_w2v = time()
        word_vecs = train_word2vec(sc, sc.parallelize(corpus))
        end_w2v = time()
        print("training w2v finished!")
        print('time of w2v %fs'%(end_w2v-start_w2v))
        # file.write('time of w2v: ' + str(end_w2v-start_w2v) + "s\n")
        
        print("reducing dimension with pca...")
        # 对词向量降维(非RDD)
        start_pca = time()
        pca_word_vec = pca(sc,word_vecs, dim=_DIM, path=path)
        end_pca = time()
        print("pca done!")
        print('time of pca %fs'%(end_pca-start_pca))
        # file.write('time of pca: ' + str(end_pca-start_pca) + "s\n")
    else:
        print('MAX_DOCUMENTS: ', str(MAX_DOCUMENTS)+'~'+path)
        pca_vecs = {}
        with open(path, 'r', encoding='utf-8') as f:
            pca_vecs = {}
            for line in f:
                sub = line.split(' ')
                # print(sub)
                vecs = []
                for value in sub[1][0:-1].split(','):
                    if value != '': vecs.append(float(value))
                pca_vecs[sub[0]] = np.array(vecs)
        pca_word_vec = pca_vecs
        print('loaded pca from txt')
    
    # test()
    # print('in for...')
    start = time()
    # for i in range(5):
        # print('the %d-th for...'%i)
    # print("dividing corpus...")
    # 将语料库中每篇文档分段
    start_divide = time()
    corpus_sents_para = sc.parallelize(corpus_sents)
    divided_corpus = corpus_sents_para.map(lambda doc: doc_divide_all(doc, pca_word_vec, dim=_DIM)).collect()
    divided_corpus_merge = mergeDividedCorpus(divided_corpus)
    end_divide = time()
    print("dividing corpus done!")
    print('time of divide %fs'%(end_divide-start_divide))
    # file.write('time of divide: ' + str(end_divide-start_divide) + "s\n")
    
    divided_question_merge = [item[0] for item in divided_corpus_merge if len(item) >= 1]
    divided_method_merge = [item[1] for item in divided_corpus_merge if len(item) >= 2]
    # 获取主题数
    # 问题
    print("calculating topic numbers...")
    start_num = time()
    # print('question', divided_question_merge)
    # print('method', divided_method_merge)
    question_num, word_count_ques= get_class_num(sc, divided_question_merge, pca_word_vec)
    # 方法
    method_num, word_count_meth= get_class_num(sc, divided_method_merge, pca_word_vec)
    end_num = time()
    print("calculating topics done!")
    print('time of geting num %fs'%(end_num-start_num))
    # file.write('time of num: ' + str(end_num-start_num) + "s\n")
    
    end = time()
    # end = clock()
    print('first&step spend : ', (end-start))
    print('--------first&second step done-------!!!!')
    # os._exit(1)
    
    print("training LDA model...")
    start_lda = time()
    # 计算LDA主题模型
    # 新增迭代次数作为参数
    lda_question = lda_model(sc, divided_question_merge, num_topics=question_num, w_c=word_count_ques, iters=_Iters)
    lda_method = lda_model(sc, divided_method_merge, num_topics=method_num, w_c=word_count_meth, iters=_Iters)
    # print('lda_question : ', lda_question)
    # print('lda_method : ', lda_method)
    end_lda = time()
    print("LDA done!")
    print('time of training lda %fs'%(end_lda-start_lda))
    # file.write('time of lda: ' + str(end_lda-start_lda) + "s\n")
    os._exit(2)

    print("predicting corpus with lda results...")
    # 对语料库每篇文档进行预测
    start_predict = time()
    result_question = predict(sc, divided_question_merge, lda_question, pca_word_vec, dim=_DIM)

    result_method = predict(sc, divided_method_merge, lda_method, pca_word_vec, dim=_DIM)
    end_predict = time()
    print('time of predict %fs'%(end_predict-start_predict))
    print("predicting finished!")
    # file.write('time of predict: ' + str(end_predict-start_predict) + "s\n")
    
    end = time()
    print('spent  %fs'%(end-start))
    # file.write('all cost: ' + str(end-start) + "\n")
    # --------------------------writing time to local-------------------------------
    file = open("(para"+")record_time_"+str(MAX_DOCUMENTS)+".txt",'w',encoding="utf-8")
    file.write("total documents: "+str(MAX_DOCUMENTS)+"\n")
    file.write("-----------!! parallelism !! -------- = "+str(parallelism)+"\n")
    file.write("time of corpus: "+ str(end_corpus-start_corpus)+"s\n")
    file.write("time of num: "+ str(end_divide-start_divide)+"s\n")
    file.write("time of lda: "+ str(end_lda-start_lda)+"s\n")
    file.write("time of predict: "+ str(end_predict-start_predict)+"s\n")
    file.write("all cost: "+ str(end-start)+"s\n")
 
    file.close()
