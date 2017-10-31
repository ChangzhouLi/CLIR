# coding: utf-8
import os
import re
import pyspark
import jieba
import codecs
import re
import random
from pyspark import SparkConf, SparkContext
import pymongo
import time
from time import time


# 文章总数
MAX_DOCUMENTS = 10000
TIMES = 64-1

#去停用词且去数字
def filterStopWords(stopwordslist, doc):
    doc_new = []
    for w in doc:
        if len(w.strip()) > 1 and w.strip() not in stopwordslist: doc_new.append(w.strip())
    return doc_new


def docs_cut(docs, stopwords, stopw):
    pa = re.compile('(\d)*')
    words = docs.map(lambda line: re.sub(pa, '', line)).map(lambda line: jieba.lcut(line))
    
    #去停用词
    if stopw:
        words = words.map(lambda w: filterStopWords(stopwords,w))
    return words.collect()


def st_cut(doc, stopwords, stopw):
    pa = re.compile('(\d)*')
    for i in range(len(doc)):
        doc[i] = re.sub(pa, '', doc[i]).strip()
        doc[i] = jieba.lcut(doc[i])
    
    doc_new = doc
    if stopw:
        doc_new = []
        for sent in doc:
            sent_new = []
            for w in sent: 
                if len(w) > 1 and w not in stopwords: sent_new.append(w)
            doc_new.append(sent_new)
    return doc_new
    

def docs_st_cut(docs_st, stopwords, stopw):
    words = docs_st.map(lambda doc: st_cut(doc, stopwords, stopw))
    return words.collect()


def get_corpus(sc, abst, abst_st, stopwordfile, stopw):
    # 从本地读入语料
    start = time()
    docs = []
    with codecs.open(abst, 'r', encoding='utf-8') as f:
        docs = [doc.strip() for doc in f.readlines()[:MAX_DOCUMENTS]]
    
    print('amounts of doc:', len(docs))
    
    docs_sents = []
    with open(abst_st, 'r', encoding='utf-8') as f:
        pa = re.compile("\[|\]|'")
        i = 0
        for each in f:
            tem = re.sub(pa, '', each).strip().split(',')
            docs_sents.append(tem)
            i += 1 
            if i > MAX_DOCUMENTS: break
    end = time()
    print('reading cost %f'%(end-start))
    docs_sents = docs_sents * TIMES
    
    stopwords = []
    #加载去停用词表
    if stopw:
        with codecs.open(stopwordfile, 'r', encoding='utf-8') as f:
            stopwords = [w.strip() for w in f.readlines()]
 
    #搜索引擎分词模式
    rdd_abs = sc.parallelize(docs)
    token_abs = docs_cut(rdd_abs, stopwords, stopw)

    rdd_st = sc.parallelize(docs_sents)
    token_st = docs_st_cut(rdd_st, stopwords, stopw)

    return token_abs, token_st
