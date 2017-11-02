#!/usr/bin/python
#coding:UTF-8

import gensim
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint
from six import iteritems
import sys
import os
import numpy as np


stoplist = []
docs_name = ""

"""
加载词典，如果是新文档直接生成词典并保存
"""
def load_data(file_name, isNew = True):
    if(isNew):
        stoplist = set('for a of the and to in'.split())
        dictionary = corpora.Dictionary(line.lower().split()[1:] for line in open(file_name))

        ##为保持与原来实验数据一致，在此不再去掉停用词和只出现一次的词
        # stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
        # once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
        # dictionary.filter_tokens(stop_ids + once_ids)

        dictionary.save("../data/dict_" + file_name)
    else:
        dictionary = corpora.Dictionary.load("../data/dict_" + file_name)
    return dictionary


class MyCorpus(object):
    def __iter__(self):
        for line in open(docs_name):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

if __name__ == "__main__":
    docs_name = sys.argv[1]
    dictionary = load_data(docs_name)
    print dictionary

    #词频矩阵 目前的文档ID尚未能够对上
    if os.path.exists("../data/corpus_" + docs_name):
        corpus = corpora.MmCorpus("../data/corpus_" + docs_name)
    else :
        corpus = MyCorpus()
        corpora.MmCorpus.serialize("../data/corpus_" + docs_name, corpus)

    # if(os.path.exists("../data/model_" + docs_name)):
    #     lda = models.LdaModel.load("../data/model_" + docs_name, )
    lda = models.LdaModel(corpus=corpus, num_topics=100, id2word=dictionary)


    #保存wz矩阵
    topicAll = lda.get_topics()
    # print topicAll
    if not os.path.exists("../data/Matrix_wz_" + docs_name + ".npy"):
        np.save("../data/Matrix_wz_" + docs_name, topicAll)


    #求出文档在某些最有可能的主题的概率
    doc_topic = []
    for bow in corpus:
        doc_topic.append(lda.get_document_topics(bow=bow))
    if not os.path.exists("../data/Matrix_zd_" + docs_name):
        fout = open("../data/Matrix_zd_" + docs_name, "w")
        for doc in doc_topic:
            fout.writelines(str(doc) + "\n")
        fout.close()

