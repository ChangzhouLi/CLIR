# coding: utf-8
import pyspark
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import Word2VecModel
import time
import numpy as np
from numpy import array
import jieba
import codecs
import re


def doc_divide_all(doc, model, dim=10):
    seg_doc = []
    doc_all_str = ''
    for sentence in doc:
        sentence_all_str = ' '.join(sentence)
        doc_all_str += sentence_all_str
    pa = re.compile(r"目的.*方法.*[结论|结果].*")
    if (re.match(pa, doc_all_str)):
        seg_doc += doc_divide_regular(doc, model)
    else:
        seg_doc += doc_divide_irregular(doc, model, dim)
    return seg_doc

def mergeDividedCorpus(divided_corpus):
    divided_corpus_merge = []
    for document in divided_corpus:
        document_new = []
        for segment in document:
            document_new.append(sum(segment, []))
        divided_corpus_merge.append(document_new)
    return divided_corpus_merge

def doc_divide_regular(doc, model):
    seg_doc = []
    problem_sents = []
    method_sents = []
    result_sents = []
    problem_word = "目的"
    method_word = "方法"
    result_word1 = "结果"
    result_word2 = "结论"
    problem_index = 0
    method_index = 0
    result_index = 0
    # 第一部分目的分类
    # 找到目的方法结论索引
    for index in range(len(doc)):
        if (problem_word in doc[index]):
            problem_index = index
        elif (method_word in doc[index]):
            method_index = index
        elif (result_word1 in doc[index]) or (result_word2 in doc[index]):
            result_index = index
    problem_sents = doc[problem_index:method_index]
    method_sents = doc[method_index:result_index]
    result_sents = doc[result_index:]
    # print("problem_sents: ", problem_sents)
    # print("method_sents: ", method_sents)
    # print("result_sents: ", result_sents)
    seg_doc.append(problem_sents)
    seg_doc.append(method_sents)
    seg_doc.append(result_sents)
    # print("seg_doc in regular: ", seg_doc)
    return seg_doc

def doc_divide_irregular(doc, model, dim=10):
    seg_doc = []
    fin_ques_signwords = 'ques_signwords.txt'
    fin_res_signwords = 'result_sign_final.txt'
    # LEN = model
    #model = pca(sc, word_vecs)
    '''
    model = {'b': array([ 1.65313651, -0.11056335,  0.01918347, -0.04302083, -0.00726535,
        0.0230763 ,  0.02722933,  0.00276523, -0.03179846, -0.00588288]), 'c': array([-0.46881218, -0.24338962,  0.01918347, -0.04302083, -0.00726535,
        0.0230763 ,  0.02722933,  0.00276523, -0.03179846, -0.00588288]), 'a': array([ 0.3381433 ,  0.65519747,  0.01918347, -0.04302083, -0.00726535,
        0.0230763 ,  0.02722933,  0.00276523, -0.03179846, -0.00588288])}
    '''
    signwords_ques = {}
    s = 0
    with open(fin_ques_signwords, 'r', encoding='utf-8') as f:
        for line in f:
            sub = line.replace('\r', '').replace('\n', '').split(' ')
            signwords_ques[sub[0]] = float(sub[1])
            s += float(sub[1])
        for word in signwords_ques: signwords_ques[word] /= s
        
    result_sign = open(fin_res_signwords, 'r', encoding = 'utf-8')
    result_signs = [ w.strip() for w in result_sign ]
    signwords_res = []
    for item in result_signs:
        signwords_res.append(''.join(item).split()[0])
    # print("signwords_res: ", signwords_res)
    
    #for doc in docs:
    vecs = []
    validSents = []
    # 每个句子的平均词向量
    # print("doc: ", doc)
    for sent in doc:
        # print("sent: ", sent)
        wordVecs = []
        for word in sent:
            if word in model.keys():
                if word in signwords_ques:
                    wordVecs.append((dim+5*signwords_ques[word])*model[word])    #　ｐｃａ有无
                else:
                    wordVecs.append(model[word])
        if len(wordVecs) != 0: 
            validSents.append(sent)
            vecs.append(sum(wordVecs)/len(wordVecs))
    
    #第一部分问题分类
    vecs = np.array(vecs)
    dis = []
    #print(vecs)
    for i in range(len(vecs)-1): dis.append(np.sqrt(sum((vecs[i]-vecs[i+1])**2)))
    if (len(dis) != 0):
        half = (max(dis)-min(dis))/2
        text = []
        domainSents = [validSents[0]]
        k = 0
        for i in range(len(dis)):
            # print("第{}句-第{}句: {}".format(i+1, i, dis[i]))
            if dis[i] > min(dis)+half and i > 2 and k == 0:
                # print("domainSents: ", domainSents)
                text.append(domainSents)
                domainSents = []
                k = 1
            domainSents.append(validSents[i+1])
        if k == 1:
            domainSents.append(validSents[i+1])
        if len(domainSents) != 0: text.append(domainSents)
    
    # 第三部分结果分类
    # global seg_doc
    if ("text" in locals().keys()):
        if len(text) >= 2:
            # i = 0
            t1 = []
            t2 = []
            # print("text: ", text)
            method_result = text[1]
            # print("method_result: ", method_result)
            for sentence in method_result:
                isResult = False
                for word in sentence:
                    # print("word:", word)
                    if word in signwords_res:
                        # print("result: ", sentence)
                        t2.append(sentence)
                        isResult = True
                        break
                if isResult == False:
                    # print("method: ", sentence)
                    t1.append(sentence)
                # i = i + 1
            texts = []
            texts.append(text[0])
            texts.append(t1)
            texts.append(t2)
            seg_doc.append(texts)
        else:
            texts = [text[0], [], []]
            seg_doc.append(texts)
    # print("seg_doc  in irregular", seg_doc)
    return sum(seg_doc, [])

