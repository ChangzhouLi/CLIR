# 一定要加下面两句话，不然就会报错
from time import time
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import IDF
import myfile
from pyspark.mllib.linalg import Vectors
from operator import itemgetter
from sklearn.cluster import AffinityPropagation
APP_NAME = "tfidf"

def filterStopWords_tf(document, stopwords):
  for word in document:
    if (word not in stopwords):
      document.remove(word)
  return document

def transListAndTop5(document_idf_data):
  # 去除idf值为０的词语
  # document_idf_data = document_idf_data.toArray()
  # document_idf_data_list = [(a,document_idf_data[a]) for a in range(len(document_idf_data)) if document_idf_data[a] != 0]
  indices = document_idf_data.indices
  values = document_idf_data.values
  document_idf_data_list = [(indices[i], values[i]) for i in range(len(indices))]
  # map里面不能用sortBy(只能在全局使用)
  document_idf_data_list_sorted = sorted(document_idf_data_list, key=itemgetter(1), reverse=True)
  # 找到索引
  # 可能去除０之后个数不足５个
  length = min(5, len(document_idf_data_list_sorted))
  document_idf_data_list_sorted_top_5 = document_idf_data_list_sorted[0:length]
  document_idf_data_list_sorted_top_5_index = [item[0] for item in document_idf_data_list_sorted_top_5]
  return document_idf_data_list_sorted_top_5_index

def getTopicNum(sc, data, pca_word_vec):
  ###计算所有文章的词语个数
  start_topic = time()
  start_count = time()
  data_dist = data
  # data_dist = sc.parallelize(data)
  document_list = data_dist.reduce(lambda a, b: a+b)
  document_list_para = sc.parallelize(document_list)
  document_count = document_list_para.distinct().collect()
  word_count = len(document_count)
  end_count = time()
  print("calculating word time", end_count-start_count)
  
  ###获取词频
  start_hash = time()
  hashingTF = myfile.HashingTF(numFeatures=2*word_count)
  tf_para = hashingTF.transform(data_dist)
  end_hash = time()
  print("hashing time: ", end_hash-start_hash)
  
  ###所有文章中每个词的idf
  start_idf = time()
  idf = IDF(minDocFreq=0)
  idf_model = idf.fit(tf_para)
  idf_result_para = idf_model.transform(tf_para)
  end_idf = time()
  print("idf time", end_idf-start_idf)
  
  ###找到所有文章前五个主题，并且计算出不同的个数
  start_top5 = time()
  documents_idf_data_list_sorted_top_5_index = idf_result_para.map(lambda document_idf_data: transListAndTop5(document_idf_data))
  print('sort time: ', time()-start_top5)
  # 用flatMap方法所有list合并在一起
  documents_idf_data_list_sorted_top_5_index_sum = documents_idf_data_list_sorted_top_5_index.flatMap(lambda item: item)
  print('flat time: ', time()-start_top5)
  documents_distinct = documents_idf_data_list_sorted_top_5_index_sum.distinct()
  print('distinct time: ', time()-start_top5)
  topic_number = documents_distinct.count()
  print('count time: ', time()-start_top5)
  end_top5 = time()
  print("top5: ", end_top5-start_top5)
  
  end_topic = time()
  topic_num_ap = int(topic_number / 75)+1
  print("topic_number: ", topic_num_ap)
  print("calculating topic time: ", end_topic-start_topic)
  return topic_num_ap, word_count

###获得主题数目
def get_class_num(sc, corpus_old, pca_word_vec):
  stopwords = []
  with open("stopwords_lda.txt", "r", encoding='utf-8') as file:
    for line in file:
      stopwords.append(line.strip())
  corpus_old_para = sc.parallelize(corpus_old)
  corpus_para = corpus_old_para.map(lambda document: filterStopWords_tf(document, stopwords))
  # corpus = corpus_para.collect()
  topic_number, word_count = getTopicNum(sc, corpus_para, pca_word_vec)
  return topic_number, word_count
