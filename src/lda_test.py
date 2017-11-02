#-*- coding : utf-8 -*-

import gensim
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint
from six import iteritems
import textrank4zh
import codecs

stoplist = set('for a of the and to in'.split())
dictionary = corpora.Dictionary(line.lower().split()[1:] for line in open("EnAbs3K.txt"))
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)
dictionary.compactify()

class MyCorpus(object):
    def __iter__(self):
        for line in open("EnAbs3k.txt"):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())





# corpus_memory_friendly = MyCorpus()
#
# corpora.MmCorpus.serialize("abc.mm", corpus_memory_friendly)
def test():
    corpus = corpora.MmCorpus("abc.mm")
    # lda = models.LdaModel(corpus=corpus, num_topics=100, id2word=dictionary)
    lda = models.LdaModel(corpus=corpus, num_topics=100, id2word=dictionary)
    # pprint(lda.print_topics(10, 7))
    print lda.get_topics()[0]
    print lda.get_document_topics(corpus[0])


# text = codecs.open("EnAbs3K.txt", "r", "utf-8").readline()
# textrank = textrank4zh.TextRank4Keyword(stoplist)
# textrank.analyze(text=text, lower=True, window=2)
# for item in textrank.get_keywords(6, word_min_len=1):
#     print item.word, item.weight

# vec = dictionary.doc2bow(text)
# vecs = MyCorpus()
# vecs_tfidf = tfidf[vecs]
# lsi = models.LsiModel(vecs_tfidf, id2word=dictionary, num_topics=10)
# vecs_lsi = lsi[vecs_tfidf]
# pprint(lsi.print_topics(10))

# n = 0
# for vec in vecs:
#     doc_lda = lda[vec]
#     for id, w in doc_lda:
#         print dictionary[id],
#     print
#     n += 1
#     if n > 10:
#         break

if __name__ == "__main__":
    test()