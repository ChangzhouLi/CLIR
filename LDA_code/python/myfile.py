import sys
import warnings
import random
import binascii
if sys.version >= '3':
    basestring = str
    unicode = str

from py4j.protocol import Py4JJavaError

from pyspark import SparkContext
from pyspark.rdd import RDD, ignore_unicode_prefix
from pyspark.mllib.common import callMLlibFunc, JavaModelWrapper
from pyspark.mllib.linalg import (
    Vector, Vectors, DenseVector, SparseVector, _convert_to_vector)
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import JavaLoader, JavaSaveable


class HashingTF(object):
    """
    .. note:: Experimental

    Maps a sequence of terms to their term frequencies using the hashing
    trick.

    Note: the terms must be hashable (can not be dict/set/list...).

    :param numFeatures: number of features (default: 2^20)

    >>> htf = HashingTF(100)
    >>> doc = "a a b b c d".split(" ")
    >>> htf.transform(doc)
    SparseVector(100, {...})
    """
    def __init__(self, numFeatures=1 << 20):
        self.numFeatures = numFeatures

    def indexOf(self, term):
        """ Returns the index of the input term. """
        return hash(term) % self.numFeatures

    def transform(self, document):
        """
        Transforms the input document (list of terms) to term frequency
        vectors, or transform the RDD of document to RDD of term
        frequency vectors.
        """
        if isinstance(document, RDD):
            return document.map(self.transform)

        freq = {}
        dict = {}
        for term in document:
            i = self.indexOf(term)
            freq[i] = freq.get(i, 0) + 1.0
        return Vectors.sparse(self.numFeatures, freq.items())
        
    def getDict(self, document):
        """
        Transforms the input document (list of terms) to term frequency
        vectors, or transform the RDD of document to RDD of term
        frequency vectors.
        """
        if isinstance(document, RDD):
            return document.map(self.getDict)

        freq = {}
        dict = {}
        for term in document:
            i = self.indexOf(term)
            freq[i] = freq.get(i, 0) + 1.0
            dict[str(i)] = term
        return dict