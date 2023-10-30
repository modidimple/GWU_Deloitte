# -*- coding: utf-8 -*-
"""
jpatrickhall@gmail.com
11.8.14
Educational use only. 
"""

import unittest
import Preprocess
import NMF
import Postprocess
#import iopro
import numpy
from scipy import sparse
from sklearn import preprocessing, cluster

class TestPostprocess(unittest.TestCase):

    """ Simple unit tests for Postprocess.py.

    Attributes:
        post: Instance of Postprocess class used for tests.

    """

    def setUp(self):

        """ Generates necessary files to run Postprocessing.

        Runs Preprocessing.py on sample data.
        Runs NMF.py.
        Instantiates Postprocessing.

        """

        nthread = 1
        working_dir = 'C:\\Temp'
        raw_file = 'sample_data.txt'
        stop_dir = 'stoplist'
        threshold = 3
        custom_replacements = 'replacements.txt'
        pre = Preprocess.Preprocess(nthread, working_dir, raw_file,\
            stop_dir, threshold, custom_replacements)
        pre.lower(0)
        pre.regex_replace_common_terms(0)
        pre.replace_terms(0)
        pre.apply_stoplist(0)
        pre.remove_short_terms(0)
        pre.remove_infrequent_terms(0, 10)
        terms = pre.get_unique_terms()
        pre.set_terms(terms)
        pre.write_tbd()
        row_fname = pre.get_row_fname()
        col_fname = pre.get_col_fname()
        val_fname = pre.get_val_fname()
        terms_fname = pre.get_terms_fname()

        features = 6
        method = None # default
        tfidf = None # default
        als_opts = None # default
        random_seed = None # default
        row = numpy.loadtxt(row_fname, dtype='int32', skiprows=1)
        col = numpy.loadtxt(col_fname, dtype='int32', skiprows=1)
        val = numpy.loadtxt(val_fname, dtype='int32', skiprows=1)
        tbd_matrix = sparse.coo_matrix((val, (row, col)),\
                                        shape=(numpy.amax(row)+1,\
                                               numpy.amax(col)+1),\
                                        dtype='float32').tocsc()
        nmf = NMF.NMF(tbd_matrix, features, method, tfidf, als_opts,\
                      random_seed)
        nmf.nmf()
        nmf.write_output(working_dir)
        w_fname = nmf.get_w_fname()
        h_fname = nmf.get_h_fname()

        clusters = 6
        normalizer = preprocessing.Normalizer()
        segmenter = cluster.KMeans(init='k-means++', n_clusters=clusters,\
                                   n_init=3)
        self.post = Postprocess.Postprocess(working_dir, w_fname, h_fname,\
                                       terms_fname, normalizer, segmenter)

    def test_all(self):

        """ Runs all methods sequentially. """

        features = self.post.sort_features()
        feature_names = self.post.output(features, 'feature_')

        centroids = self.post.cluster_docs(feature_names)
        self.post.output(centroids, 'centroids_', get_names=False)



def main():

    """ Run tests. """

    unittest.main()

if __name__ == '__main__':
    main()
