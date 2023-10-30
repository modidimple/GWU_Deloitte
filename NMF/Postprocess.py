# -*- coding: utf-8 -*-
"""
jpatrickhall@gmail.com
11.6.14
Educational use only.
"""
import getopt
import sys
#import iopro
import numpy
import pandas
import time
import os
from datetime import datetime
from sklearn import preprocessing, cluster

class Postprocess(object):
    """ Class to postprocess features derived from NMF.

    Sorts features by terms weights.
    Clusters documents in the feature space.

    Attributes:
        working_dir: Directory in which results are written. Expected to\
                     contain input as well, but not necessarily.
        w_matrix_fname: Filename for features in the term space.
        h_matrix_fname: Filename for features in the document space.
        terms_fname: Filename for the list of unique terms in the NMF analysis.
        normalizer: Scikit-learn normalization class.
        segmenter: Sckit-learn clustering class.

    """

    def __init__(self, working_dir, w_matrix_fname, h_matrix_fname,\
                 terms_fname, normalizer, segmenter):
        """ Sets class attributes and loads feature matrices and terms. """

        def load_terms():
            """ Loads the unique terms into a numpy numpy.chararray.

            Returns:
                Numpy numpy.chararray of unique terms.
            """

            return numpy.loadtxt(terms_fname, dtype='S200', delimiter=' ',\
                usecols=(1,), unpack=True)

        def load_factor_matrix(fname):
            """ Loads feature matrix into numpy array.

            Returns:
                Numpy array of features.
            """

            return numpy.loadtxt(fname, dtype='float64')

        self.working_dir = working_dir

        self.terms = load_terms()
        self.terms = self.terms.reshape(self.terms.shape[0], 1)

        self.w_matrix = load_factor_matrix(w_matrix_fname)
        self.h_matrix = load_factor_matrix(h_matrix_fname)

        self.normalizer = normalizer
        self.segmenter = segmenter

        self.feature_name = {}

        ### Display option for Pandas dataframes
        pandas.options.display.float_format = '{:,.4f}'.format

    def normalize_rows(self, matrix):
        """ Calculates the L2 norm of a row of a matrix.

            Args:
                matrix

            Returns:
                Matrix with normalized rows.
            """
        for i in range(0, matrix.shape[0]):
            row_i = matrix[i].reshape(1, matrix.shape[1])
            row_i = self.normalizer.fit_transform(row_i)
            matrix[i] = row_i.reshape(matrix.shape[1])
        return matrix

    def sort_features(self):
        """ Sorts each feature by highest term weight.

            Writes file containing each feature's highest weighted terms to
            working directory.
            Prints highest weighted terms to terminal.

            Returns:
                A dataframe of the features with joined terms.
        """

        self.w_matrix = self.normalize_rows(self.w_matrix.T).T
        features = numpy.concatenate((self.terms, self.w_matrix), axis=1)

        col_names = ['d'+ str(i) for i in range(0, self.w_matrix.shape[1])]
        col_names.insert(0, 'term')
        df_ = pandas.DataFrame(features, columns=col_names)

        return df_.convert_objects(convert_numeric=True)


    def output(self, df_, fname_prefix, get_names=True):
        """ Writes a sorted set of features to file and to the terminal.

        Args:
            df_: The dataframe to print.
            fname_prefix: The output filename prefix.
            get_names: If true, return the names of each feature, i.e. the term
                corresponding to the highest weight in the feature.

        Returns: If get_names=True, An indexed dictionary of feature names. 
            Feature names correspond to the highest weight term in a feature.

        """

        dfs = []
        feature_names = {}

        time_stamp = datetime.fromtimestamp(time.time()).\
            strftime('%Y-%m-%d_%H-%M-%S')
        fname = self.working_dir + os.sep + fname_prefix +\
            time_stamp + '.txt'

        with open(fname, 'wb') as named_feature_file:
            for j in range(1, len(df_.columns)):
                col_df = df_.iloc[:, [0, j]]
                col_df = col_df.convert_objects(convert_numeric=True)
                dfs.append(pandas.DataFrame.reset_index(col_df.sort_values(\
                                     by='d'+str(j-1),\
                                     ascending=False).head(10), drop=True))

                # determine feature name
                if numpy.power(dfs[(j%4)-1]['d'+str(j-1)][0], 2) -\
                        numpy.power(dfs[(j%4)-1]['d'+str(j-1)][1], 2) > 0.9:
                    feature_names[j - 1] = str(dfs[(j % 4) - 1]['term'][0])
                else:
                    feature_names[j - 1] = str(dfs[(j % 4) - 1]['term'][0]) + '+' + str(dfs[(j % 4) - 1]['term'][1])

                # append name onto topic
                dfs[(j % 4) - 1] = dfs[(j % 4) - 1].append({'term': feature_names[j - 1],
                                                            'd'+str(j-1): numpy.nan},
                                                            ignore_index=True)

                # formatted output
                if j%4 == 0 or j == len(df_.columns)-1:
                    lines = pandas.DataFrame.to_string(pandas.concat(dfs,\
                        axis=1), index=False) + '\n'
                    print lines
                    named_feature_file.write(lines+'\n')
                    dfs = []
        if get_names:
            return feature_names
        else:
            return None

    def cluster_docs(self, feature_names):

        """ Clusters the documents in the feature space and names each
            cluster by its most representative features.

            Args:
                feature_names: Dictionary of the names of the features; used to
                    name the centroids.
                
            Returns: 
                A dataframe of centroids with feature names joined.
            
        """

        self.h_matrix = self.normalize_rows(self.h_matrix)
        self.segmenter.fit(self.h_matrix.T)
        centroids_ = self.normalize_rows(self.segmenter.cluster_centers_)

        time_stamp = datetime.fromtimestamp(time.time()).\
                                          strftime('%Y-%m-%d_%H-%M-%S')
        fname = self.working_dir + os.sep + 'normalized_centroids_' +\
            time_stamp + '.txt' 
        numpy.savetxt(fname, centroids_)
        
        names = numpy.chararray((self.h_matrix.shape[0], 1), itemsize=200)

        for i in sorted(feature_names):
            names[i, 0] = feature_names[i]

        centroids = numpy.hstack((names, centroids_.T))
    
        col_names = ['d'+ str(i) for i in range(0, centroids_.shape[0])]
        col_names.insert(0, 'term')

        df_ = pandas.DataFrame(centroids, columns=col_names)

        return df_.convert_objects(convert_numeric=True)

def main(argv):
    """ Main method to act as driver if class is used standalone.

    Args:
        argv: Command line args.

    Raises:
        GetoptError: Error parsing command line options.
    """

    working_dir = None
    w_matrix_fname = 'W.txt'
    h_matrix_fname = 'H.txt'
    terms_fname = 'terms.txt'
    normalizer = preprocessing.Normalizer()
    segmenter = cluster.KMeans(init='k-means++', n_clusters=5, n_init=3)

    try:
        opts, _ = getopt.getopt(argv, "d:W:H:t:n:c:h")
        for opt, arg in opts:
            if opt == '-d':
                working_dir = arg
            elif opt == '-W':
                w_matrix_fname = arg
            elif opt == '-H':
                h_matrix_fname = arg
            elif opt == '-t':
                terms_fname = arg
            elif opt == '-n':
                normalizer = eval(arg)
            elif opt == '-c':
                segmenter = eval(arg)
            elif opt == '-h':
                print 'Postprocess.py \n \
-d <directory containing W and H matrix input files and term input file (string)>\n \
-W <name of the file containing the values of the W matrix (string)> (default= W.txt)\n \
-H <name of the file containing the values of the H matrix (string)> (default= H.txt)\n \
-t <name of the file containing the unique term counts (string)> (default= terms.txt)\n \
-n <scikit-learn normalizer class (string)> (default= sklearn.preprocessing.Normalizer())\n \
-c <scikit-learn cluster class (string)> (default= sklearn.cluster.KMeans(init="k-means++", n_clusters= 200, n_init=3))'
                exit(-1)
    except getopt.GetoptError:
        print sys.exc_info()
        print 'Postprocess.py \n \
-d <directory containing W and H matrix input files and term input file (string)>\n \
-W <name of the file containing the values of the W matrix (string)> (default= W.txt)\n \
-H <name of the file containing the values of the H matrix (string)> (default= H.txt)\n \
-t <name of the file containing the unique term counts (string)> (default= terms.txt)\n \
-n <scikit-learn normalizer class (string)> (default= sklearn.preprocessing.Normalizer())\n \
-c <scikit-learn cluster class (string)> (default= sklearn.cluster.KMeans(init="k-means++", n_clusters= 200, n_init=3))'
        print '---------------------------------------------------------------'
        print 'Working directory (-d)   = %s' % (working_dir)
        print 'W matrix file (-W)       = %s' % (w_matrix_fname)
        print 'H matrix file (-H)       = %s' % (h_matrix_fname)
        print 'Terms file (-t)          = %s' % (terms_fname)
        print 'Normalization class (-n) = %s' % (normalizer)
        print 'Cluster class (-c)       = %s' % (segmenter)
        exit(-1)

    if working_dir == None:
        print 'Mandatory argument -d must be supplied.'
        print 'Working directory (-d)   = %s' % (working_dir)
        exit(-1)

    if w_matrix_fname == 'W.txt':
        w_matrix_fname = working_dir + os.sep + w_matrix_fname
    if h_matrix_fname == 'H.txt':
        h_matrix_fname = working_dir + os.sep + h_matrix_fname
    if terms_fname == 'terms.txt':
        terms_fname = working_dir + os.sep + terms_fname

    print '-------------------------------------------------------------------'
    print 'Proceeding with options:'
    print 'Working directory (-d)   = %s' % (working_dir)
    print 'W matrix file (-W)       = %s' % (w_matrix_fname)
    print 'H matrix file (-H)       = %s' % (h_matrix_fname)
    print 'Terms file (-t)          = %s' % (terms_fname)
    print 'Normalization class (-n) = %s' % (normalizer)
    print 'Cluster class (-c)       = %s' % (segmenter)
    print '-------------------------------------------------------------------'
    print 'Initializing postprocessing ...'

    big_tic = time.time()
    post = Postprocess(working_dir, w_matrix_fname, h_matrix_fname,\
                       terms_fname, normalizer, segmenter)

    tic = time.time()
    print '-------------------------------------------------------------------'
    print 'Writing sorted features ... \n'
    feature_names = post.sort_features()
    print 'Features sorted in %.2f s.' % (time.time() - tic)
    tic = time.time()
    print '-------------------------------------------------------------------'
    print 'Clustering documents ... \n'
    post.cluster_docs(feature_names)
    print 'Documents clustered in %.2f s.' % (time.time() - tic)
    big_toc = time.time()-big_tic
    print '-------------------------------------------------------------------'
    print 'Total run time: %.2f s.' % (big_toc)

if __name__ == '__main__':
    main(sys.argv[1:])
