# -*- coding: utf-8 -*-
"""

jpatrickhall@gmail.com
11.9.2014
Educational use only.

"""

import Preprocess
import NMF
import Postprocess
import sys
import time
#import iopro
import numpy
import re
import os
from sklearn import preprocessing, cluster
from scipy import sparse
from multiprocessing import Process

def main():
    """ Main driver method for NMF classes.

    Creates a list of stems and other replacements to clean raw text.
    Preprocesses raw text into term-by-document (TBD) matrix.
    Decomposes TBD matrix into representative, non-negative features.
    Labels term features and clusters documents.

    Assign parameters below:

    nthread - threads for preprocessing (int)
    working_dir - working dir for preprocessing (string)
    raw_file - file containing raw text for preprocessing (string)
    stop_dir - directory containing stoplists (string)
    threshold - the number of times a token must occur to be used (int)
    custom_replacements - a list of terms and their replacements (string)
    make_replacements - just make replacements.txt and stop (boolean)

    n_features - number of nmf features to generate (int)
    method - als nmf method (string)
    tfidf - perform tf/idf weighting (boolean)
    als_opts - options for als optimization (dict)
    default: {'l': 0.5, 'max_iter': 50, check_iter': 1, 'tolerance': 0.001,
              'patience': None, 'patience_increase': None,
              'row_batch_size': None, 'col_batch_size': None}
    random_seed
    default: 12345 (int)

    n_clusters - number of document clusters
    normalizer - normalizing technique from scikit learn
    (scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
    segmenter - segmentation algorithm from scikit learn
    (scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster)

    """

    nthread = 4
    working_dir = 'C:\\Users\\imdim\\OneDrive\\Documents\\SEM3\\BPA\\nmf_data'
    raw_file = 'Description_TypeBreach.txt'
    stop_dir = 'stoplist'
    threshold = 3 # lower means more resolution in topics, but also overfitting
    custom_replacements = "replacements.txt"
    make_replacements = False # needs to be set to true once for new data

    n_features = 25 # k, topics or no. of features in the NMF
    method = 'DENSE' # DENSE is faster, SPARSE uses less memory
    tfidf = False 
    als_opts = None
    random_seed = None

    n_clusters = 6
    normalizer = preprocessing.Normalizer()
    segmenter = cluster.KMeans(init='k-means++', n_clusters=n_clusters,\
                               n_init=3, n_jobs=-1)

    huge_tic = time.time()

    print '-------------------------------------------------------------------'
    print 'Proceeding with options: '
    print 'nthread              = %s' % (nthread)
    print 'working_dir          = %s' % (working_dir)
    print 'raw_file             = %s' % (raw_file)
    print 'stop_dir             = %s' % (stop_dir)
    print 'threshold            = %s' % (threshold)
    print 'custom_replacements  = %s' % (custom_replacements)
    print 'make_replacements    = %s' % (make_replacements)
    print 'n_features           = %s' % (n_features)
    print 'method               = %s' % (method)
    print 'tfidf                = %s' % (tfidf)
    print 'als_opts             = %s' % (als_opts)
    print 'random_seed          = %s' % (random_seed)
    print 'n_clusters           = %s' % (n_clusters)
    print 'normalizer           = %s' % (normalizer)
    print 'segmenter            = %s' % (segmenter)

    ### Just make replacements.txt and stop
    if make_replacements:

        big_tic = time.time()

        print '---------------------------------------------------------------'
        print 'Initializing Preprocessing ... '
        pre = Preprocess.Preprocess(nthread, working_dir, raw_file, stop_dir,\
                                    threshold, None)

        print '---------------------------------------------------------------'
        print 'Converting to lowercase ... '
        tic = time.time()
        processes = []
        try:
            for i in range(0, int(nthread)):
                process_name = 'Process_' + str(i)
                process = Process(target=pre.lower, name=process_name,\
                                 args=(i,))
                process.start()
                processes.append(process)
            for process_ in processes:
                process_.join()
            toc = time.time()-tic
            print 'Converted to lowercase in %.2f s.' % (toc)
        except:
            print 'ERROR: Could not convert to lowercase.'

            print sys.exc_info()
            exit(-1)

        print '---------------------------------------------------------------'
        print 'Generating stems ... '
        tic = time.time()
        processes = []
        try:
            for i in range(0, int(nthread)):
                process_name = 'Process_' + str(i)
                process = Process(target=pre.lemmatize, name=process_name,\
                                  args=(i, '_raw_lower'))
                process.start()
                processes.append(process)
            for process_ in processes:
                process_.join()
            toc = time.time()-tic
            print 'Stems generated in %.2f s.' % (toc)
        except:
            print 'ERROR: Could not generate stems.'
            print sys.exc_info()
            exit(-1)

        print '---------------------------------------------------------------'
        print 'Resolving unique terms from different threads ... '
        tic = time.time()
        pre.resolve_replacements()
        toc = time.time()-tic
        print 'Resolved unique terms in %.2f s.' % (toc)

        print '---------------------------------------------------------------'
        print 'Updating replacement dictionary ... '

        def load_dict(in_file):
            """ Loads a dictionary from a two column csv or space separated
                file.

            Args:
                in_file: Two column text file containing dictionary key-value
                pairs.

            Returns:
                A Python dictionary from the file.
            """

            dict_ = {}
            with open(in_file) as file_:
                for line in file_:
                    if not re.split(r'\s|,', line)[0].startswith('#') or \
                        re.split(r'\s|,', line)[0] in ['', '\r\n', '\n']:
                        pair = re.split(r'\s|,', line)
                        key_ = pair[0].strip().lower()
                        value_ = pair[1].strip().lower()
                        dict_[key_] = value_
                return dict_

        replace_dict = load_dict(working_dir + os.sep + 'replacements.txt')
        print 'Original stem dictionary contains %i terms.'\
            % (len(replace_dict))

        add_dict = load_dict('add.txt')
        print 'Add dictionary contains %i terms.' % (len(add_dict))

        change_dict = load_dict('change.txt')
        print 'Change dictionary contains %i terms.' % (len(change_dict))

        del_dict = load_dict('delete.txt')
        print 'Delete dictionary contains %i terms.' % (len(del_dict))

        ### Add terms
        replace_dict.update(add_dict)

        ### Change terms
        for k in change_dict:
            replace_dict[k] = change_dict[k]

        ### Delete terms
        for k in del_dict:
            if k in replace_dict:
                del replace_dict[k]

        print 'Updated stem dictionary contains %i terms.'\
            % (len(replace_dict))

        ### Output replacements.txt
        with open(working_dir + os.sep + 'replacements.txt', 'wb') as repl_out:
            repl_out.write('### word lemma\n')
            for key_, value_ in sorted(replace_dict.items()):
                repl_out.write(str(key_))
                repl_out.write(' ')
                repl_out.write(str(value_))
                repl_out.write('\n')

    ### Do the whole shebang
    else:

        big_tic = time.time()

        print '---------------------------------------------------------------'
        print 'Initializing Preprocessing ... '
        pre = Preprocess.Preprocess(nthread, working_dir, raw_file, stop_dir,\
                                    threshold, custom_replacements)

        print '---------------------------------------------------------------'
        print 'Converting to lowercase ... '
        tic = time.time()
        processes = []
        try:
            for i in range(0, int(nthread)):
                process_name = 'Process_' + str(i)
                process = Process(target=pre.lower, name=process_name,\
                                  args=(i,))
                process.start()
                processes.append(process)
            for process_ in processes:
                process_.join()
            toc = time.time()-tic
            print 'Converted to lowercase in %.2f s.' % (toc)
        except:
            print 'ERROR: Could not convert to lowercase.'
            print sys.exc_info()
            exit(-1)

        print '---------------------------------------------------------------'
        print 'Cleaning characters ... '
        tic = time.time()
        processes = []
        try:
            for i in range(0, int(nthread)):
                process_name = 'Process_' + str(i)
                process = Process(target=pre.regex_replace_common_terms,\
                                 name=process_name, args=(i,))
                process.start()
                processes.append(process)
            for process_ in processes:
                process_.join()
            toc = time.time()-tic
            print 'terms replaced in %.2f s.' % (toc)
        except:
            print 'ERROR: Could not concatenate common pop-culture terms.'
            print sys.exc_info()
            exit(-1)

        print '---------------------------------------------------------------'
        print 'Generating stems ... '
        tic = time.time()
        processes = []
        try:
            for i in range(0, int(nthread)):
                process_name = 'Process_' + str(i)
                process = Process(target=pre.lemmatize, name=process_name,\
                                  args=(i,))
                process.start()
                processes.append(process)
            for process_ in processes:
                process_.join()
            toc = time.time()-tic
            print 'Stems generated in %.2f s.' % (toc)
        except:
            print 'ERROR: Could not generate stems.'
            print sys.exc_info()
            exit(-1)

        print '---------------------------------------------------------------'
        print 'Resolving unique terms from different threads ... '
        tic = time.time()
        pre.resolve_replacements()
        toc = time.time()-tic
        print 'Resolved unique terms in %.2f s.' % (toc)

        print '---------------------------------------------------------------'
        print 'Replacing stemmed terms ... '
        tic = time.time()
        processes = []
        try:
            for i in range(0, int(nthread)):
                process_name = 'Process_' + str(i)
                process = Process(target=pre.replace_terms,\
                                 name=process_name, args=(i,))
                process.start()
                processes.append(process)
            for process_ in processes:
                process_.join()
            toc = time.time()-tic
            print 'Terms replaced in %.2f s.' % (toc)
        except:
            print 'ERROR: Could not replace stems.'
            print sys.exc_info()
            exit(-1)

        print '---------------------------------------------------------------'
        print 'Applying stop list ... '
        tic = time.time()
        processes = []
        try:
            for i in range(0, int(nthread)):
                process_name = 'Process_' + str(i)
                process = Process(target=pre.apply_stoplist,\
                                  name=process_name, args=(i,))
                process.start()
                processes.append(process)
            for process_ in processes:
                process_.join()
            toc = time.time()-tic
            print 'Stopped terms removed in %.2f s.' % (toc)
        except:
            print 'ERROR: Could not apply stop list.'
            print sys.exc_info()
            exit(-1)

        print '---------------------------------------------------------------'
        print 'Removing short terms ... '
        tic = time.time()
        processes = []
        try:
            for i in range(0, int(nthread)):
                process_name = 'Process_' + str(i)
                process = Process(target=pre.remove_short_terms,\
                                 name=process_name, args=(i,))
                process.start()
                processes.append(process)
            for process_ in processes:
                process_.join()
            toc = time.time()-tic
            print 'Short terms removed in %.2f s.' % (toc)
        except:
            print 'ERROR: Could not remove short terms.'
            print sys.exc_info()
            exit(-1)

        print '----------------------------------------------------------------'
        print 'Counting all terms ... '
        tic = time.time()
        counts = pre.get_counts(nthread)
        toc = time.time()-tic
        print 'Terms counted in %.2f s.' % (toc)

        print '---------------------------------------------------------------'
        print 'Removing infrequent terms ... '
        tic = time.time()
        processes = []
        try:
            for i in range(0, int(nthread)):
                process_name = 'Process_' + str(i)
                process = Process(target=pre.remove_infrequent_terms,\
                                  name=process_name, args=(i, counts))
                process.start()
                processes.append(process)
            for process_ in processes:
                process_.join()
            toc = time.time()-tic
            print 'Infrequent terms removed in %.2f s.' % (toc)
        except:
            print 'ERROR: Could not remove infrequent terms.'
            print sys.exc_info()
            exit(-1)

        print '---------------------------------------------------------------'
        print 'Processing unique terms ... '
        tic = time.time()
        terms = pre.get_unique_terms()
        toc = time.time()-tic
        print 'Terms processed in %.2f s.' % (toc)
        print 'Number of unique terms: %i' % (len(terms))

        print '---------------------------------------------------------------'
        print 'Writing COO term-by-document matrix ... '
        tic = time.time()
        pre.write_tbd()
        toc = time.time()-tic
        print 'Matrix written in %.2f s.' % (toc)

        row_fname = pre.get_row_fname()
        col_fname = pre.get_col_fname()
        val_fname = pre.get_val_fname()
        terms_fname = pre.get_terms_fname()

        big_toc = time.time()-big_tic
        print '---------------------------------------------------------------'
        print 'PREPROCESSING run time: %.2f s.' % (big_toc)

        big_tic = time.time()

        try:
            tic = time.time()
            print '-----------------------------------------------------------'
            print 'Loading data for NMF...'
            ### Highest term count is 2147483647 for int32
            row = numpy.loadtxt(row_fname, dtype='int32', skiprows=1)
            col = numpy.loadtxt(col_fname, dtype='int32', skiprows=1)
            val = numpy.loadtxt(val_fname, dtype='int32', skiprows=1)

            tbd_matrix = sparse.coo_matrix((val, (row, col)),\
                shape=(numpy.amax(row)+1, numpy.amax(col)+1),\
                dtype='float32').tocsc()
            del row
            del col
            del val

            toc = time.time()-tic
            print 'Data loaded in %f s.' % (toc)

        except:
            print 'ERROR: Problem loading row, col, or val text files from %s\
                   and converting to Scipy sparse matrix.' % (working_dir)
            print sys.exc_info()
            exit(-1)

        nmf = NMF.NMF(tbd_matrix, n_features, method, tfidf, als_opts,\
                      random_seed)
        nmf.nmf()
        nmf.write_output(working_dir)
        w_fname = nmf.get_w_fname()
        h_fname = nmf.get_h_fname()

        big_toc = time.time()-big_tic
        print '---------------------------------------------------------------'
        print 'NMF run time: %.2f s.' % (big_toc)
        big_tic = time.time()
        print '---------------------------------------------------------------'
        print 'Initializing postprocessing ...'
        post = Postprocess.Postprocess(working_dir, w_fname, h_fname,\
                                       terms_fname, normalizer, segmenter)

        print '---------------------------------------------------------------'
        print 'Writing sorted features ... \n'
        features = post.sort_features()
        feature_names = post.output(features, 'features_')
        print 'Feature names for Gephi:'
        for i, k in enumerate(feature_names.keys()):
            if i + 1 == len(feature_names.keys()):
                print str(k) + ": '" + feature_names[k] + "'"
            else:
                print str(k) + ": '" + feature_names[k] + "',"

        print '---------------------------------------------------------------'
        print 'Clustering features ... \n'
        centroids = post.cluster_docs(feature_names)
        post.output(centroids, 'centroids_', get_names=False)
        big_toc = time.time()-big_tic
        print '---------------------------------------------------------------'
        big_toc = time.time()-big_tic
        print 'POSTPROCESSING run time: %.2f s.' % big_toc
        print '---------------------------------------------------------------'
        huge_toc = time.time() - huge_tic
        print 'TOTAL run time: %.2f s.' % huge_toc


if __name__ == '__main__':
    main()

