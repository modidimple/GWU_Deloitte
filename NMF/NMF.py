# -*- coding: utf-8 -*-
"""
jpatrickhall@gmail.com
11.8.14
Educational use only.
"""

import sys
import getopt
import os
#import iopro
import time
import ast
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from scipy import sparse
from scipy.sparse import linalg
from sklearn.feature_extraction import text

class NMF(object):
    """
    ALS NMF with mini-batches for training and calculating
    - L1 norms for mini-batch training
    - Frobenious norms for conventional training

    Method summary:

    - SPARSE_MINI_BATCH (experimental)
        Define row_batch_size and col_batch_size in als_opts dictionary
        Define patience and patience increase in als_opts dictionary
        Do not define tolerance in als_opts dictionary
        Optimization uses large, sparse matrix mini-batches for training and
            for L1 norm calculation
        Optimization continues until patience threshold is reached and L1 norm
            increases for several iterations

    - DENSE_MINI_BATCH (experimental)
        Same as above except optimization uses large, dense mini-batches for
            training and for L1 norm calculation

    - SPARSE (default)
        Define row_batch_size and col_batch_size in als_opts dictionary to
            calculate error on small mini-batches, otherwise error is
            calculated on full, sparse matrices
        If you define row_batch_size and col_batch_size in als_opts dictionary
            to calculate error on small mini-batches then you must also define
            patience and patience increase in als_opts dictionary
        If you do not define row_batch_size and col_batch_size in als_opts
            dictionary then error will be calculated on full sparse matrices
            and you must define tolerance in als_opts dictionary
        Optimization uses full, sparse matrices for training
        If you define row_batch_size and col_batch_size in als_opts dictionary
            to calculate error on small mini-batches, optimization continues
            until patience threshold is reached and Frobenius norm increases
            for several iterations
        If you do not define row_batch_size and col_batch_size in als_opts
            dictionary then error will be calculated on full sparse matrices
            and optimization proceeds until training error decreases below the
            tolerance threshold

    - DENSE
        Same as above except optimization uses full, dense matrices for
            training and for Frobenius norm calculation

    General Usage Comments:

    - To skip calculating error, set check_iter > max_iter
    - Either define tolernace for full matrix error calculations or define
        patience, patience_increase, row_batch_size, and col_batch_size for
        mini-batch error calculations
    - Sparse calculations will be slower, but require less memory
    - Dense calculations will be faster, but require more memory
    - Mini-batches:
        Possibly increases speed
        Possibly reduces memory requirements
        Possibly reduces accuracy
        Definitely invalidate convergence gaurantees
    - In preliminary testing, ideal training mini-batches contained:
       ~50% of the cols from M (cols = terms)
       100% of the rows from M (rows = docs)
    - In preliminary testing, ideal error mini-batches contained:
        ~30% of rows from M (cols = terms)
        ~30% of the rows from M (rows = docs)

    Attributes:
    M: scipy sparse matrix in CSR format
    n_features: number of features
    als_opts =: dictionary with 8 mandatory options
    Example for mini-batch training:
    {'l': 0.5, 'max_iter': 50, 'check_iter': 1, tolerance: 'None',\
        'patience': 10, 'patience_increase': 2,\
        'row_batch_size': <~80% of rows>, 'col_batch_size': <~50% of cols>}
    Example for conventional training without mini-batch error:
    {'l': 0.5, 'max_iter': 50, 'check_iter':1, 'tolerance': 0.01,\
        'patience': None, 'patience_increase': None, 'row_batch_size': None,\
        'col_batch_size': None} (default)
    Example for conventional training with mini-batch error:
    {'l': 0.5, 'max_iter': 50, 'check_iter':1, tolerance: 'None',\
        'patience': 10, 'patience_increase': 2,\
        'row_batch_size': <10% of  rows>, 'col_batch_size': <10% of cols>}
    method =: 'SPARSE' (default), 'DENSE', 'SPARSE_MINI_BATCH' (experimental),\
        'DENSE_MINI_BATCH' (experimental)
    random_seed =: 12345 (default)

    Variables representing matrices are named with capital letters.

    """

    def __init__(self, M, n_features, method, tfidf, als_opts, random_seed):
        """ Initializes class attributes, including several large matrices. """

        try:
            import mkl
            print '-----------------------------------------------------------'
            print "Running with MKL acceleration version %s ... "\
                  % mkl.__version__
        except:
            pass

        def init_l(n_features, lambda_):
            """ Initiliazes matrix of shrinkage paramters.

            Args:
                n_features: Dimension of square matrix L.
                lambda_: Value of shrinkage parameter.

            Returns: Square scipy sparse matrix with values specified by l down
                the diagonal as 32-bit Numpy floats.

            """
            print '-----------------------------------------------------------'
            print 'Initializing L matrix using with l = %f ...' % (lambda_)
            return sparse.identity(n_features, dtype='float32').tocsr()*lambda_

        def init_w(self, n_features):

            """ Returns initialized W matrix.

            Args:
                n_features: Number of features. W will be number of terms by k.

            Returns: ACOL initialized scipy sparse matrix in CSR format as
                32-bit Numpy floats.

            """

            tic = time.time()
            print '-----------------------------------------------------------'
            print 'Initializing W matrix using ACOL method ...'
            w_matrix = sparse.lil_matrix((self.M.get_shape()[0], n_features),\
                                          dtype='float32')
            for i in range(0, n_features):
                rand_col = self.M[:, np.random.randint(self.M.get_shape()[1],\
                                     size=n_features)].todense()
                rand_col_mean = np.mean(rand_col, axis=1)
                w_matrix[:, i] = sparse.lil_matrix(rand_col_mean)
                print 'Column %d of %d intialized ...' % (i+1, n_features)
            toc = time.time()-tic
            print 'W matrix initialized in %f s.' % (toc)
            return w_matrix.tocsr()

        self.M = sparse.csc_matrix(M, dtype='float32')
        self.n_features = n_features

        if method == None:
            method = 'SPARSE'
        self.method = method

        if tfidf == None:
            tfidf = True
        self.tfidf = tfidf
        if self.tfidf == True:
            tic = time.time()
            print '-----------------------------------------------------------'
            print 'Applying TF-IDF weighting to M ...'
            transformer = text.TfidfTransformer()
            self.M = transformer.fit_transform(self.M)
            self.M = sparse.csc_matrix(self.M, dtype='float32')
            toc = time.time()-tic
            print 'Weighting completed in %f s.' % (toc)

        if als_opts == None:
            als_opts = {'l': 0.5, \
                      'max_iter': 50, \
                      'check_iter': 1, \
                      'tolerance': 0.001, \
                      'patience': None, \
                      'patience_increase': None, \
                      'row_batch_size': None, \
                      'col_batch_size': None}
        self.als_opts = als_opts

        if random_seed == None:
            random_seed = 12345
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.W = init_w(self, n_features)
        self.L = init_l(n_features, als_opts['l'])
        self.H = None

        self.w_fname = None
        self.h_fname = None

        self.best_err = None

    def get_w(self):
        """ Getter for W.

        Returns: W.
        """
        return self.W

    def get_w_fname(self):
        """ Getter for the output file containing W.

        Returns: Name of the output file containing W.
        """
        return self.w_fname

    def set_w_fname(self, name):
        """ Setter for the output file containing W. """
        self.w_fname = name

    def get_h(self):
        """ Getter for H.

        Returns: H.
        """
        return self.H

    def get_h_fname(self):
        """ Getter for the output file containing H.

        Returns: Name of the output file containing H.
        """
        return self.h_fname

    def set_h_fname(self, name):
        """ Setter for the output file containing H. """
        self.h_fname = name

    def get_best_err(self):
        """ Getter for the NMF optimization best error.

        Returns: The NMF optimization best error.
        """
        return self.best_err

    def nmf(self):

        """ Decomposes M into W and H; saves W and H as instance vars. """

        def plot_error(method, toc, err, best_iter, best_err, check_iter):
            """ Plot optimization error.

            Ensure error decreases appropriately for mini-batch methods.

            Args:
                method: The NMF method.
                toc: Total time in seconds for NMF.
                err: Numpy array of errors at each iteration.
                best_iter: Iteration with with best error
                    (not alway last iter for mini-batch training).
                best_err: Lowest error (achieved at best_iter).
                check_iter: The number of iterations between evaluating
                    reconstruction error.

            """

            line1, = plt.plot(np.asarray([j for j in range(check_iter,\
                best_iter+1, check_iter)]),\
                err[0:int(np.floor(best_iter/check_iter))], 'bo-')
            plt.title('ALS NMF by %s Method' % (method))
            plt.xlabel('Iterations')
            plt.ylabel('Objective Function Value')
            plt.legend([line1], ['Obj. Function Val.'])
            annotation = 'NMF completed in %.1f s.' % (toc)
            plt.text(1+(best_iter/100.), best_err, annotation)
            plt.show()


        def mf_to_instance_var(W, H):
            """ Save matrices of features, W and H, to the given instance.

            Args:
                W: W matrix (term features).
                H: H matrix (document features).
            """

            self.W = sparse.csr_matrix(W)
            self.H = sparse.csc_matrix(H)


        def get_sparse_mini_batch_error(M_row_batch, batch_col_index, W_batch,\
                                        H_batch):
            """ Calculates error matrix using sparse matrix mini-batches.

            Args:
                M_row_batch: The mini-batch of the matrix being decomposed.
                batch_col_index: The indices used to pick the mini-batch.
                W_batch: The mini-batch of the term features.
                H_batch: The mini-batch of the document features.

            Returns: The error as a sparse matrix.
            """

            E = sparse.csr_matrix(\
                (M_row_batch[:, batch_col_index]).get_shape(),\
                dtype='float32')
            E = M_row_batch[:, batch_col_index] - W_batch.dot(H_batch)

            return E

        # Calculate error matrix using dense matrix mini-batches
        def get_dense_mini_batch_error(M_row_batch, batch_col_index, W_batch,\
                                       H_batch):

            """ Calculates error matrix using dense matrix mini-batches.

            Args:
                M_row_batch: The mini-batch of the matrix being decomposed.
                batch_col_index: The indices used to pick the mini-batch.
                W_batch: The mini-batch of the term features.
                H_batch: The mini-batch of the document features.

            Returns: The error as a dense matrix.
            """

            E = np.zeros(M_row_batch[:, batch_col_index].shape)
            E = M_row_batch[:, batch_col_index] - W_batch.dot(H_batch)

            return E

        # General optimization inits
        max_iter = self.als_opts['max_iter']
        check_iter = self.als_opts['check_iter']
        err = np.zeros(int(np.floor(max_iter/check_iter)))
        iter_err = None

        W = self.W
        self.W = None
        H = None
        E = None

        # Mini-batch optimization inits
        patience = self.als_opts['patience']
        patience_increase = self.als_opts['patience_increase']
        best_err = np.Inf
        converged = False

        # Conventional optimization inits
        tolerance = self.als_opts['tolerance']
        tol = None
        row_batch_size = self.als_opts['row_batch_size']
        col_batch_size = self.als_opts['col_batch_size']

        # Iteration counter
        i = 1

        # Start timer
        big_tic = time.time()

        print '---------------------------------------------------------------'
        print 'Beginning NMF using the %s method ... ' % (self.method)

        # Init matrices
        if (self.method in ['DENSE', 'DENSE_MINI_BATCH']):
            W_dense = np.zeros(W.get_shape(), dtype='float32')
            W.todense(out=W_dense)
            del W
            M_dense = np.zeros(self.M.get_shape(), dtype='float32')
            self.M.todense(out=M_dense)
            self.M = None
            L_dense = np.zeros(self.L.get_shape(), dtype='float32')
            self.L.todense(out=L_dense)
            self.L = None
            H_dense = np.zeros((self.n_features, M_dense.shape[1]),\
                               dtype='float32')
        elif (self.method in ['SPARSE', 'SPARSE_MINI_BATCH']):
            H = sparse.csc_matrix((self.n_features, self.M.get_shape()[1]),\
                                   dtype='float32')
        else:
            print 'ERROR: Undefined NMF method %s.' % (self.method)
            print 'NMF method must be one of: "DENSE",\
                   "DENSE_MINI_BATCH" (experimental), "SPARSE" (default),\
                   "SPARSE_MINI_BATCH" (experimental).'
            exit(-1)

        if self.method == 'SPARSE' or self.method == 'DENSE':
            if self.method == 'DENSE' and (row_batch_size == None and\
                col_batch_size == None):
                E = np.zeros(M_dense.shape, dtype='float32')
            if self.method == 'SPARSE' and (row_batch_size == None and\
                col_batch_size == None):
                E = sparse.csr_matrix(self.M.get_shape(), dtype='float32')

        # Warn about mini-batch training
        if self.method == 'SPARSE_MINI_BATCH' or\
            self.method == 'DENSE_MINI_BATCH':

            print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            print '! THIS METHOD IS EXPERIMENTAL                             !'
            print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

        # Mini-batch compatible optimization loop
        while i <= max_iter and not converged:

            tic = time.time()

            if self.method == 'SPARSE_MINI_BATCH':

                batch_row_index = np.sort(np.random.randint(\
                    self.M.get_shape()[0], size=row_batch_size))
                batch_col_index = np.sort(np.random.randint(\
                    self.M.get_shape()[1], size=col_batch_size))

                # Calculate H matrix from a random mini-batch using sparse
                # matrix calculations
                M_row_batch = self.M[batch_row_index]
                W_batch = W[batch_row_index]
                W_batchT = W_batch.transpose()
                H = (linalg.inv(W_batchT.dot(W_batch) +\
                    self.L)).dot(W_batchT.dot(M_row_batch))
                del W_batchT
                del W_batch
                H[H < 0] = 0
                H = sparse.csc_matrix(H, dtype='float32')
                H.eliminate_zeros()

                # Calculate W matrix from a random mini-batch using sparse
                # matrix calculations
                H_batch = H[:, batch_col_index]
                H_batchT = H_batch.transpose()
                M_col_batchT = self.M[:, batch_col_index].transpose()
                W = ((linalg.inv(H_batch.dot(H_batchT) +\
                    self.L)).dot(H_batch.dot(M_col_batchT))).transpose()
                W_batch = W[batch_row_index]
                del M_col_batchT
                del H_batchT
                W[W < 0] = 0
                W = sparse.csr_matrix(W, dtype='float32')
                W.eliminate_zeros()

                toc = time.time()-tic
                print 'Iteration %d complete in %.2f s...' % (i, toc)

                # Iteration mini-batch error/L1 norm
                if np.mod(i, check_iter) == 0:
                    E = get_sparse_mini_batch_error(M_row_batch,\
                        batch_col_index, W_batch, H_batch)
                    iter_err = np.sum(np.abs(E.data))
                    del E

                # Free memory
                del M_row_batch
                del W_batch
                del H_batch

            if self.method == 'DENSE_MINI_BATCH':

                batch_row_index = np.sort(np.random.randint(M_dense.shape[0],\
                    size=row_batch_size))
                batch_col_index = np.sort(np.random.randint(M_dense.shape[1],\
                    size=col_batch_size))

                # Calculate H matrix from a random mini-batch using dense
                # matrix calculations
                M_row_batch = M_dense[batch_row_index]
                W_batch = W_dense[batch_row_index]
                H_dense = (scipy.linalg.inv((W_batch.T).dot(W_batch) +\
                    L_dense)).dot((W_batch.T).dot(M_row_batch))
                H_dense[H_dense < 0] = 0
                del W_dense
                del W_batch

                # Calculate W matrix from a random mini-batch using dense
                # matrix calculations
                M_col_batch = M_dense[:, batch_col_index]
                H_batch = H_dense[:, batch_col_index]
                W_dense = ((scipy.linalg.inv(H_batch.dot(H_batch.T) +\
                    L_dense)).dot(H_batch.dot(M_col_batch.T))).T
                W_dense[W_dense < 0] = 0
                W_batch = W_dense[batch_row_index]

                toc = time.time()-tic
                print 'Iteration %d complete in %.2f s...' % (i, toc)

                # Iteration mini-batch error/L1 norm
                if np.mod(i, check_iter) == 0:
                    E = get_dense_mini_batch_error(M_row_batch,\
                        batch_col_index, W_batch, H_batch)
                    iter_err = np.abs(E).sum()
                    del E

                # Free memory
                del M_row_batch
                del M_col_batch
                del W_batch
                del H_batch

            if self.method == 'SPARSE':

                # Calculate H matrix using sparse matrix calculations
                WT = W.transpose()
                H = (linalg.inv(WT.dot(W) + self.L)).dot(WT.dot(self.M))
                del WT
                H[H < 0] = 0
                H = sparse.csc_matrix(H, dtype='float32')
                H.eliminate_zeros()

                # Calculate W matrix using sparse matrix calculations
                WT = (linalg.inv(H.dot(H.transpose()) +\
                      self.L)).dot(H.dot(self.M.transpose()))
                W = WT.transpose()
                del WT
                W[W < 0] = 0
                W = sparse.csr_matrix(W, dtype='float32')
                W.eliminate_zeros()

                toc = time.time()-tic
                print 'Iteration %d complete in %.2f s...' % (i, toc)

                # Evaluate iteration error/Frobenius norm
                if np.mod(i, check_iter) == 0:
                    if row_batch_size == None and col_batch_size == None:
                        E = self.M - W.dot(H)
                    else:
                        batch_row_index =\
                          np.unique(np.sort(np.random.randint(self.M.shape[0],\
                          size=row_batch_size)))
                        batch_col_index =\
                          np.unique(np.sort(np.random.randint(self.M.shape[1],\
                          size=col_batch_size)))
                        M_row_batch = self.M[batch_row_index]
                        W_batch = W[batch_row_index]
                        H_batch = H[:, batch_col_index]
                        E = get_sparse_mini_batch_error(M_row_batch,\
                            batch_col_index, W_batch, H_batch)
                    iter_err = np.sum(np.power(E.data, 2))
                    del E

            if self.method == 'DENSE':

                # Calculate H matrix using dense matrix calculations
                H_dense = scipy.linalg.inv((W_dense.T).dot(W_dense) +\
                    L_dense).dot((W_dense.T).dot(M_dense))
                H_dense[H_dense < 0] = 0
                del W_dense

                # Calculate W matrix using dense matrix calculations
                W_denseT = scipy.linalg.inv(H_dense.dot(H_dense.T) +\
                    L_dense).dot(H_dense.dot(M_dense.T))
                W_dense = W_denseT.T
                del W_denseT
                W_dense[W_dense < 0] = 0

                toc = time.time()-tic
                print 'Iteration %d complete in %.2f s...' % (i, toc)

                # Evaluate iteration error/Frobenius Norm
                if np.mod(i, check_iter) == 0:
                    if row_batch_size == None and col_batch_size == None:
                        E = M_dense - W_dense.dot(H_dense)
                    else:
                        batch_row_index =\
                         np.unique(np.sort(np.random.randint(M_dense.shape[0],\
                         size=row_batch_size)))
                        batch_col_index =\
                         np.unique(np.sort(np.random.randint(M_dense.shape[1],\
                         size=col_batch_size)))
                        M_row_batch = M_dense[batch_row_index]
                        W_batch = W_dense[batch_row_index]
                        H_batch = H_dense[:, batch_col_index]
                        E = get_dense_mini_batch_error(M_row_batch,\
                            batch_col_index, W_batch, H_batch)
                    iter_err = np.sum(np.power(E, 2))
                    del E

            # If error was calculated ...
            if iter_err != None:

                # Evaluate iteration results
                if np.mod(i, check_iter) == 0:
                    print 'Error at iteration %d: %f ...' % (i, iter_err)
                    if iter_err < best_err:
                        if patience != None:
                            patience = max(patience, i+patience_increase)
                        best_err = iter_err
                        self.best_err = iter_err
                        best_iter = i
                    err[(i/check_iter)-1] = iter_err

                # Evaluate stopping criteria
                if patience != None:
                    if patience <= i or i == max_iter:
                        converged = True
                        break

                if tolerance != None:
                    if tol == None:
                        tol = iter_err*tolerance
                    if (i/check_iter) > 1:
                        if err[(i/check_iter)-2]-err[(i/check_iter)-1] < tol:
                            best_iter = i
                            best_err = iter_err
                            self.best_err = iter_err
                            break
            i = i+1

        if (self.method in ['DENSE', 'DENSE_MINI_BATCH']):
            mf_to_instance_var(W_dense, H_dense)
            del W_dense
            del H_dense
            del M_dense
        else:
            mf_to_instance_var(W, H)
            del W
            del H

        # Output total time and error plot
        big_toc = time.time()-big_tic
        print 'NMF completed in %f s.' % (big_toc)
        if self.best_err != None:
            plot_error(self.method, toc, err, best_iter, best_err, check_iter)


    def write_output(self, working_dir):
        """ Write out W and H to file.

        Args:
            working_dir: The directory to which to write W and H.
        """

        time_stamp =\
          datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        self.set_w_fname(working_dir + os.sep + 'W_' + time_stamp + '.txt')
        self.set_h_fname(working_dir + os.sep + 'H_' + time_stamp + '.txt')

        try:
            np.savetxt(self.get_w_fname(), self.get_w().todense(), fmt='%.4f')
            np.savetxt(self.get_h_fname(), self.get_h().todense(), fmt='%.4f')
        except:
            print 'ERROR: Could not save matrix factors to file.'
            print sys.exc_info()
            exit(-10)

def main(argv):

    """ If class is used stand-alone.

    Args:
        argv: Command line args.

    Raises:
        GetoptError: Error parsing command line options.
    """

    ### Init optional command line args
    working_dir = None
    n_features = None
    method = None
    als_opts = None
    random_seed = None
    tfidf = True

    try:
        opts, _ = getopt.getopt(argv, "d:k:m:o:r:w:h")
        for opt, arg in opts:
            if opt == '-d':
                working_dir = arg
            elif opt == '-k':
                n_features = arg
            elif opt == '-m':
                method = arg
            elif opt == '-o':
                als_opts = ast.literal_eval(arg)
            elif opt == '-r':
                random_seed = arg
            elif opt == '-w':
                tfidf = ast.literal_eval(arg)
            elif opt == '-h':
                print 'NMF.py \n \
-d <directory containing row, col, val .txt (string)>\n \
-k <positive integer number of features (int)>\n \
-m <method (string)> (default = "SPARSE")\n \
-w <TF-IDF weighting (boolean)> (default = True) \n \
-o <ALS options (dictionary)> (default = {"l": 0.5, "max_iter": 50, "check_iter": 1, "tolerance": 0.01, "patience": None, "patience_increase": None, "row_batch_size": None, "col_batch_size": None})\n \
-r <random seed (int)> (default 12345)'
                exit(-1)
    except getopt.GetoptError:
        print '---------------------------------------------------------------'
        print 'working_dir          = %s' % (working_dir)
        print 'n_features           = %s' % (n_features)
        print 'method               = %s' % (method)
        print 'tfidf                = %s' % (tfidf)
        print 'als_opts             = %s' % (als_opts)
        print 'random_seed          = %s' % (random_seed)
        print '---------------------------------------------------------------'
        print 'NMF.py \n \
-d <directory containing row, col, val .txt (string)>\n \
-k <positive integer number of features (int)>\n \
-m <method (string)> (default = "SPARSE")\n \
-w <TF-IDF weighting (boolean)> (default = True) \n \
-o <ALS options (dictionary)> (default = {"l": 0.5, "max_iter": 50, "check_iter": 1, "tolerance": 0.01, "patience": None, "patience_increase": None, "row_batch_size": None, "col_batch_size": None})\n \
-r <random seed (int)> (default 12345)'
        exit(-1)

    ### INPUT PROTECTION ##################################################

    ### ATTEMPT TO LOAD DATA (-d)
    try:

        tic = time.time()
        print '---------------------------------------------------------------'
        print 'Loading data ...'
        ### Highest term count is 2147483647 for int32
        row = np.loadtxt(working_dir + os.sep + 'row.txt', dtype='int32',\
            skiprows=1)
        col = np.loadtxt(working_dir + os.sep + 'col.txt', dtype='int32',\
            skiprows=1)
        val = np.loadtxt(working_dir + os.sep + 'val.txt', dtype='int32',\
            skiprows=1)

        ### CONSTRUCT SPARSE MATRIX
        M = sparse.coo_matrix((val, (row, col)), shape=(np.amax(row)+1,\
            np.amax(col)+1), dtype='float32').tocsc()
        del row
        del col
        del val

        toc = time.time()-tic
        print 'Data loaded in %f s.' % (toc)

    except:
        print 'ERROR: Problem loading row.txt, col.txt, or val.txt from %s and converting to Scipy sparse matrix.' % (working_dir)
        print 'Check mandatory switch -d (data directory).'
        print sys.exc_info()
        exit(-1)

    ### CHECK k (-k)
    try:
        if n_features > 0:
            n_features = int(n_features)
        else:
            print 'Mandatory switch -k (number of features) must be an integer that is greater than 0.'
            exit(-1)
    except:
        print 'Mandatory switch -k (number of features) must be a positive integer (>0).'
        print sys.exc_info()
        exit(-1)

    ### CHECK method (-m)
    if ((method != None) and method not in ['DENSE', 'DENSE_MINI_BATCH', 'SPARSE', 'SPARSE_MINI_BATCH']):
        print 'Optional switch -m (NMF method) must be one of: "DENSE", "DENSE_MINI_BATCH", "SPARSE" (default), "SPARSE_MINI_BATCH" (experimental).'
        exit(-1)

    ### CHECK als_opts (-o)
    if als_opts != None:
        if type(als_opts) is not dict or len(als_opts) != 8:
            print 'Optional switch -o (ALS options) must be a dictionary:\n \
Example for mini-batch training: \n \
{"l": 0.5, "max_iter": 50, "check_iter": 1, tolerance: "None", "patience": 10, "patience_increase": 2, "row_batch_size": <~80% of rows>, "col_batch_size": <~50% of cols>} \n \
Example for conventional training without mini-batch error: \n \
{"l": 0.5, "max_iter": 50, "check_iter":1, "tolerance": 0.01, "patience": None, "patience_increase": None, "row_batch_size": None, "col_batch_size": None} (default) \n \
Example for conventional training with mini-batch error: \n \
{"l": 0.5, "max_iter": 50, "check_iter":1, tolerance: "None", "patience": 10, "patience_increase": 2, "row_batch_size": <10% of  rows>, "col_batch_size": <10% of cols>}'
            exit(-1)

    ### CHECK random_seed
    if random_seed != None:
        try:
            random_seed = int(random_seed)
        except:
            print 'Optional switch -r (random seed) must be an integer.'
            print sys.exc_info()
            exit(-1)

    if tfidf != True and tfidf != False:
        print 'Optional switch -w (TF-IDF weighting) must be True or False.'
        print sys.exc_info()
        exit(-1)

    print '---------------------------------------------------------------'
    print 'Proceeding with options: '
    print 'working_dir          = %s' % (working_dir)
    print 'n_features           = %s' % (n_features)
    print 'method               = %s' % (method)
    print 'tfidf                = %s' % (tfidf)
    print 'als_opts             = %s' % (als_opts)
    print 'random_seed          = %s' % (random_seed)

    nmf = NMF(M, n_features, method, tfidf, als_opts, random_seed)
    nmf.nmf()
    nmf.write_output(working_dir)


### Call main with args if class is used stand-alone
if __name__ == '__main__':
    main(sys.argv[1:])
