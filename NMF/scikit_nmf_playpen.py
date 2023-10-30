# -*- coding: utf-8 -*-
"""
jpatrickhall@gmail.com
11.8.14
Educational use only.
"""

import os
import iopro
import numpy as np
from scipy import sparse
from time import time
from datetime import datetime
from sklearn.decomposition import ProjectedGradientNMF

class SkNMF(object):
    """
    Simple class for experimenting with scikit learn projected gradient NMF.
    Set all params here (in this file).

    Attributes:
        h_matrix: Matrix of document features.
        best_err: Best error from NMF optimization.

    Scikit doc:
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF\
        .html#sklearn.decomposition.NMF.set_params
    """

    def __init__(self):
        """ Initiate attributes. """

        self.h_matrix = None
        self.best_err = None

    def get_h_matrix(self):
        """ Getter for H. """

        return self.h_matrix

    def get_best_err(self):
        """ Getter for best_err """

        return self.best_err

    def nmf(self):
        """ Run projected gradient NMF from scikit-learn.

        Can be run on simulated data or data loaded from row, col, val text
        files.

        """

        data_dir = 'C:\\Temp'       # Location for data from data_in= 'LOAD'
        sim_row = 25                # Simulation params for data_in= 'SIM'
        sim_col = 2500
        data_in = 'SIM'
        k = 6                       # Number of features

        ### INPUT DATA
        tic = time()
        print '---------------------------------------------------------------'
        print 'Loading data ...'

        if data_in == 'LOAD':

            row = iopro.loadtxt(data_dir + os.sep + 'row.txt', dtype='int32',\
                skiprows=1)
            col = iopro.loadtxt(data_dir + os.sep + 'col.txt', dtype='int32',\
                skiprows=1)
            val = iopro.loadtxt(data_dir + os.sep + 'val.txt', dtype='int32',\
                skiprows=1)

            ### CONSTRUCT SPARSE MATRIX REPRESENTATION
            matrix = sparse.coo_matrix((val, (row, col)),\
                                       shape=(np.amax(row)+1,\
                                              np.amax(col)+1)).tocsr()
            del row
            del col
            del val

        if data_in == 'SIM':
            np.random.seed(12345)
            matrix = sparse.rand(sim_row, sim_col, format='csr')
            matrix = (matrix*1000).ceil()

        toc = time()-tic
        print 'Data loaded in %f s.' % (toc)

        ### NMF
        tic = time()
        model = ProjectedGradientNMF(n_components=k, init='nndsvdar',\
                                     random_state=0)
        print '---------------------------------------------------------------'
        print 'Beginning NMF with parameters: %s ...' % (model.get_params())
        model.fit(matrix)
        self.h_matrix = model.components_
        self.best_err = model.reconstruction_err_
        toc = time()-tic
        print 'NMF completed in %f s.' % (toc)
        print 'Frobenius norm = %s.' % (str(self.best_err))
        time_stamp = datetime.fromtimestamp(time())\
            .strftime('%Y-%m-%d_%H-%M-%S')
        np.savetxt(data_dir + os.sep + 'H_' + time_stamp + '.txt',\
                   self.h_matrix, fmt='%.4f')

def main():

    """ Instantiate and execute NMF. """

    nmf = SkNMF()
    nmf.nmf()

if __name__ == '__main__':
    main()
