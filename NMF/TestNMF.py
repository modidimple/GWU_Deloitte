# -*- coding: utf-8 -*-
"""
jpatrickhall@gmail.com
11.8.14
Educational use only.
"""

import unittest
import NMF
import numpy as np
from scipy import sparse


class TestNMF(unittest.TestCase):
    """ Simple unit tests for NMF.py

    Test different NMF methods including methods for:
        Dense matrix decomposition.
        Sparse matrix decomposition.
        Mini-batch decomposition.
        Checking the error (very expensive) at different iterations.

    Attributes:
        random_seed: Random seed for initialization and mini-batches.
        sim_row: Number of rows in the simulated matrix for this class.
        sim_col: Number of columns in the simulated matrix for this class.
        matrix: Simulated matrix to be decomposed.
        k: Number of features to extract.
        sparse_err: The best reconstruction error in a sparse matrix
            decomposition.
        dense_err: The best reconstruction error in a dense matrix
            decomposition.
        dense_mini_batch_err: The best reconstruction error in a dense
            matrix mini-batch decomposition.

    """

    def setUp(self):
        """ Set test parameters here. """

        self.random_seed = 12345
        np.random.seed(self.random_seed)
        self.sim_row = 100
        self.sim_col = 10000
        self.matrix = (sparse.rand(self.sim_row, self.sim_col, \
                                   format='csr') * 1000).ceil()
        self.k = 20
        self.sparse_err = None
        self.dense_err = None
        self.dense_mini_batch_err = None

    def test_sparse(self):
        """ Unit test for sparse matrix decomposition. """

        nmf = NMF.NMF(self.matrix, self.k, 'SPARSE', None, None, None)
        nmf.nmf()
        self.assertEquals(nmf.get_w().get_shape(), \
                          (self.matrix.get_shape()[0], self.k))
        self.sparse_err = nmf.get_best_err()
        print '---------------------------------------------------------------'
        print 'SPARSE BEST ERROR: %.2f' % (self.sparse_err)


#    def test_dense(self):
#        """ Unit test for dense matrix decomposition. """
#
#        nmf = NMF.NMF(self.matrix, self.k, 'DENSE', False, None, 12345)
#        nmf.nmf()
#        self.assertEquals(nmf.get_w().get_shape(),\
#                          (self.matrix.get_shape()[0], self.k))
#        self.dense_err = nmf.get_best_err()
#        print '---------------------------------------------------------------'
#        print 'DENSE BEST ERROR: %.2f' % (self.dense_err)

#    def test_dense_check_iter_mb_error(self):
#        """ Unit test for:
#            Dense matrix decomposition.
#            Using minibatches.
#            Checking error at check_iter iterations.
#        """

#        nmf = NMF.NMF(self.matrix, self.k, 'DENSE', True, {'l': 0.5,\
#                            'max_iter': 50,\
#                            'check_iter': 2,\
#                            'tolerance': None,\
#                            'patience': 15,\
#                            'patience_increase' : 2,\
#                            'row_batch_size': np.floor(0.3*self.sim_row),\
#                            'col_batch_size': np.floor(0.3*self.sim_col)},\
#                            12345)
#        nmf.nmf()
#        self.assertEquals(nmf.get_w().get_shape(),\
#                          (self.matrix.get_shape()[0], self.k))
#        error = nmf.get_best_err()
#        print '---------------------------------------------------------------'
#        print 'DENSE BEST MB ERROR: %.2f' % (error)


#    def test_sparse_check_iter_mb_error(self):
#        """ Unit test for:
#            Sparse matrix decomposition.
#            Using minibatches.
#            Checking error at check_iter iterations.
#        """
#
#        nmf = NMF.NMF(self.matrix, self.k, 'SPARSE', False, {'l': 0.5,\
#                            'max_iter': 50,\
#                            'check_iter': 2,\
#                            'tolerance': None,\
#                            'patience': 15,\
#                            'patience_increase' : 2,\
#                            'row_batch_size': np.floor(0.3*self.sim_row),\
#                            'col_batch_size': np.floor(0.3*self.sim_col)},\
#                            12345)
#        nmf.nmf()
#        self.assertEquals(nmf.get_w().get_shape(),\
#                          (self.matrix.get_shape()[0], self.k))
#        error = nmf.get_best_err()
#        print '---------------------------------------------------------------'
#        print 'SPARSE BEST MB ERROR: %.2f' % (error)

#    def test_sparse_mini_batch(self):
#        """ Unit test for:
#            Sparse matrix decomposition.
#            Using minibatches.
#        """
#
#        nmf = NMF.NMF(self.matrix, self.k, 'SPARSE_MINI_BATCH', True,\
#                          {'l': 0.5,\
#                            'max_iter': 50,\
#                            'check_iter': 1,\
#                            'tolerance': None,\
#                            'patience': 15,\
#                            'patience_increase': 2,\
#                            'row_batch_size': np.floor(self.sim_row),\
#                            'col_batch_size': np.floor(0.5*self.sim_col)},\
#                            self.random_seed)
#        nmf.nmf()
#        self.assertEquals(nmf.get_w().shape,\
#                          (self.matrix.get_shape()[0], self.k))
#        self.sparse_mini_batch_err = nmf.get_best_err()
#        print '---------------------------------------------------------------'
#        print 'SPARSE_MINI_BATCH BEST ERROR: %.2f'\
#            % (self.sparse_mini_batch_err)

#    def test_dense_mini_batch(self):
#        """ Unit test for:
#            Dense matrix decomposition.
#            Using minibatches.
#        """
#
#        nmf = NMF.NMF(self.matrix, self.k, 'DENSE_MINI_BATCH', False,\
#                          {'l': 0.5,\
#                            'max_iter': 50,\
#                            'check_iter': 1,\
#                            'tolerance': None,\
#                            'patience': 15,\
#                            'patience_increase': 2,\
#                            'row_batch_size': np.floor(self.sim_row),\
#                            'col_batch_size': np.floor(0.5*self.sim_col)},\
#                            self.random_seed)
#        nmf.nmf()
#        self.assertEquals(nmf.get_w().shape,\
#                          (self.matrix.get_shape()[0], self.k))
#        self.dense_mini_batch_err = nmf.get_best_err()
#        print '---------------------------------------------------------------'
#        print 'DENSE_MINI_BATCH BEST ERROR: %.2f' % (self.dense_mini_batch_err)

def main():
    """ Execute tests. """

    unittest.main()


if __name__ == '__main__':
    main()
