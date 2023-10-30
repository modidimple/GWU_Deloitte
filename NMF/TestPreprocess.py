# -*- coding: utf-8 -*-
"""
jpatrickhall@gmail.com
11.8.14
Educational use only.
"""

import unittest
import Preprocess


class TestPreprocess(unittest.TestCase):
    """ Simple unit tests for Preprocess.py.

    Attributes:
        pre: Single-threaded instance of Preprocess class used for tests.

    """

    def setUp(self):
        """ Instantiates Preprocess to run all methods sequentially. """

        nthread = 1
        working_dir = 'C:\\Temp'
        raw_file = 'sample_data.txt'
        stop_dir = 'stoplist'
        threshold = 3
        custom_replacements = 'replacements.txt'
        self.pre = Preprocess.Preprocess(nthread, working_dir, raw_file,\
            stop_dir, threshold, custom_replacements)

    def test_all(self):

        """ Runs all methods in sequence on sample data. """

        self.pre.lower(0)
        self.pre.regex_replace_common_terms(0)
        self.pre.lemmatize(0)
        self.pre.resolve_replacements()
        self.pre.replace_terms(0)
        self.pre.apply_stoplist(0)
        self.pre.remove_short_terms(0)
        counts = self.pre.get_counts(self.pre.nthread)
        self.pre.remove_infrequent_terms(0, counts)
        terms = self.pre.get_unique_terms()
        self.pre.set_terms(terms)
        self.pre.write_tbd()


def main():

    """ Run tests. """

    unittest.main()


if __name__ == '__main__':
    main()
