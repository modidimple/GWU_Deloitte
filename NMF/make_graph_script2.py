# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 20:10:21 2014
@author: p
Educational use only.
"""

import pandas
import numpy


def main():

    K = 54 # match to Batchdriver.py
    H = numpy.genfromtxt('C:\\Users\\imdim\\OneDrive\\Documents\\SEM3\\BPA\\nmf_Data\\H_2023-10-21_15-06-38.txt') # H file from C:\Temp
    OUT_FILE = 'C:\\Users\\imdim\\OneDrive\\Documents\\SEM3\\BPA\\nmf_Data\\graph.gdf'
    HT = H.T

    # paste in from output of Batchdriver.py
    fn = {
        0
        : 'identity+theft',
        1
        : 'credit',
        2
        : 'experian+monitor',
        3
        : 'law+enforcement',
        4
        : 'activity+review',
        5
        : 'security+freeze',
        6
        : 'protect+step',
        7
        : 'company+insurance',
        8
        : 'attorney+office',
        9
        : 'alert+fraud',
        10
        : 'social+security',
        11
        : 'server+box',
        12
        : 'card+payment',
        13
        : 'train+xgb',
        14
        : 'unauthorized+party',
        15
        : 'investigation+forensic',
        16
        : 'attorney+office',
        17
        : 'experian+monitor',
        18
        : 'breach+business',
        19
        : 'unauthorized+party',
        20
        : 'investigation+forensic',
        21
        : 'employee',
        22
        : 'social+security',
        23
        : 'system',
        24
        : 'customer',
        25
        : 'card+payment',
        26
        : 'personal',
        27
        : 'computer+laptop',
        28
        : 'incident+write',
        29
        : 'protect+step',
        30
        : 'personal',
        31
        : ' security+freeze',
        32
        : 'investigation+forensic',
        33
        : 'unauthorized+party',
        34
        : 'social+security',
        35
        : 'card+payment',
        36
        : 'breach+business',
        37
        : 'patient+medical',
        38
        : 'health+insurance',
        39
        : 'social+security',
        40
        : 'employee',
        41
        : 'server+box',
        42
        : ' protect+step',
        43
        : 'system',
        44
        : 'investigation+forensic',
        45
        : 'alert+fraud',
        46
        : 'credit',
        47
        : 'identity+theft',
        48
        : ' protect+step ',
        49
        : 'activity+review',
        50
        : 'security+freeze',
        51
        : 'company+insurance',
        52
        : 'experian+monitor',
        53
        : 'law+enforcement',
        54
        : 'social+security',
    }
    HTCorr = pandas.DataFrame(HT, columns = fn).corr().as_matrix()

    with open(OUT_FILE, 'w+') as out_file_:
        out_file_.write('nodedef>name VARCHAR,label VARCHAR\n')
        for i in range(0, K):
            out_file_.write(str(i) + ',' + fn[i] + '\n')

        out_file_.write('edgedef>node1 VARCHAR,node2 VARCHAR, weight DOUBLE\n')
        for i in range(0, K):
            for j in range(0, K):
                if HTCorr[i,j] > 0.1 and i != j:
                    out_file_.write(str(i) + ',' + str(j) + ',' +\
                                        str(HTCorr[i, j]) + '\n')


if __name__ == '__main__':
    main()
