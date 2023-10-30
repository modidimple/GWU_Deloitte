# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 20:10:21 2014
@author: p
Educational use only.
"""
#%%
#import pandas

#HT = numpy.genfromtxt('c:/users/p/szl.it/wdense.csv', delimiter=',', skip_header=1)

OUT_FILE = 'C:\\Temp\\graph.gdf'

#HT = post.h_matrix.T
#fn = feature_names.values()
#HTCorr = pandas.DataFrame(HT, columns = fn).corr().as_matrix()

with open(OUT_FILE, 'w+') as out_file_:
    out_file_.write('nodedef>name VARCHAR,label VARCHAR\n')
    for i in range(0,850):
        out_file_.write(str(i) + ',' + fn[i] + '\n')
    
    out_file_.write('edgedef>node1 VARCHAR,node2 VARCHAR, weight DOUBLE\n')
    for i in range(0,850):
        for j in range(0,850):
            if HTCorr[i,j] > 0.0001 and i != j:
                out_file_.write(str(i) + ',' + str(j) + ',' +\
                                    str(HTCorr[i, j]) + '\n')
       
#%%
    
fn = pandas.read_csv('c:/users/p/szl.it/wdense.csv').columns    
fn = [x.encode('ascii') for x in fn]
HT = numpy.genfromtxt('c:/users/p/szl.it/wdense.csv', delimiter=',', skip_header=1)
HTCorr = numpy.corrcoef(HT.T)
   
df = pandas.read_csv('C:\\Temp\\weights2.csv')
df.sort(['node1','weight'],ascending=[1,0], inplace=True)
df = df.groupby('node1').head(2)
df.to_clipboard()

#%%
fn= []

HT = numpy.zeros((2000, 850))
i = 0

with open('centroids.txt') as c:
   for line in c:
        line = line.split('", "')
        fn.append(line[0].replace('"', '').strip())
        a = numpy.asarray(ast.literal_eval(line[1].replace('"', '').strip()))        
        a = a.reshape(2000,)
        HT[::,i] = a
        i = i + 1
#%%
