# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 12:22:15 2014
@author: p
Educational use only.
"""

c = post.segmenter.fit_predict(post.h_matrix.T)
cr = c.reshape(10000,1)
crt = cr.astype(str)
for i in range(0,10000):
    crt[i,0] = 'cluster '+crt[i,0]
crt = numpy.hstack((post.h_matrix.T, crt))   
fn = feature_names.values()
fn.append('label')
import pandas
crtdf = crtdf.convert_objects(convert_numeric=True)
fig = pandas.tools.plotting.parallel_coordinates(crtdf, 'Label', colormap='prism')
fig.legend_.remove()
fig.grid(False)
fig.axes.get_yaxis().set_visible(False)
