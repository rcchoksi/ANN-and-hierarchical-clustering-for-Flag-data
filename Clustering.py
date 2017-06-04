# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 01:16:13 2017

@author: choks_000
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, maxdists
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

#Importing data from csv file
file = 'Flagdata.csv'
df = pd.read_csv(file)

#Scaling the data
scaler = StandardScaler()
scaler.fit(df)
data = scaler.transform(df)

#Applying the clustering algorithm using average distance method
z = linkage(data, "average")

#Calculating cophenet correlation coefficient 
c, coph_dists = cophenet(z, pdist(data))
print('Cophenet Correlation coefficient = ', c)
print('Cophenet pairwise distances = ', coph_dists)

#Printing the first two points merged and the distance between them
print("1st Cluster is ", z[0])

#Distance array
m = maxdists(z)
print("Distance Array ", m)

#Plotting the full dendogram
plt.figure(figsize=(30, 15))
plt.title('Dendogram for Flag data')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=10.,  # font size for the x axis labels
)
plt.show()

#Plotting a truncated dendogram showing last 12 cluster iterations
plt.figure(figsize=(20, 10))
dendrogram(
    z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

