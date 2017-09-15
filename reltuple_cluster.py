#Ade Romadhony
#Cluster hasil encode vektor subject-trigger-object

import os
import numpy as np
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.decomposition import PCA
import sys

def read_tuple_data(input_path):
    reltuple_list = []
    with open(input_path,"r") as fi_reltuple:
        words = fi_reltuple.readlines();
        content = [x.strip() for x in words]
        lenLines = len(content)
        for countLines in range(0,lenLines):
            reltuple_list.append(words[countLines].strip())
    fi_reltuple.close()
    return reltuple_list

#load encoded data
X = np.load("encoded_relevantsample_1000_30000.npy")

#coba 3 metode clustering, spectral, k-means, dan agglomerative
#jumlah cluster diset hardcoded (n_clusters)

label = cluster.SpectralClustering(n_clusters=50 ,affinity='nearest_neighbors').fit_predict(X[0])
print("label spectral = ")
print(label)

kmeans = cluster.KMeans(n_clusters=50, random_state=0).fit(X[0])
print("label kmeans = ")
print(kmeans.labels_)

agglomerative = cluster.AgglomerativeClustering(n_clusters=50, affinity='euclidean', linkage='ward').fit_predict(X[0])
print("label agglomerative = ")
print(agglomerative)

reltuple = read_tuple_data("relevant_reltuple_sample_1000.txt")
for idx_cluster_agglomerative in range(0, 50):
    print("**************************************************************************")
    print("anggota cluster ke-"+str(idx_cluster_agglomerative))
    #sys.stdout.write("\r anggota cluster ke-{}".format(idx_cluster_agglomerative))
    #sys.stdout.flush()
    for idx_reltuple in range(0,1000):
        if(agglomerative[idx_reltuple]==idx_cluster_agglomerative):
            print(reltuple[idx_reltuple])
    print("**************************************************************************")
