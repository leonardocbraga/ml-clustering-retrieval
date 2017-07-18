import numpy as np
import sys
import os
import graphlab

from preprocess import *
from kmeans import *
from visualization import *

# load wikipedia data
wiki = graphlab.SFrame('../../data/people_wiki.gl/')

# preprocess TF IDF structure
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['text'])

# transform into a sparse matrix
tf_idf, map_index_to_word = sframe_to_scipy(wiki, 'tf_idf')

# then normalize
tf_idf = normalize(tf_idf)

# test kmeans with 3 cluster centers
k = 3
heterogeneity = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400, record_heterogeneity=heterogeneity, verbose=True)

# plot heterogeneity of 3 clusters on wiki data 
plot_heterogeneity(heterogeneity, k)

# print each cluster size
np.bincount(cluster_assignment)

# test kmeans with 10 cluster centers and different seeds
k = 10
heterogeneity = {}
import time
start = time.time()
for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    initial_centroids = get_initial_centroids(tf_idf, k, seed)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)

    print 'Max number: ', np.max(np.bincount(cluster_assignment))

    # To save time, compute heterogeneity only once in the end
    heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
    print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
    sys.stdout.flush()
end = time.time()
print(end-start)

# load preprocessed kmeans results
filename = '../../data/kmeans-arrays.npz'

heterogeneity_values = []
k_list = [2, 10, 25, 50, 100]

# plot K vs heterogeneity value
if os.path.exists(filename):
    arrays = np.load(filename)
    centroids = {}
    cluster_assignment = {}
    for k in k_list:
        print k
        sys.stdout.flush()

        centroids[k] = lambda k=k: arrays['centroids_{0:d}'.format(k)]
        cluster_assignment[k] = lambda k=k: arrays['cluster_assignment_{0:d}'.format(k)]
        score = compute_heterogeneity(tf_idf, k, centroids[k](), cluster_assignment[k]())
        heterogeneity_values.append(score)
    
    plot_k_vs_heterogeneity(k_list, heterogeneity_values)

else:
    print('File not found. Skipping.')

# visualize document clusters from kmeans result with 10 clusters
k = 10
visualize_document_clusters(wiki, tf_idf, centroids[k](), cluster_assignment[k](), k, map_index_to_word)

# print each cluster size from keamns-10
np.bincount(cluster_assignment[10]())

# visualize document clusters from kmeans result with 100 clusters
k=100
visualize_document_clusters(wiki, tf_idf, centroids[k](), cluster_assignment[k](), k, map_index_to_word, display_content=False)
