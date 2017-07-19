import numpy as np
import graphlab
import time

from lsh import *
from preprocess import *
from visualization import *

# load wikipedia data
wiki = graphlab.SFrame('../../data/people_wiki.gl/')
wiki = wiki.add_row_number()

# preprocess TF IDF of Documents
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['text'])

# convert to sparse matrix
start=time.time()
corpus, mapping = sframe_to_scipy(wiki['tf_idf'])
end=time.time()
print end-start

# train LSH model
model = train_lsh(corpus, num_vector=16, seed=143)

# retrieve bin indices from model
bin_indices = model['bin_indices']

# retrieve bin index bits from model
bin_index_bits = model['bin_index_bits']

# get Obama's article id and find bin that contains Barack Obama's article
obama_id = wiki[wiki['name'] == 'Barack Obama']['id'][0]
bin_indices[obama_id]

# get Biden's article and find how many places they agree
biden_id = wiki[wiki['name'] == 'Joe Biden']['id'][0]
(bin_index_bits[obama_id] == bin_index_bits[biden_id]).sum()

# run LSH multiple times, each with different radius for nearby bin search
num_candidates_history = []
query_time_history = []
max_distance_from_query_history = []
min_distance_from_query_history = []
average_distance_from_query_history = []

for max_search_radius in xrange(17):
    start=time.time()
    result, num_candidates = query(corpus[35817,:], model, k=10,
                                   max_search_radius=max_search_radius)
    end=time.time()
    query_time = end-start
    
    print 'Radius:', max_search_radius
    
    average_distance_from_query = result['distance'][1:].mean()
    max_distance_from_query = result['distance'][1:].max()
    min_distance_from_query = result['distance'][1:].min()
    
    num_candidates_history.append(num_candidates)
    query_time_history.append(query_time)
    average_distance_from_query_history.append(average_distance_from_query)
    max_distance_from_query_history.append(max_distance_from_query)
    min_distance_from_query_history.append(min_distance_from_query)

    print average_distance_from_query

# plot num candidates at each iteration
plot_num_candidates(num_candidates_history)

# plot query time at each iteration
plot_query_time(query_time_history)

# plot neighbors distances (avg, min and max) at each iteration
plot_neighbors_distances(average_distance_from_query_history, max_distance_from_query_history, min_distance_from_query_history)
