import graphlab
import numpy as np

def top_words(wiki, name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False)

def top_words_tf_idf(wiki, name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)

# load wikipedia data
wiki = graphlab.SFrame('../../data/people_wiki.gl')

# preprocess word count structure
wiki['word_count'] = graphlab.text_analytics.count_words(wiki['text'])

# create nearest neighbors model using word counts
model = graphlab.nearest_neighbors.create(wiki, label='name', features=['word_count'], method='brute_force', distance='euclidean')

# query 10 Obama's nearest neighbors
model.query(wiki[wiki['name']=='Barack Obama'], label='name', k=10)

# get top words on Obama's and Barrio's articles
obama_words = top_words(wiki, 'Barack Obama')

barrio_words = top_words(wiki, 'Francisco Barrio')

# combine sets
combined_words = obama_words.join(barrio_words, on='word')
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})

# generate set of common words in Obama's and Barrio's articles
common_words = set(combined_words.sort('Obama', ascending=False)['word'][0:5])

# apply function to dataset to know how many registers contains top words
wiki['has_top_words'] = wiki['word_count'].apply(lambda word_count_vector: True if common_words.issubset(set(word_count_vector.keys())) else False)

(wiki['has_top_words'] == 1).sum()

# compute euclidean distance between Obama's, Bush's and Biden's pages
obama_page = wiki[wiki['name']=='Barack Obama'][0]
bush_page = wiki[wiki['name']=='George W. Bush'][0]
biden_page = wiki[wiki['name']=='Joe Biden'][0]

graphlab.distances.euclidean(obama_page['word_count'], bush_page['word_count'])
graphlab.distances.euclidean(obama_page['word_count'], biden_page['word_count'])
graphlab.distances.euclidean(bush_page['word_count'], biden_page['word_count'])

# compare Obama's article to Bush's article via top words
obama_words = top_words(wiki, 'Barack Obama')
bush_words = top_words(wiki, 'George W. Bush')

combined_words = obama_words.join(bush_words, on='word')
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Bush'})
set(combined_words.sort('Obama', ascending=False)['word'][0:10])

# preprocess TF IDF structure
wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['word_count'])

# create nearest neighbors model using TF IDF
model_tf_idf = graphlab.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                                 method='brute_force', distance='euclidean')

# query 10 nearest neighbors of Obama
model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10)

# get top words on Obama's and Schiliro's articles
obama_tf_idf = top_words_tf_idf(wiki, 'Barack Obama')
schiliro_tf_idf = top_words_tf_idf(wiki, 'Phil Schiliro')

combined_words = obama_tf_idf.join(schiliro_tf_idf, on='word')
combined_words = combined_words.rename({'weight':'Obama', 'weight.1':'Schiliro'})
combined_words.sort('Obama', ascending=False)

common_words = set(combined_words.sort('Obama', ascending=False)['word'][0:5])

# apply function to dataset to know how many registers contains top words using TF IDF
wiki['has_top_words'] = wiki['word_count'].apply(lambda word_count_vector: True if common_words.issubset(set(word_count_vector.keys())) else False)

(wiki['has_top_words'] == 1).sum()

# compute euclidean distance of Obama's and Biden's articles using TF IDF
obama_page = wiki[wiki['name']=='Barack Obama'][0]
biden_page = wiki[wiki['name']=='Joe Biden'][0]

graphlab.distances.euclidean(obama_page['tf_idf'], biden_page['tf_idf'])
