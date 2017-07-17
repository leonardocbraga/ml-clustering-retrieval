import graphlab as gl
import numpy as np
import matplotlib.pyplot as plt
import graphlab.aggregate as agg

def average_predictions(model, themes, test_document, num_trials=100):
    avg_preds = np.zeros((model.num_topics))
    topic_numbers = range(model.num_topics)
    for i in range(num_trials):
        avg_preds += model.predict(test_document, output_type='probability')[0]
    avg_preds = avg_preds/num_trials
    result = gl.SFrame({'topics':themes, 'average predictions':avg_preds, 'topic_numbers': topic_numbers})
    result = result.sort('average predictions', ascending=False)
    return result

def plot_probabilities_top_100(topic_model):
    for i in range(10):
        plt.plot(range(100), topic_model.get_topics(topic_ids=[i], num_words=100)['score'])
    plt.xlabel('Word rank')
    plt.ylabel('Probability')
    plt.title('Probabilities of Top 100 Words in each Topic')
    plt.show()

def plot_probabilities_top_10(topic_model):
    top_probs = [sum(topic_model.get_topics(topic_ids=[i], num_words=10)['score']) for i in range(10)]

    ind = np.arange(10)
    width = 0.5

    fig, ax = plt.subplots()

    ax.bar(ind-(width/2),top_probs,width)
    ax.set_xticks(ind)

    plt.xlabel('Topic')
    plt.ylabel('Probability')
    plt.title('Total Probability of Top 10 Words in each Topic')
    plt.xlim(-0.5,9.5)
    plt.ylim(0,0.15)
    plt.show()

def param_bar_plot(a,b,c,ind,width,ylim,param,xlab,ylab):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    b1 = ax.bar(ind, a, width, color='lightskyblue')
    b2 = ax.bar(ind+width, b, width, color='lightcoral')
    b3 = ax.bar(ind+(2*width), c, width, color='gold')

    ax.set_xticks(ind+width)
    ax.set_xticklabels(range(10))
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.set_ylim(0,ylim)
    ax.legend(handles = [b1,b2,b3],labels=['low '+param,'original model','high '+param])

    plt.tight_layout()
    plt.show()

wiki = gl.SFrame('../../data/people_wiki.gl/')

wiki_docs = gl.text_analytics.count_words(wiki['text'])
wiki_docs = wiki_docs.dict_trim_by_keys(gl.text_analytics.stopwords(), exclude=True)

topic_model = gl.topic_model.create(wiki_docs, num_topics=10, num_iterations=200)

# topics found
topic_model.get_topics()

# 50 words from topic 3
words_3 = topic_model.get_topics(num_words=50)
words_3 = words_3[words_3['topic'] == 2]
words_3['score'].sum()

[x['words'] for x in topic_model.get_topics(output_type='topic_words', num_words=10)]

# Plot probabilities of Top 100 Words in each Topic
plot_probabilities_top_100(topic_model)

# Plot total probability of Top 10 Words in each Topic
plot_probabilities_top_10(topic_model)

# all themes found
themes = ['science and research','team sports','music, TV, and film','American college and politics','general politics', 'art and publishing','Business','international athletics','Great Britain and Australia','international music']

# retrieve Obama article
obama = gl.SArray([wiki_docs[int(np.where(wiki['name']=='Barack Obama')[0])]])

# average 100 probability predictions on Obama's article
print average_predictions(topic_model, themes, obama, 100)

# retrieve Bush article
bush = gl.SArray([wiki_docs[int(np.where(wiki['name']=='George W. Bush')[0])]])

# average 100 probability predictions on Bush's article
print average_predictions(topic_model, themes, bush, 100)

# retrieve Steven Gerrard article
gerrard = gl.SArray([wiki_docs[int(np.where(wiki['name']=='Steven Gerrard')[0])]])

# average 100 probability predictions on Gerrard's article
print average_predictions(topic_model, themes, gerrard, 100)

# preprocessing word count and TF IDF
wiki['lda'] = topic_model.predict(wiki_docs, output_type='probability')
wiki['word_count'] = gl.text_analytics.count_words(wiki['text'])
wiki['tf_idf'] = gl.text_analytics.tf_idf(wiki['word_count'])

# comparing nearest neighbors search using TF IDF and LDA result
model_tf_idf = gl.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                           method='brute_force', distance='cosine')
model_lda_rep = gl.nearest_neighbors.create(wiki, label='name', features=['lda'],
                                            method='brute_force', distance='cosine')

model_tf_idf.query(wiki[wiki['name'] == 'Paul Krugman'], label='name', k=10)

model_lda_rep.query(wiki[wiki['name'] == 'Paul Krugman'], label='name', k=10)

# how close Mariano Rivera is to Alex Rodriguez using TF IDF and LDA
alex_nn = model_tf_idf.query(wiki[wiki['name'] == 'Alex Rodriguez'], label='name', k=5000)
list(alex_nn['reference_label']).index('Mariano Rivera')

alex_nn = model_lda_rep.query(wiki[wiki['name'] == 'Alex Rodriguez'], label='name', k=5000)
list(alex_nn['reference_label']).index('Mariano Rivera')

# loading preprocessed models with high and low alpha
tpm_low_alpha = gl.load_model('../../data/topic_models/lda_low_alpha')
tpm_high_alpha = gl.load_model('../../data/topic_models/lda_high_alpha')

# max probabilities on Obama's article using low, medium and high alpha
a = np.sort(tpm_low_alpha.predict(obama, output_type='probability')[0])[::-1]
b = np.sort(topic_model.predict(obama, output_type='probability')[0])[::-1]
c = np.sort(tpm_high_alpha.predict(obama, output_type='probability')[0])[::-1]
ind = np.arange(len(a))
width = 0.3

param_bar_plot(a,b,c,ind,width,ylim=1.0,param='alpha', xlab='Topics (sorted by weight of top 100 words)',ylab='Topic Probability for Obama Article')


krugman = gl.SArray([wiki_docs[int(np.where(wiki['name']=='Paul Krugman')[0])]])

print average_predictions(tpm_low_alpha, themes, krugman, 100)

print average_predictions(tpm_high_alpha, themes, krugman, 100)

# 11 12
del tpm_low_alpha
del tpm_high_alpha
tpm_low_gamma = gl.load_model('../../data/topic_models/lda_low_gamma')
tpm_high_gamma = gl.load_model('../../data/topic_models/lda_high_gamma')

a_top = np.sort([sum(tpm_low_gamma.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]
b_top = np.sort([sum(topic_model.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]
c_top = np.sort([sum(tpm_high_gamma.get_topics(topic_ids=[i], num_words=100)['score']) for i in range(10)])[::-1]

a_bot = np.sort([sum(tpm_low_gamma.get_topics(topic_ids=[i], num_words=547462)[-1000:]['score']) for i in range(10)])[::-1]
b_bot = np.sort([sum(topic_model.get_topics(topic_ids=[i], num_words=547462)[-1000:]['score']) for i in range(10)])[::-1]
c_bot = np.sort([sum(tpm_high_gamma.get_topics(topic_ids=[i], num_words=547462)[-1000:]['score']) for i in range(10)])[::-1]

ind = np.arange(len(a))
width = 0.3
    
param_bar_plot(a_top, b_top, c_top, ind, width, ylim=0.6, param='gamma',
               xlab='Topics (sorted by weight of top 100 words)', 
               ylab='Total Probability of Top 100 Words')

param_bar_plot(a_bot, b_bot, c_bot, ind, width, ylim=0.0002, param='gamma',
               xlab='Topics (sorted by weight of bottom 1000 words)',
               ylab='Total Probability of Bottom 1000 Words')

topics_low_gamma = tpm_low_gamma.get_topics(cdf_cutoff = 0.5, num_words=10000)

topics_high_gamma = tpm_high_gamma.get_topics(cdf_cutoff = 0.5, num_words=10000)

count_low_gama = topics_low_gamma.groupby(key_columns='topic', operations={'count': agg.COUNT()})

count_high_gama = topics_high_gamma.groupby(key_columns='topic', operations={'count': agg.COUNT()})

count_low_gama['count'].sum() / 10

count_high_gama['count'].sum() / 10
