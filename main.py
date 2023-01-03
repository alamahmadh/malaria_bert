import os
import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

date = '08252022'
task = 'ngram12_pudmedbertfulltext'
result_dir = 'results/{}_{}/'.format(date,task)
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
print('Date: {}; Task: {}'.format(date,task))

print('##################################################################')

print('load data...')
articles =pd.read_csv('malaria_trial1.csv')
#articles["publication_date"] = articles["publication_date"].apply(pd.to_datetime)
#articles = articles.sort_values(by='publication_date')
#articles = articles[(articles['publication_date']>'2016-12-31') & (articles['publication_date']<'2022-12-31')]

docs = articles['abstract'].to_list()
titles = articles['title'].to_list()
print('there are {} clean articles from 2017-2022'.format(len(articles)))

# Prepare embeddings
print('##################################################################')
print('perform embeddings...')
#sentence_model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
#embeddings = sentence_model.encode(docs)
embeddings = np.load('embeddings_pubmedbertfulltext_ver1.npy')

# Train our topic model using our pre-trained sentence-transformers embeddings
print('##################################################################')
print('train our topic model using our pre-trained embeddings...')
umap_model = UMAP(n_neighbors=50, n_components=5, metric='cosine', random_state=42)
stop_words = text.ENGLISH_STOP_WORDS.union(["malaria", "der", "und", "die", "von", "la", "les", "et", "des", "los", "en", "el", "se",

                                            "las", "que", "para", "le", "dans", "du"])
vectorizer_model = CountVectorizer(
    min_df=0.01,
    ngram_range=(1, 2),
    stop_words=stop_words
)

hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    min_samples=1
)

pca_model = PCA(n_components=50)
cluster_model = KMeans(n_clusters=20)

topic_model = BERTopic(
    #min_topic_size=20,
    top_n_words=15,
    hdbscan_model=hdbscan_model,
    nr_topics="auto",
    umap_model=umap_model,
    #embedding_model=sentence_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(
    docs,
    embeddings
)

print('##################################################################')

print('Number of topics: {}'.format(len(topic_model.get_topic_info())))
outliers = topic_model.get_topic_info().query('Topic==-1')['Count'].values
print('Number of docs belong to outliers: {}'.format(outliers))

#Save 3 most representative docs from each topic
l_topics = np.asarray([topic for topic in topics if topic != -1])
l_topics = np.unique(l_topics)

labels = topic_model.get_topic_info()['Name'][1:].values
rep_docs = pd.DataFrame(columns=['Topic', 'Label', 'Abstract1', 'Abstract2', 'Abstract3'])

for i in range(len(l_topics)):
    rep_docs.loc[i, 'Topic'] = i
    rep_docs.loc[i, 'Label'] = labels[i]
    rep_docs.loc[i, 'Abstract1'] = topic_model.get_representative_docs(i)[0]
    rep_docs.loc[i, 'Abstract2'] = topic_model.get_representative_docs(i)[1]
    rep_docs.loc[i, 'Abstract3'] = topic_model.get_representative_docs(i)[2]

rep_docs.to_csv(result_dir + 'docs_{}_{}.csv'.format(date,task), index=False)

#hierarchical model
hierarchical_topics = topic_model.hierarchical_topics(docs, topics)
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(result_dir + "{}_{}.html".format(date,task))

#Coherence score
# Preprocess Documents
documents = pd.DataFrame({"Document": docs,
                          "ID": range(len(docs)),
                          "Topic": topics})
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})

# Extract vectorizer and analyzer from BERTopic
vectorizer = topic_model.vectorizer_model
analyzer = vectorizer.build_analyzer()

# Extract features for Topic Coherence evaluation
words = vectorizer.get_feature_names()
tokens = [analyzer(doc) for doc in cleaned_docs]
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]
topic_words = [[words for words, _ in topic_model.get_topic(topic)]
               for topic in range(len(set(topics))-1)]

# Evaluate
coherence_model = CoherenceModel(topics=topic_words,
                                 texts=tokens,
                                 corpus=corpus,
                                 dictionary=dictionary,
                                 coherence='c_npmi')
coherence = coherence_model.get_coherence()
print('##################################################################')
print('coherence score: {}'.format(coherence))

#Sillhoutte score
# Generate `X` and `labels` only for non-outlier topics (as they are technically not clusters)
umap_embeddings = topic_model.umap_model.transform(embeddings)
indices = [index for index, topic in enumerate(topics) if topic != -1]
X = umap_embeddings[np.array(indices)]
labels = [topic for index, topic in enumerate(topics) if topic != -1]

# Calculate silhouette score
scoring = silhouette_score(X, labels)
print('sillhoutte score: {}'.format(scoring))

#save results
print('##################################################################')
print('save the results...')

#topic_model.save(result_dir + '{}_{}.bt'.format(date,task))

topic_model.get_topic_info().to_csv(result_dir + 'topic_info_{}_{}.csv'.format(date,task), index=False)
result = pd.DataFrame(columns=['pubmed_id'])
result['pubmed_id'] = articles['pubmed_id'].values
for i in range(len(probs[0])):
    result['prob_{}'.format(i-1)] = probs[:,i]

result.to_csv(result_dir + 'topic_probs_{}_{}.csv'.format(date,task))

Df = pd.DataFrame(columns=['words_outliers', 'probs_outliers'])
Df['words_outliers'] = np.asarray(topic_model.get_topic(-1)).T[0]
Df['probs_outliers'] = np.asarray(topic_model.get_topic(-1)).T[1]

for i in topic_model.get_topic_info()["Topic"][1:].values:
    Df['words_topic_{}'.format(i)] = np.asarray(topic_model.get_topic(i)).T[0]
    Df['probs_topic_{}'.format(i)] = np.asarray(topic_model.get_topic(i)).T[1]

Df.to_csv(result_dir + 'ctfidf_{}_{}.csv'.format(date,task), index=False)

print('############################################################################################################')
print('############################################################################################################')
print('MERGING TOPICS...')

#biobert
#topics_to_merge = [[14,8,5,6,15,13], [17,12], [10,22,11,25], [21,16], [18,23], [24,3,2], [4,9], [1,7,20]]
#pubmedbert
#topics_to_merge = [[17,18],[8,15],[9,22],[11,2,7,10]]
#pubmedbert
topics_to_merge = [[6,17],[21,3,9],[2,26],[10,15],[1,12,14]]
#merging topics
topic_model.merge_topics(docs, topics, topics_to_merge)
new_topics1 = topic_model._map_predictions(topic_model.hdbscan_model.labels_)
new_probs = hdbscan.all_points_membership_vectors(topic_model.hdbscan_model)
new_probs = topic_model._map_probabilities(new_probs, original_topics=True)

probability_threshold = 0.01
new_topics2 = [np.argmax(prob) if max(prob) >= probability_threshold else -1 for prob in new_probs]

#Coherence score
# Preprocess Documents
documents = pd.DataFrame({"Document": docs,
                          "Title": titles,
                          "ID": range(len(docs)),
                          "Topic1": new_topics1,
                          "Topic2": new_topics2})

documents_per_topic1 = documents.groupby(['Topic1'], as_index=False).agg({'Document': ' '.join})
documents_per_topic2 = documents.groupby(['Topic2'], as_index=False).agg({'Document': ' '.join})

documents.to_csv(result_dir + 'freqpertopic_{}_{}.csv'.format(date,task))

cleaned_docs1 = topic_model._preprocess_text(documents_per_topic1.Document.values)
cleaned_docs2 = topic_model._preprocess_text(documents_per_topic2.Document.values)

# Extract vectorizer and analyzer from BERTopic
vectorizer = topic_model.vectorizer_model
analyzer = vectorizer.build_analyzer()

# Extract features for Topic Coherence evaluation
words = vectorizer.get_feature_names()
tokens1 = [analyzer(doc) for doc in cleaned_docs1]
tokens2 = [analyzer(doc) for doc in cleaned_docs2]
dictionary1 = corpora.Dictionary(tokens1)
dictionary2 = corpora.Dictionary(tokens2)
corpus1 = [dictionary1.doc2bow(token) for token in tokens1]
corpus2 = [dictionary2.doc2bow(token) for token in tokens2]
topic_words1 = [[words for words, _ in topic_model.get_topic(topic)]
               for topic in range(len(set(new_topics1))-1)]
topic_words2 = [[words for words, _ in topic_model.get_topic(topic)]
               for topic in range(len(set(new_topics2))-1)]
# Evaluate
coherence_model1 = CoherenceModel(topics=topic_words1,
                                 texts=tokens1,
                                 corpus=corpus1,
                                 dictionary=dictionary1,
                                 coherence='c_npmi')

coherence_model2 = CoherenceModel(topics=topic_words2,
                                 texts=tokens2,
                                 corpus=corpus2,
                                 dictionary=dictionary2,
                                 coherence='c_npmi')

coherence1 = coherence_model1.get_coherence()
coherence2 = coherence_model2.get_coherence()
print('##################################################################')
print('coherence score from merged topics: {}'.format(coherence1))
print('coherence score from merged topics no outliers: {}'.format(coherence2))

#Sillhoutte score
# Generate `X` and `labels` only for non-outlier topics (as they are technically not clusters)
umap_embeddings = topic_model.umap_model.transform(embeddings)
indices1 = [index for index, topic in enumerate(new_topics1) if topic != -1]
X1 = umap_embeddings[np.array(indices1)]
labels1 = [topic for index, topic in enumerate(new_topics1) if topic != -1]

# Calculate silhouette score
scoring1 = silhouette_score(X1, labels1)

#merged topic no outliers
indices2 = [index for index, topic in enumerate(new_topics2) if topic != -1]
X2 = umap_embeddings[np.array(indices2)]
labels2 = [topic for index, topic in enumerate(new_topics2) if topic != -1]

# Calculate silhouette score
scoring1 = silhouette_score(X1, labels1)

#merged topic no outliers
indices2 = [index for index, topic in enumerate(new_topics2) if topic != -1]
X2 = umap_embeddings[np.array(indices2)]
labels2 = [topic for index, topic in enumerate(new_topics2) if topic != -1]

# Calculate silhouette score
scoring2 = silhouette_score(X2, labels2)

print('sillhoutte score from merged topics: {}'.format(scoring1))
print('sillhoutte score from merged topics no outliers: {}'.format(scoring2))

topic_model.get_topic_info().to_csv(result_dir + 'merged_topic_info_{}_{}.csv'.format(date,task), index=False)
result = pd.DataFrame(columns=['pubmed_id'])
result['pubmed_id'] = articles['pubmed_id'].values
for i in range(len(new_probs[0])):
    result['prob_{}'.format(i-1)] = new_probs[:,i]

result.to_csv(result_dir + 'merged_topic_probs_{}_{}.csv'.format(date,task))

Df0 = pd.DataFrame(columns=['words_outliers', 'probs_outliers'])
Df0['words_outliers'] = np.asarray(topic_model.get_topic(-1)).T[0]
Df0['probs_outliers'] = np.asarray(topic_model.get_topic(-1)).T[1]

for i in topic_model.get_topic_info()["Topic"][1:].values:
    Df0['words_topic_{}'.format(i)] = np.asarray(topic_model.get_topic(i)).T[0]
    Df0['probs_topic_{}'.format(i)] = np.asarray(topic_model.get_topic(i)).T[1]

Df0.to_csv(result_dir + 'merged_ctfidf_{}_{}.csv'.format(date,task), index=False)