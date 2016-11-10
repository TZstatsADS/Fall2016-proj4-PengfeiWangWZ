import scipy.sparse as sparse
import pandas as pd
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords

# print the top words
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print ' '

# clean the stop word dim(_,4826)
def data_clean(data_set):
	fre_stopwords = stopwords.words('french')
	eng_stopwords = stopwords.words('english')
	stop_words = fre_stopwords + eng_stopwords
	col_names_clean = [i for i in data_set.columns if i not in stop_words]
	data_set = data_set.ix[:,col_names_clean]
	sparse_matrix = sparse.csr_matrix(data_set.values)
	return(sparse_matrix)


# load the data
dta = pd.read_csv("/Users/pengfeiwang/Desktop/prj4/Project4_data/lyr.csv", index_col= ['track_id'])
test = pd.
print (dta == 0).sum().sum() / dta.shape[0] / float(dta.shape[1])
#  0 consists 98.9% of whole data points


# set the param
n_topics=10
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100, learning_method='online', learning_offset=50., random_state=2016)

train = data_clean(dta)
lda.fit_transform(train)  # 2350 * n_topics

# get the top words
feature_names = dta.columns
n_top_words = 20
print_top_words(lda, feature_names, n_top_words)






