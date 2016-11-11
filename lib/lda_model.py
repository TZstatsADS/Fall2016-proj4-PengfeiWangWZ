import scipy.sparse as sparse
from scipy.stats import rankdata
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
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
	return(sparse_matrix, col_names_clean)


# load the data
dta = pd.read_csv("/Users/pengfeiwang/Desktop/prj4/Project4_data/lyr.csv", index_col= ['track_id'])
test = pd.
print (dta == 0).sum().sum() / dta.shape[0] / float(dta.shape[1])
#  0 consists 98.9% of whole data points


# set the param
n_topics = 500
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100, learning_method='online', learning_offset=50., random_state=2016)

# dimention reduction via lda
train, feature_names = data_clean(dta)
new_train = lda.fit_transform(train)  # 2350 * n_topics

# get the top words
n_top_words = 20
print_top_words(lda, feature_names, n_top_words)

# fuzz c-means clustering
n_centers = 10
fuzziness_degree = 2
error = 0.005
maxiter = 1000
centers, predict_prob, u0, d, jm, n_iters, fpc = fuzz.cluster.cmeans(new_train, c=n_centers, m=fuzziness_degree, error=error, maxiter=maxiter, init=None)
# cluster_result = np.argmax(predict_prob, axis=0)

prob = zip(*[i for i in predict_prob])
prob = [list(i) for i in prob]

