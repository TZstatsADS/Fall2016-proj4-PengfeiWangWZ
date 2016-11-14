from gensim import corpora, models, similarities
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import pyLDAvis.gensim as gensimvis
import pyLDAvis

dta = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/lyr_new.csv',index_col = ['track_id'])
tfidf =TfidfTransformer(norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
data =tfidf.fit_transform(dta.values)
data = pd.DataFrame(data.todense())
corpus = data.values()

lda = models.ldamodel.LdaModel(corpus, num_topics=10)
gensimvis.prepare(lda, corpus,)