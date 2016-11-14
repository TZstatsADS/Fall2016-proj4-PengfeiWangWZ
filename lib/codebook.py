import sys, os
sys.path.append('/Users/pengfeiwang/Desktop/prj4/Fall2016-proj4-PengfeiWangWZ/lib/')
import numpy as np
from utilities import *
from scipy.cluster.vq import *


# all the h5 file path
base_path = '/Users/pengfeiwang/Desktop/prj4/Project4_data/data/'
path_list =  get_file_path(base_path)


all_features = ["get_bars_confidence", "get_bars_start", "get_beats_confidence", "get_beats_start", \
				"get_sections_confidence","get_sections_start", "get_segments_confidence", \
				"get_segments_loudness_max", "get_segments_loudness_max_time", "get_segments_pitches", \
			    "get_segments_loudness_start", "get_segments_start", "get_segments_timbre", \
			    "get_tatums_confidence", "get_tatums_start"]

selected_feature = ['get_segments_timbre']
dta = ls2df(path_list, selected_feature) # (2350, 1)


bow_train = []
for i in dta['segments_timbre']: 
	bow_train += i.tolist()


# kmean clustering
n_clusters = 10
codebook = kmeans(bow_train, n_clusters)[0]
# codebook = pickle.load(open('','rb')) 
new_feature = pd.DataFrame()
dis = []
for i in range(dta.shape[0]):
	test = dta['segments_timbre'][i].tolist()
	train_clusters = vq(test, codebook)[0]
	# distance = vq(test, codebook)[1]
	# index = [i for i,j in enumerate(train_cluster) if j ==1]
	# dis_pred_mean = np.mean([distance[i] for i in index])
	# dis.append(dis_pred_mean)
	new_feature[dta.index[i]] = get_prop(train_clusters)


# find the class of each song
genre = new_feature.apply(lambda x: np.argmax(x))
# genre.tp_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/cla.csv',index = None)


# dist of each songs
lyc = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/lyr_new.csv',index_col = ['track_id'])
genre = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/cla.csv',header=None)
lyc_c = lyc.copy()
cla = genre.ix[:,0].tolist()
lyc_c['genre'] = cla

freq = pd.DataFrame()
for i in range(max(cla)+1):
	print i
	count = lyc[lyc_c.genre == i].sum(axis=0).tolist()
	freq[i] = [j/float(sum(count)) for j in count]

## calculate the percent of each class, then sum(percent * distribution)



