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
# tune the param
n_clusters = 10

param = [5,10,15]
for k in param:
	codebook = kmeans(bow_train, k)[0]
	new_feature = pd.DataFrame()
	dis = []
	for i in range(dta.shape[0]):
		test = dta['segments_timbre'][i].tolist()
		# train_clusters = vq(test, codebook)[0]
		distance = vq(test, codebook)[1]
		index = [i for i,j in enumerate(train_cluster) if j ==1]
		dis_pred_mean = np.mean([distance[i] for i in index])
		dis.append(dis_pred_mean)
		# new_feature[dta.index[i]] = get_prop(train_clusters)
	m = dis/2350.0
	print m



# after determine the k
# find the class of each song
genre = new_feature.apply(lambda x: np.argmax(x))


lyc = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/lyr_new.csv')
lyc_c = lyc_c.copy()
lyc_c['genre'] = genre

freq = pd.DataFrame()
for i in range(max(genre)+1):
	count = lyc[lyc_c.genre == i].map(lambda x: x.sum())
	freq[i] = count/float(count)

freq.head()




## calculate the percent of each class, then sum(percent * distribution)



