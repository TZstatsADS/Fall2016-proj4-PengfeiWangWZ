import sys, os, pickle
sys.path.append('/Users/pengfeiwang/Desktop/prj4/Fall2016-proj4-PengfeiWangWZ/lib/')
import numpy as np
import pandas as pd
from utilities import *
from scipy.cluster.vq import *


# get h5 file path
base_path = '/Users/pengfeiwang/Desktop/prj4/Project4_data/data/'
path_list =  get_file_path(base_path)

all_features = ["get_bars_confidence", "get_bars_start", "get_beats_confidence", "get_beats_start", \
				"get_sections_confidence", "get_sections_start", "get_segments_confidence", \
				"get_segments_loudness_max", "get_segments_loudness_max_time", "get_segments_pitches", \
			    "get_segments_loudness_start", "get_segments_start", "get_segments_timbre", \
			    "get_tatums_confidence", "get_tatums_start"]

selected_feature = ['get_segments_timbre', 'get_segments_pitches']
stat = ["get_bars_confidence", "get_beats_confidence", "get_segments_loudness_max"]
dta = ls2df(path_list, selected_feature)


bow_train = []
for i in dta['segments_timbre']: 
	bow_train += i.tolist()


# kmean clustering
n_clusters = 50
codebook = kmeans(bow_train, n_clusters)[0]
codebook = pickle.dump(codebook, open('/Users/pengfeiwang/Desktop/prj4/Project4_data/timbre.p','rb')) 


codebook_timbre = pickle.load(open('/Users/pengfeiwang/Desktop/prj4/Project4_data/timbre.p','rb')) 
new_timbre = pd.DataFrame()
dis = []
for i in range(dta.shape[0]):
	test = dta['segments_timbre'][i].tolist()
	train_clusters = vq(test, codebook_timbre)[0]
	# distance = vq(test, codebook_timbre)[1]
	# index = [i for i,j in enumerate(train_cluster) if j ==1]
	# dis_pred_mean = np.mean([distance[i] for i in index])
	# dis.append(dis_pred_mean)
	new_timbre[dta.index[i]] = get_prop(train_clusters)

new_timbre = new_timbre.transpose()
new_timbre.to_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/timbre.csv')


codebook_pitch = pickle.load(open('/Users/pengfeiwang/Desktop/prj4/Project4_data/pitch.p','rb'))
new_pitch = pd.DataFrame()
dis = []
for i in range(dta.shape[0]):
	test = dta['segments_pitches'][i].tolist()
	train_clusters = vq(test, codebook_pitch)[0]
	new_pitch[dta.index[i]] = get_prop(train_clusters)

new_pitch = new_pitch.transpose()
new_pitch.to_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/pitch.csv')





