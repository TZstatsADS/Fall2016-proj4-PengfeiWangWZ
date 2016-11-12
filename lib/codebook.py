import sys, os
import cv2
sys.path.append('/Users/pengfeiwang/Desktop/prj4/Fall2016-proj4-PengfeiWangWZ/lib/')
from utilities import *
from scipy.cluster.vq import *


# all the h5 file path
base_path = '/Users/pengfeiwang/Desktop/prj4/Project4_data/data/'
path_list =  get_file_path(base_path)


all_features = ["get_bars_confidence", "get_bars_start", "get_beats_confidence", "get_beats_start", "get_sections_confidence", \
			   "get_sections_start", "get_segments_confidence", "get_segments_loudness_max", "get_segments_loudness_max_time", \
			   "get_segments_loudness_start", "get_segments_pitches", "get_segments_start", "get_segments_timbre", \
			    "get_tatums_confidence", "get_tatums_start"]

selected_feature = ['get_segments_timbre']

dta = ls2df(path_list, selected_feature) # (2350, 1)

bow_train = []
for i in dta['segments_timbre']: 
	bow_train += i.tolist()

# kmean clustering
n_clusters = 10
codebook = kmeans(bow_train, k)[0]
train_clusters = kmeans2(dta, k)[1] 
# train_clusters = vq(dta, codebook)[0]


# todo 
## all class = 1 => dist of words



# predict the new obs
data = vq(obs, codebook)[0]

## calculate the percent of each class, then sum(percent * distribution)



