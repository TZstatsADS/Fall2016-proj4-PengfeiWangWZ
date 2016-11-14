import pickle
import pandas as pd
from utilities import *
from scipy.stats import rankdata
from scipy.cluster.vq import *



dta = ls2df(path_list, selected_feature) # test path_list
codebook = pickle.load(open('','rb')) 
new_feature =  get_sc(dta, codebook)


# load the worc dist of each class
result = pd.DataFrame()
submit = pd.DataFrame()
for j in range(len(path_list)):
	for i in range(10):
		result[i] = new_feature.ix[j,i] * freq.ix[:,i]
	result = result.sum(axis=1).tolist()
	submit[path_list[j]] = rankdata(result)

submit = pd.DataFrame.transpose(submit)