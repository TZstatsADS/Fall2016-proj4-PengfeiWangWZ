import pickle
import pandas as pd
import numpy as np
from utilities import *
from scipy.stats import rankdata
from scipy.cluster.vq import *
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

new_timbre = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/timbre.csv',index_col=['track_id'])
new_pitch = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/pitch.csv',index_col=['track_id'])
stats =  pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/stat.csv',index_col=['track_id'])
x = pd.merge(new_timbre, new_pitch, left_index = True, right_index = True)
x = pd.merge(x, stats, left_index = True, right_index = True)
x[np.isnan(x)]=0

y = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/lsyc_new.csv', index_col=['track_id'])

clf = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(x, y)
pred = clf.predict(x)



# test path_list
dta = ls2df(path_list, selected_feature) 
codebook_timbre = pickle.load(open('/Users/pengfeiwang/Desktop/prj4/Project4_data/timbre.p','rb')) 
new_feature = get_sc(dta, codebook_timbre)


clf = pickle.load(open('/Users/pengfeiwang/Desktop/prj4/Project4_data/clf.p','rb'))
pred = clf.predict(x)

rank_bm = pd.DataFrame(index = range(100))
for i in pred.shape[0]:
	rank_bm.iloc[i] = [4973]*4973 - rankdata(pred.iloc[i])


# pred_df = pd.DataFrame(pred)
# freq = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/freq.csv')

# result = pd.DataFrame()
# submit = pd.DataFrame()
# for j in range(len(path_list)):
# 	for i in range(10):
# 		result[i] = np.dot(pred_df.ix[j,i], freq.iloc[i]).tolist()
# 	submit[path_list[j]] = rankdata(result.sum(axis=1).tolist())

# submit = pd.DataFrame.transpose(submit)



