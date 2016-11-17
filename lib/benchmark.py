import pandas as pd
import numpy as np
import random
from scipy.stats import rankdata

def calu_result(pred_df, true_df):
	new_df = pd.concat([pred_df,true_df])
	# pd.concat([dist,vali.iloc[[0]]])
	mean_rank = new_df.iloc[[0]].loc[:, new_df.iloc[1] != 0].sum(axis=1)[0]/sum(new_df.iloc[1] != 0)
	return(mean_rank)


dta = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/lyr_new.csv',index_col = ['track_id'])
word_count = dta.apply(lambda x: x.sum())
word_prop = word_count/float(sum(word_count))

rank_bm = [4973]*4973 - rankdata(word_prop)
a = pd.DataFrame(rank_bm).transpose()
a.columns = dta.columns
calu_result(a, dta.iloc[[0]])

# np.savetxt('/Users/pengfeiwang/Desktop/benchmark.csv', rank_bm, delimiter=",")


# train-vali split
seed = 2016
random.seed(seed)
s_index = dta.index.tolist()
random.shuffle(s_index)
train_index = s_index[235:]
vali_index = s_index[:235]


train = dta.loc[train_index]
vali = dta.loc[vali_index]

dist = rankdata(train.sum())
dist = pd.DataFrame(dist)
index_name = dist.columns.copy()
dist = dist.transpose()
dist.columns = train.columns

a = []
for i in range(vali.shape[0]):
	b = calu_result(dist, vali.iloc[[i]])
	# print b
	a.append(b)

np.mean(a)/((4974)/2.)
# 0.26078588698759286
