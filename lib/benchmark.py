import pandas as pd
import numpy as np
import random
from scipy.stats import rankdata

dta = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/lyr_new.csv',index_col = ['track_id'])
word_count = dta.apply(lambda x: x.sum())
word_prop = word_count/float(sum(word_count))
rank_bm = rankdata(word_prop)
# np.savetxt('/Users/pengfeiwang/Desktop/benchmark.csv', rank_bm, delimiter=",")

# train-vali split
seed = 2016
random.seed(seed)
s_index = dta.index.tolist()
random.shuffle(s_index)
train = s_index[235:]
vali = s_index[:235]


train = dta.loc[train]
vali = dta.loc[vali]


rank_vali = [rankdata(i) for i in vali.values]
np.mean([sum(abs(i-rank_bm))/2350 for i in rank_vali])
# 2577.6740117700319
