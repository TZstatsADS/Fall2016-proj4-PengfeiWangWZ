import pandas as pd
import numpy as np
from scipy.stats import rankdata

dta = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/lyr_new.csv',index_col = ['track_id'])
word_count = dta.apply(lambda x: x.sum())
word_prop = word_count/float(sum(word_count))
rank = rankdata(word_prop)
np.savetxt('/Users/pengfeiwang/Desktop/benchmark.csv', rank, delimiter=",")
