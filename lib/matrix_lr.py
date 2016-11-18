import pandas as pd
import numpy as np
from utilities import *

dta = pd.read_csv('/Users/pengfeiwang/Desktop/prj4/Project4_data/lyr_new.csv',index_col = ['track_id'])
dta = dta.apply(lambda x: x/float(x.sum()), axis = 1)
y = dta.values # => y (2350, 4973)

# fetch x
base_path = '/Users/pengfeiwang/Desktop/prj4/Project4_data/data/'
path_list =  get_file_path(base_path)
selected_feature = ['get_tatums_confidence', 'get_segments_loudness_max', \
					'get_sections_confidence', 'get_bars_confidence', \
					'get_segments_timbre', 'get_segments_pitches']

x = pd.DataFrame()
dta_x = ls2df(path_list, selected_feature)
for item in selected_feature:
	item = item.split('_',1)[1]
	x[str('_'.join(('%s','mean'))) % item] = [i.mean() for i in dta_x[item]]
	x[str('_'.join(('%s','std'))) % item] = [np.std(i) for i in dta_x[item]]

x = x.values 
x[np.isnan(x)]=0 # => (2350, 12)

# x * b + c = y
# b = np.linalg.lstsq(x, y)[0] # => (12, 4973)
# c = np.linalg.lstsq(x, y)[1] # => (4973, )

