import sys, re
sys.path.append('/Users/pengfeiwang/Desktop/prj4/MSongsDB/PythonSrc/')
import hdf5_getters

base_path = '/Users/pengfeiwang/Desktop/prj4/Project4_data/data/'

# all the h5 file path
path_list = []
for root, dirs, files in os.walk(base_path):
    for f in files:
        fullpath = os.path.join(root, f)
        if os.path.splitext(fullpath)[1] == '.h5':
            path_list.append(fullpath)


h5 = hdf5_getters.open_h5_file_read('/Users/pengfeiwang/Desktop/prj4/Project4_data/data/A/P/A/TRAPAES128E0786E60.h5')

# get all the attr command 
filter(lambda x: x[:3] == 'get', hdf5_getters.__dict__.keys())
title = hdf5_getters.get_title(h5)



duration = []
for file in path_list:
	file_read = hdf5_getters.open_h5_file_read(file)
	duration.append(hdf5_getters.get_duration(file_read))
