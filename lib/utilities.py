import sys
import re
import os
sys.path.append('/Users/pengfeiwang/Desktop/prj4/MSongsDB/PythonSrc/')
import hdf5_getters
import pandas as pd

# all the h5 file path


def get_file_path(base_path):
    path_list = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            fullpath = os.path.join(root, f)
            if os.path.splitext(fullpath)[1] == '.h5':
                path_list.append(fullpath)
    return(path_list)


def get_list_attr(path_list, attr):
    attr_list = []
    i = 1
    for file in path_list:
        try:
            file_read = hdf5_getters.open_h5_file_read(file)
            attr_list.append(hdf5_getters.__getattribute__(attr)(file_read))
            file_read.close()
            print 'Finished ' + str(i) + '/2350'
            i += 1
        except:
            print '---- Failed to get ' + file + ' ---- No:' + str(i)
            attr_list.append(0)
            i += 1
    return(attr_list)


# construct the df
def ls2df(path_list, feature_selected):
    df = pd.DataFrame()
    # df.columns = feature_selected
    for feature in feature_selected:
        name = feature.split('_', 1)[1]
        df[name] = get_list_attr(path_list, feature)

    df.index = [re.split('[/.]', i)[-2] for i in path_list]
    return(df)


# count the proportion of result
def get_prop(alist):
    alist = list(alist)
    cls = range(10)
    prop = [alist.count(i) / float(len(alist)) for i in cls]
    return(dict(zip(cls, prop)).values())


# soft classification
def get_sc(dta, codebook):
    dis = []
    new_feature = pd.DataFrame()
    for i in range(dta.shape[0]):
        test = dta['segments_timbre'][i].tolist()
        train_clusters = vq(test, codebook)[0]
        new_feature[dta.index[i]] = get_prop(train_clusters)
    return(new_feature)
