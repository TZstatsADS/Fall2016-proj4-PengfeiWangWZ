import pandas as pd
import ast

with open('/Users/pengfeiwang/Desktop/prj4/Project4_data/mxm_dataset_train.txt','rb') as g:
	dta = g.readlines()

punc = ['#', '%']
dta_normal = [i for i in dta if not i.startswith(tuple(punc))]
values = [i.split(',',2)[2:] for i in dta_normal]
names = [i.split(',',1)[0] for i in dta_normal]
values = [ast.literal_eval(''.join(("{",value[0],"}"))) for value in values] #27
dictionary = dict(zip(name, values))
df = pd.DataFrame.from_dict(dictionary)
df = pd.transpose(df)
df = df.fillna(0)


