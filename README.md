# Project: A_W_K_Word!!😱

### [Project Description](doc/Project4_desc.md)

![image](http://cdn.newsapi.com.au/image/v1/f7131c018870330120dbe4b73bb7695c?width=650)

Term: Fall 2016

+ [Data link](https://courseworks2.columbia.edu/courses/11849/files/folder/Project_Files?preview=763391)-(**courseworks login required**)
+ [Data description](doc/readme.html)
+ Contributor's name: Pengfei Wang
+ Projec title: Words 4 Music
+ Project summary: 
	+ Used the overall lyrics' frequencies as benchmark [(Click here!)](https://github.com/TZstatsADS/Fall2016-proj4-PengfeiWangWZ/blob/master/lib/benchmark.py)
	+ Extracted segments_timbre, segments_pitches features. Used K-means to mapped features with codebook and re-constructed the features according to majority vote [(Click here!)](https://github.com/TZstatsADS/Fall2016-proj4-PengfeiWangWZ/blob/master/lib/codebook.py)
	+ Calculated the statistics of bars_confidence, segments_loudness_max and beats_confidence features, including mean and standard deviation [(Click here!)](https://github.com/TZstatsADS/Fall2016-proj4-PengfeiWangWZ/blob/master/lib/matrix_lr.py)
	+ Conducted multiple target regression with XGB model and ranked the result of each lyrics [(Click here!)](https://github.com/TZstatsADS/Fall2016-proj4-PengfeiWangWZ/blob/master/lib/model.py)
	+ Also tried to do topic model on the lyrics to reduce the dimension of y, both with the stopwords cleaned or not. Here is the process to constructed Dictionary and Corpus for Latent Dirichlet Allocation Model (LDA) basing on bag-of-words [(Click here!)](http://nbviewer.jupyter.org/github/TZstatsADS/Fall2016-proj4-PengfeiWangWZ/blob/master/lib/lda.ipynb). And here is a prototype of the cleaned stopwords version [(Click here!)](https://github.com/TZstatsADS/Fall2016-proj4-PengfeiWangWZ/blob/master/lib/try_lda_stopwords.py). However, both of them did no good basing on the features we built
	
	+ [utilities.py](https://github.com/TZstatsADS/Fall2016-proj4-PengfeiWangWZ/blob/master/lib/utilities.py) has almost all the function we need and [hdf5_getters.py](https://github.com/TZstatsADS/Fall2016-proj4-PengfeiWangWZ/blob/master/lib/hdf5_getters.py) is used to retrive all the features we need in h5 format, which was written by [tbertinmahieux](https://github.com/tbertinmahieux/MSongsDB)

	
Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
