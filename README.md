# Project: A_W_K_Word!!😱

### [Project Description](doc/Project4_desc.md)

![image](http://cdn.newsapi.com.au/image/v1/f7131c018870330120dbe4b73bb7695c?width=650)

Term: Fall 2016

+ [Data link](https://courseworks2.columbia.edu/courses/11849/files/folder/Project_Files?preview=763391)-(**courseworks login required**)
+ [Data description](doc/readme.html)
+ Contributor's name: Pengfei Wang
+ Projec title: Words 4 Music
+ Project summary: 
	+ Extracted segments_timbre, segments_pitches features and used K-means to do dimension reduction
	+ Mapped features with codebook and re-constructed the features according to majority vote
    + Calculated the statistics of bars_confidence, segments_loudness_max and beats_confidence features, including mean and standard deviation
    + Conducted multiple target regression with XGB model and ranked the result of each lyrics
    + Constructed Dictionary and Corpus for Latent Dirichlet Allocation Model (LDA) basing on bag-of-words 

	
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
