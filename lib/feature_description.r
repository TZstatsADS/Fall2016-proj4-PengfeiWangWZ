library(rhdf5)
h5ls("Desktop/prj4/Project4_data/data/A/A/A/TRAAARJ128F9320760.h5")

# three major sectors 
analysis <- h5read("Desktop/prj4/Project4_data/data/A/A/A/TRAAARJ128F9320760.h5", "/analysis")
metadata <- h5read("Desktop/prj4/Project4_data/data/A/A/A/TRAAARJ128F9320760.h5", "/metadata")
musicbrainz <- h5read("Desktop/prj4/Project4_data/data/A/A/A/TRAAARJ128F9320760.h5", "/musicbrainz")

# show the features in each sectors
names(analysis)
names(metadata)
names(musicbrainz)

# get the feature
segments_start <- h5read("Desktop/prj4/Project4_data/data/A/A/A/TRAAARJ128F9320760.h5", "/analysis/segments_start")

