curl https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip --output beijing_data.zip
unzip beijing_data.zip
mv ./PRSA_Data_20130301-20170228/* .
rm -r PRSA_Data_20130301-20170228
rm beijing_data.zip
