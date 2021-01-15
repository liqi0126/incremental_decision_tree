#!/bin/bash

wget http://archive.ics.uci.edu/ml/machine-learning-databases/00362/HT_Sensor_UCIsubmission.zip
unzip HT_Sensor_UCIsubmission.zip
unzip HT_Sensor_dataset.zip

python convert2csv.py

rm HT_Sensor_UCIsubmission.zip HT_Sensor_dataset.zip HT_Sensor_dataset.dat HT_Sensor_metadata.dat
rm -rf __MACOSX
