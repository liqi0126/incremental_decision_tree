#!/bin/bash

wget http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip
unzip Activity\ recognition\ exp.zip
mv Activity\ recognition\ exp/Phones_accelerometer.csv .
mv Activity\ recognition\ exp/Phones_gyroscope.csv .
mv Activity\ recognition\ exp/Watch_accelerometer.csv .
mv Activity\ recognition\ exp/Watch_gyroscope.csv .

python convert2csv.py
rm Activity\ recognition\ exp.zip
rm -rf Activity\ recognition\ exp
rm Phones_accelerometer.csv Phones_gyroscope.csv Watch_accelerometer.csv Watch_gyroscope.csv