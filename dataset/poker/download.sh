#!/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data

python convert2csv.py

rm poker-hand-testing.data
rm poker-hand-training-true.data