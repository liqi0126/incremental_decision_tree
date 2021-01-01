#!/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info
gzip -d covtype.data.gz

python convert2csv.py

rm covtype.data covtype.info