#!/bin/bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt

python convert2csv.py

rm Skin_NonSkin.txt