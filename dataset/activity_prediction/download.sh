#!/bin/bash

wget https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
tar -xvzf WISDM_ar_latest.tar.gz
mv WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt .
python convert2csv.py
rm -rf WISDM_ar_v1.1
rm WISDM_ar_latest.tar.gz
rm WISDM_ar_v1.1_raw.txt