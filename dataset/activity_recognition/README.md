# Heterogeneity Activity Recognition Data Set

**[NEW]**: Download using `./download.sh` to make life easier

This dataset is [UCI Machine Learning Repository: Heterogeneity Activity Recognition Data Set](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition).

Download `Activity recognition exp.zip`  from [Index of /ml/machine-learning-databases/00344: Activity recognition exp.zip](http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip) , move `Phones_accelerometer.csv`、 `Phones_gyroscope.csv`、 `Watch_accelerometer.csv`、`Watch_gyroscope.csv`  into this dir and integrate them into a single CSV file as followings:

```bash
python convert2csv.py
```