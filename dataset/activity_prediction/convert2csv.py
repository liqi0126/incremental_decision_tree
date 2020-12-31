import pandas as pd

f = open("WISDM_ar_v1.1_raw.txt")
dataset = []
name = ['x-acceleration', 'y-accel','z-accel','activity']
line = f.readline()
while line: 
    attr = line.split(',')
    if len(attr) != 6:
        line = f.readline()
        continue
    data_id = attr[1]
    attr = attr[3:]
    attr[-1] = attr[-1][:-2]
    attr.append(data_id)
    dataset.append(attr)
    line = f.readline()
f.close()
df = pd.DataFrame(columns=name, data=dataset)
df.to_csv('activity_prediction.csv',index=None)