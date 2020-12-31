import csv

label = {}

with open('HT_Sensor_metadata.dat', 'r') as f:
    for line in f.readlines()[1:]:
        x = line.strip().split()
        label[x[0]] = x[2]

headers =  ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'Temp.', 'Humidity']
rows = []

with open('HT_Sensor_dataset.dat', 'r') as f:
    for line in f.readlines()[1:]:
        word = line.strip().split()
        row = [float(x) for x in word[2:]]
        row.append(label[word[0]])
        rows.append(row)


with open('gas.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)