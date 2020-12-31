import csv

headers = ['feat%d' %i for i in range(11)] + ['y']
rows = []

with open('poker-hand-training-true.data', 'r') as f:
    for line in f.readlines():
        rows.append([int(x) for x in line.strip().split(',')])

with open('poker-hand-testing.data', 'r') as f:
    for line in f.readlines():
        rows.append([int(x) for x in line.strip().split(',')])

with open('poker.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)