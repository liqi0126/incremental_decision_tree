import csv

headers = ['feat%d' %i for i in range(54)] + ['y']
rows = []

with open('covtype.data', 'r') as f:
    for line in f.readlines():
        rows.append([int(x) for x in line.strip().split(',')])

with open('covtype.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)