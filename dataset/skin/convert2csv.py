import csv

headers = ['B', 'G', 'R', 'y']
rows = []

with open('Skin_NonSkin.txt', 'r') as f:
    for line in f.readlines():
        rows.append([int(x) for x in line.strip().split()])

with open('skin.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)