import csv

filename = "data/imdb/classLabelsimdb.csv"

with open(filename, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    ans = [row for row in reader]
    print("No. of classes =", len(ans))
    print("Header =", ans[0].keys())