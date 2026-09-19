import csv

with open("data/shape_features.csv", newline="") as infile, open("data/labeled.csv", "w", newline="") as outfile:
    writer = csv.writer(outfile)
    for row in csv.reader(infile):
        if row and row[-1].strip():  # last column non-empty
            writer.writerow(row)