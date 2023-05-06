import csv

with open('/work/output.csv', "w") as outputFile, open('/work/output.csv') as inputFile:
    writer, reader = csv.DictWriter(
        outputFile, fieldnames=['cust_wid', 'label']
    ), csv.DictReader(inputFile)
    writer.writeheader()
    for row in reader:
        writer.writerow({'cust_wid': row['cust_wid'], 'label': row['label']})
