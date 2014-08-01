"""
Parse the world bank data country-wise
"""

import csv

def dump_csv(data):
    with open('Countries/%s.csv'%(data[0]['Country Name']),'wb') as outfile:
        csvwriter = csv.DictWriter(outfile, data[0].keys())
        outfile.write(','.join(data[0].keys())+'\n')
        for i in data:
            csvwriter.writerow(i)

with open('Data/WDI_Data.csv','rb') as infile:
    csvreader = csv.DictReader(infile)
    current_data = []
    for i, rows in enumerate(csvreader):
        # print i
        # print rows
        # print rows.keys()
        # print rows.values()
        if i % 1300 == 0 and i!=0:
            dump_csv(current_data)
            current_data = []
            pass
        current_data.append(rows)
    dump_csv(current_data)