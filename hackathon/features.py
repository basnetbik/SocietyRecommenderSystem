"""
Find out the complete features of the countries
"""

import os, csv
feature_dict = {}

def extract_full_features(country_name):
    feature_dict[country_name[:-4]] = []
    with open('NIC/%s'%country_name,'rb') as infile:
        csvreader = csv.DictReader(infile)
        for rows in csvreader:
            # non_blank = sum([int(i!='') for i in rows.values()])
            non_blank = sum([int(rows[i] != '') for i in rows if len(i)==4 and int(i) >= 1970])
            if non_blank > 23:
                feature_dict[country_name[:-4]].append(rows)


for countries in os.listdir('NIC/'):
    extract_full_features(countries)

#Find out the common indicators
count = 0
reference_country = 'Brazil'
features_list = []
for features in feature_dict[reference_country]:
    for othercountry in feature_dict:
        if othercountry == reference_country:
            continue
        for otherfeatures in feature_dict[othercountry]:
            if otherfeatures['Indicator Name'] == features['Indicator Name']:
                break
        else:
            break
    else:
        #This feature is available
        count += 1
        print features['Indicator Name']
        features_list.append(features['Indicator Name'])

import pickle
with open("Indicators.txt","wb") as outfile:
    pickle.dump(features_list, outfile)
# c = 0
# with open('Countries/Nepal.csv','r') as nepalfile:
#     csvreader = csv.DictReader(nepalfile)
#     for rows in csvreader:
#         if rows['Indicator Name'] in features_list:
#             if rows['2011']:
#                 c += 1
# print c

for countries in feature_dict:
    feature_data_list = []
    for features in feature_dict[countries]:
        if features['Indicator Name'] in features_list:
            feature_data_list.append(features)
    feature_data_list = sorted(feature_data_list, key=lambda x: x['Indicator Name'])
    with open("NIC_ordered/%s.csv"%countries,"wb") as outfile:
        key = sorted(feature_data_list[0].keys())
        # key.remove('2012')
        key.remove('2013')
        csvwriter = csv.DictWriter(outfile, key)
        outfile.write(','.join(key)+'\n')
        for rows in feature_data_list:
            del rows['2013']
            # del rows['2012']
            csvwriter.writerow(rows)


print count


