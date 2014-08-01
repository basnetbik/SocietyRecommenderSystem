"""
Simple demo of a horizontal bar chart.
"""
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import csv

# Example data
Indicators = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
score = 3 + 10 * np.random.rand(len(Indicators))

Indicators = None
score = None
with open('results0.001.csv','rb') as infile:
    csvreader = csv.DictReader(infile)
    rows = csvreader.next()
    del rows['Lambda']
    # Indicators, score = rows.keys(), np.array(rows.values(),dtype=np.float)
    stat = sorted(zip(rows.values(), rows.keys()), reverse=True)
    score, Indicators = zip(*stat)
    Indicators, score = Indicators[:15], score[:15]
    Indicators, score = list(reversed(Indicators)), list(reversed(score))
    # Indicators, score = zip(*zip(Indicators, score))
    score = np.array(score, dtype=np.float)


y_pos = np.arange(len(Indicators))

# error = np.random.rand(len(people))
plt.barh(y_pos, score, height= 0.1, align='center')
plt.yticks(y_pos, Indicators)
plt.xlabel('Score')
plt.title('Comparative analysis of Indicators')

plt.show()