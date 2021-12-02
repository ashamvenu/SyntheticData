#import libraries
import pandas as objpandas
import matplotlib.pyplot as objplt
objplt.style.use('ggplot')
import numpy as objnumpy


#Load the data-set
dataset = objpandas.read_csv('F:/MS-CS/Thesis/Dataset/Output10.csv', encoding='unicode_escape', low_memory=False) 


x_position = ('GAN', 'VAE', 'SMOTE')

y_position = objnumpy.arange(len(x_position))

dataset['e'] = dataset.sum(axis=1)

a = round(abs(dataset.iloc[0 : 1,  8 : 9].values)[0][0], 2);
b = round(abs(dataset.iloc[1 : 2,  8 : 9].values)[0][0], 2);
c = round(abs(dataset.iloc[2 : 3,  8 : 9].values)[0][0], 2);

print(a);
print(b);
print(c);

data = [a, b, c]

colors = ['#970da6', '#1673b5', '#690eab']

objplt.bar(y_position, data, align='center', alpha=1, color = colors)
objplt.xticks(y_position, x_position,fontweight='bold')
objplt.ylabel('Proximity Level',fontweight='bold')
#objplt.title('Proximity between the real and the synthetic data for each algorithm')

objplt.savefig('F:/MS-CS/Thesis/Dataset/Output10.png', dpi=1200)

objplt.show()