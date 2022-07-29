#import libraries
import pandas as objpandas
import matplotlib.pyplot as objplt
objplt.style.use('ggplot')
import numpy as objnumpy


#Load the data-set
dataset = objpandas.read_csv('/home/mannara/SyntheticData/Output/OutputAdult10.csv', encoding='unicode_escape', low_memory=False)


x_position = ('GAN', 'VAE', 'SMOTE', 'DS', 'SDV-G', 'SDV-GAN', 'SP-NP')
#x_position = ('SMOTE', 'DS', 'SDV-G', 'SP-NP')

y_position = objnumpy.arange(len(x_position))

dataset['e'] = dataset.sum(axis=1)

a = round(abs(dataset.iloc[0 : 1,  8 : 9].values)[0][0], 9);
b = round(abs(dataset.iloc[1 : 2,  8 : 9].values)[0][0], 9);
c = round(abs(dataset.iloc[2 : 3,  8 : 9].values)[0][0], 9);
d = round(abs(dataset.iloc[3 : 4,  8 : 9].values)[0][0], 9);

e = round(abs(dataset.iloc[4 : 5,  8 : 9].values)[0][0], 9);
f = round(abs(dataset.iloc[5 : 6,  8 : 9].values)[0][0], 9);

g = round(abs(dataset.iloc[-1 : ,  8 : 9].values)[0][0], 9);

print(a);
print(b);
print(c);
print(d);

print(e);
print(f);

print(g);


data = [a, b, c, d, e, f, g]

#data = [a, b, c, d]
colors = ['#d096d6', '#85acd4', '#7cd99e', '#bad474', '#cf9661', '#d66a65', '#d96297']
#colors = ['#d096d6', '#7cd99e', '#bad474', '#cf9661', '#d66a65', '#d96297']
#colors = ['#d096d6', '#85acd4', '#7cd99e', '#bad474']
objplt.barh(x_position, data, align='center', alpha=1, color = colors)
objplt.yticks(y_position, x_position)
objplt.xlabel('Proximity Level',fontweight='bold')
#objplt.title('Proximity between the real and the synthetic data for each algorithm')

objplt.savefig('/home/mannara/SyntheticData/Output/OutputAdult.png', dpi=1200)

objplt.show()

#Load the data-set
dataset1 = objpandas.read_csv('/home/mannara/SyntheticData/Output/ROCOutputAdult10.csv', encoding='unicode_escape', low_memory=False)


x_position1 = ('GAN', 'VAE', 'SMOTE', 'DS', 'SDV-G', 'SDV-GAN', 'SP-NP')

y_position1 = objnumpy.arange(len(x_position1))

dataset1['e'] = dataset1.sum(axis=1)

p = round(abs(dataset1.iloc[0 : 1].values)[0][0], 4);
q = round(abs(dataset1.iloc[1 : 2].values)[0][0], 4);
r = round(abs(dataset1.iloc[2 : 3].values)[0][0], 4);
s = round(abs(dataset1.iloc[3 : 4].values)[0][0], 4);
t = round(abs(dataset1.iloc[4 : 5].values)[0][0], 4);
u = round(abs(dataset1.iloc[5 : 6].values)[0][0], 4);

v = round(abs(dataset1.iloc[-1 : ].values)[0][0], 4);

print(p);
print(q);
print(r);
print(s);
print(t);
print(u);

print(v);


data1 = [p, q, r, s, t, u, v]
#data1 = [p, q, r, s, t, u]

colors1 = ['#d096d6', '#85acd4', '#7cd99e', '#bad474', '#cf9661', '#d66a65', '#d96297']
#colors1 = ['#d096d6', '#7cd99e', '#bad474', '#cf9661', '#d66a65', '#d96297']

objplt.barh(x_position1, data1, align='center', alpha=1, color = colors1)
objplt.yticks(y_position1, x_position1)
objplt.xlabel('SD Metrics',fontweight='bold')
#objplt.title('Proximity between the real and the synthetic data for each algorithm')

objplt.savefig('/home/mannara/SyntheticData/Output/ROCOutputAdult.png', dpi=1200)

objplt.show()
