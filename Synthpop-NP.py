from collections import defaultdict
import csv
import json
import numpy as objnumpy
import numpy as np
import pandas as objpandas
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from synthpop import Synthpop
import os, multiprocessing
os.environ["OMP_NUM_THREADS"] = "100"
from sdv.evaluation import evaluate
import threading
from multiprocessing.pool import ThreadPool as Pool
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import warnings
from pandas.plotting import scatter_matrix
import statistics

class PandasLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_dict = defaultdict(list)

    def fit(self, X):
        X = X.astype('category')
        cols = X.columns
        values = list(map(lambda col: X[col].cat.categories, cols))
        self.label_dict = dict(zip(cols, values))
        # return as category for xgboost or lightgbm 
        return self

    def transform(self, X):
        # check missing columns
        missing_col = set(X.columns)-set(self.label_dict.keys())
        if missing_col:
            raise ValueError('the column named {} is not in the label dictionary. Check your fitting data.'.format(missing_col)) 
        return X.apply(lambda x: x.astype('category').cat.set_categories(self.label_dict[x.name]).cat.codes.astype('category').cat.set_categories(objnumpy.arange(len(self.label_dict[x.name]))))

    def inverseTransform(self, X):
        return X.apply(lambda x: objpandas.Categorical.from_codes(codes=x.values,
                       categories=self.label_dict[x.name]))

startTime = time.time();

dataset = objpandas.read_csv('/Input/adult10k.csv', encoding='unicode_escape', low_memory=False)

#Print the count of rows and coulmns in csv file
print("Dimensions of Dataset: {}".format(dataset.shape))

# Dropped all the Null, Empty, NA values from csv file 
new_dataset = dataset.dropna(axis=0, how='any') 

print("Dimensions of Dataset after Pre-processing : {}".format(new_dataset.shape))

#Encoding data other than int, float and double
#le = preprocessing.LabelEncoder()
#enc = PandasLabelEncoder()
enc  = {}

#Get Other Datatype Columns Only
n_columns = []
n_transform = []
n_name = []
count = 0;

#Function to convert string, objects to numeric ones
def numericsConverter(columnName):
  
    #print("Argumnet: ", columnName) 
  
    #input data
    dff = objpandas.DataFrame({columnName: new_dataset[columnName]})

    enc[count] = PandasLabelEncoder()

    input = enc[count].fit_transform(dff);

    #print("Input ", input)
  
    return input;

#Method to find datatype of columns in csv file
for name, dtype in new_dataset.dtypes.iteritems():
    
    print("Data Type of Coulmn " + name + " is : ", dtype)
    
    if(dtype == 'object'):
        
        #print("Transform Coulmn " + name)	

        input = numericsConverter(name);
        
        new_dataset[name] = input;
        
        n_columns.append(objnumpy.array(count))
        n_name.append(objnumpy.array(name))
        n_transform.append(objnumpy.array(input))
        
    count += 1;

n_columns = objnumpy.array(n_columns);
n_name = objnumpy.array(n_name);
n_transform = objnumpy.array(n_transform);

#Get numeric Values Only
#new_dataset = new_dataset._get_numeric_data()
#new_dataset = new_dataset.iloc[ : , 0 : 10000]

#New dataset after encoding
#new_dataset = new_dataset.apply(le.fit_transform)

#Get the coulmn header
columnHeaders = new_dataset.head(0)

#Get Feature Coulmns Only
n_features = len(new_dataset.columns)
    
print("Features: ", n_features)

B = range(0, n_features);
  
new_dataset.columns = list(B)

print("new_dataset.columns: ", new_dataset.columns)

# you want all rows, and the feature_cols' columns
p = new_dataset.iloc[:, 0: n_features].values
q = new_dataset.iloc[:, 2: 3].values


a = new_dataset.iloc[:, 0: n_features]
b = new_dataset.iloc[:, 2: 3]

print(a)

X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

cf_df = objpandas.DataFrame(a) 

print(cf_df)

column_ind = [0,1,2,3,4,5,6,7,8,9,10]
new_names = ['age','user_id','education','category','martial_status','occupation','relationship','gender','hours','country_from','salary']
old_names = cf_df.columns[column_ind]
cf_df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
sp = Synthpop()


dtypes = {"age": "int",
 "user_id": "int",
 "education": "int",
 "category": "int",
 "martial_status": "int",
 "occupation": "int",
 "relationship": "int",
 "gender": "int",
 "hours": "int",
 "country_from": "int",
 "salary": "int"
 }

sp.fit(a, dtypes)

synth_df= spop.generate(len(a))

print(synth_df)

m = synth_df.iloc[:, 0: n_features].values
n = synth_df.iloc[:, 2: 3].values

print(m)

data_list = [columnHeaders]

#Function to match output
def matcher(test, output):
        
    test.append([' '.join(map(str, output.values[i])) for i in range(len(output))][0])
    
    #for i in range(len(output)):

        # using list comprehension 
        #listToStr = ' '.join(map(str, output.values[i]))

        #print(listToStr) 

        #test.append(listToStr)
        #test.add(listToStr)
    
#Function to convert numeric to string and objects ones
def numericstoobjectConverter(k, z, t, test):
    
    #print(n_transform[k]);
    
    #for i in range(len(n_transform[k])):
	
        #print(round(t), ":::" , n_transform[k][i]);
		
        #if round(t) in n_transform[k].values[i]:
        if (round(t) in n_transform[k]):

            #print('[round(t)]', [round(t)])  
            #print('Present', n_transform[k])  
            #print('n_name[k]::', n_name[k])

            list1 = []
            list1.append(round(t))

            #input data
            dff = objpandas.DataFrame({n_name[k]: list1})

            output = enc[z].inverseTransform(dff)

            #print(output.values[0])

            matcher(test, output);
            
            #t2 = threading.Thread(target=matcher, args=(test, output,))     
            #t2.start()
            #t2.join()

            #break;              
   
#Function to iterate objects
def objectItertaor(k, z, test, d):
    
    for t in d[0]:  
        
        #print('t::', t) 
        
        if (z in n_columns):
            
            #print('z::', z)
            
            #print('-----------------------')     
            
            numericstoobjectConverter(k, z, t, test);
            
            #t2 = threading.Thread(target=numericstoobjectConverter, args=(k, z, t, test,))     
            #t2.start()
            #t2.join()

            #print('-----------------------') 
            
            k += 1;
            
        else:
        
            test.append(t)            
        
        z += 1;
	
def processData(*m):
    
	#print('train_Xaxis::', train_Xaxis)

	#for d in range(len(train_Xaxis)):

	#print('-----------------------')

	k = 0;
	z = 0;

	test = [];    

	objectItertaor(k, z, test, m);    

	#t1 = threading.Thread(target=objectItertaor, args=(k, z, test, train_Xaxis[d],))     
	#t1.start()
	#t1.join()

	#print('-----------------------')            

	data_list.append(test)
    
def writetofile(data_list):
    
    #write data to csv files
    with open('/Output/SYNTHPOP/SpOutputAdult10.csv', 'w', newline='') as file:
        objwriter = csv.writer(file, delimiter=',')
        objwriter.writerows(data_list)

pool = Pool(1000);
pool.map(processData, m);
pool.close()
pool.join()

#t4 = threading.Thread(target=processData, args=(train_Xaxis,))     
#t4.start()
#t4.join()

t5 = threading.Thread(target=writetofile, args=(data_list,))     
t5.start()
t5.join()

synthetic = objpandas.read_csv('/Output/SYNTHPOP/SpOutputAdult10.csv', encoding='unicode_escape', low_memory=False)


endTime = time.time();

print("Processing Time In Seconds:::", (endTime - startTime))

df_real = objpandas.DataFrame(data = p,
          index = objnumpy.array(range(0, len(p))),
          columns = [columnHeaders])

df_synthetic = objpandas.DataFrame(data = m,
          index = objnumpy.array(range(0, len(m))),
          columns = [columnHeaders])
          
		
fig, ax = plt.subplots(figsize=(10,10))
		
sns_plot = sns.heatmap(df_real.corr() - df_synthetic.corr(), annot=True, fmt=".2f", ax=ax, cmap="pink_r")

plt.title('Sp-Np', fontsize = 20)
plt.xlabel("Real Data",fontweight='bold') 
plt.ylabel("Synthetic Data",fontweight='bold') 

sns_plot.figure.savefig("/Output/SYNTHPOP/SPN_Heatmap200.png", dpi=1200)
#sns_plot.figure.savefig(args["heatmap"], dpi=1200)

plt.show()

diff = df_real.corr() - df_synthetic.corr();

meanValue = diff.mean()

#print(r2_score(df_real, df_synthetic))
  
print("Mean is :", meanValue)

meanList = []
meanList.append(meanValue.values)

#write data to csv files
with open('/Output/OutputAdult10.csv', 'a', newline='') as file:
#with open(args["finaloutput"], 'a', newline='') as file:
	objwriter = csv.writer(file, delimiter=',')
	objwriter.writerows(meanList)

warnings.filterwarnings('ignore')

eval_score = evaluate(synthetic, dataset)
eval_roc = eval_score.mean()
eval_list = []
eval_list.append(eval_roc)
print('SD Metrics :', eval_score)

with open('/Output/ROCOutputAdult10.csv', 'a', newline='') as file:
	objwriter = csv.writer(file, delimiter=',')
	objwriter.writerows(map(lambda x: [x], eval_list))
