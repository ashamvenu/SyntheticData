from collections import defaultdict
import csv
import random
import time
import os, multiprocessing
os.environ["OMP_NUM_THREADS"] = "100"
from dateutil.parser import parse
import numpy as objnumpy
import pandas as objpandas
import keras.models
from keras.models import Sequential
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
from DataSynthesizer.datatypes.utils.DataType import DataType
from sdv.evaluation import evaluate
import datetime
from datetime import timedelta
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import warnings
from pandas.plotting import scatter_matrix
import statistics

startTime = time.time();
# input dataset
input_data = 'Input/adult10k.csv'
 
# location of output file
mode = 'random_mode'
description_file = f'/Output/DS/out/description.json'
synthetic_data = f'/Output/DS/out/data_synth.csv'
# An attribute is categorical if its domain size is less than this threshold.
threshold_value = 30
# Number of tuples generated in synthetic dataset.
num_tuples_to_generate = 10000 # Here 32561 is the same as input dataset, but it can be set to another number.

describer = DataDescriber(category_threshold=threshold_value)
describer.describe_dataset_in_random_mode(input_data)
describer.save_dataset_description_to_file(description_file)

generator = DataGenerator()
generator.generate_dataset_in_random_mode(num_tuples_to_generate, description_file)
generator.save_synthetic_data(synthetic_data)

synthetic = objpandas.read_csv(synthetic_data, encoding='unicode_escape', low_memory=False)
'''
def str_time_prop(start, end, format, prop):
    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))
    ptime = stime + prop * (etime - stime)
    return time.strftime(format, time.localtime(ptime))

selected_format = '%d-%m-%Y'

def random_date(start, end, prop):
    return parse(str_time_prop(start, end, selected_format, prop)).strftime(selected_format)


print(random_date("01-01-2010", "01-01-2010", random.random()))

def make_date(x):
    return random_date("01-01-2010", "30-06-2014", random.random())

synthetic['timestamp'] = synthetic['timestamp'].apply(make_date)
#dataset['timestamp'] = pd.to_datetime(dataset['timestamp']).apply(lambda x: x.date())
#dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], format='%y%m%d')
#dataset['timestamp'] = dataset['timestamp'].apply(lambda d: datetime.date.fromtimestamp(d/1000.0))
print (synthetic)
'''

dataset = objpandas.read_csv(input_data, encoding='unicode_escape', low_memory=False)

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
a = new_dataset.iloc[:, 0: n_features].values
b = new_dataset.iloc[:, 2: 3].values

print(a)


class DataLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_dict = defaultdict(list)

    def fit(self, Y):
        Y = Y.astype('category')
        cols1 = Y.columns
        values1 = list(map(lambda col: Y[col].cat.categories, cols1))
        self.label_dict = dict(zip(cols1, values1))
        # return as category for xgboost or lightgbm 
        return self

    def transform(self, Y):
        # check missing columns
        missing_col1 = set(Y.columns)-set(self.label_dict.keys())
        if missing_col1:
            raise ValueError('the column named {} is not in the label dictionary. Check your fitting data.'.format(missing_col1)) 
        return Y.apply(lambda y: y.astype('category').cat.set_categories(self.label_dict[y.name]).cat.codes.astype('category').cat.set_categories(objnumpy.arange(len(self.label_dict[y.name]))))

    def inverseTransform(self, Y):
        return Y.apply(lambda y: objpandas.Categorical.from_codes(codes=y.values,
                       categories=self.label_dict[y.name]))

#Print the count of rows and coulmns in csv file
print("Dimensions of Dataset: {}".format(synthetic.shape))

# Dropped all the Null, Empty, NA values from csv file 
new_dataset1 = synthetic.dropna(axis=0, how='any') 

print("Dimensions of Dataset after Pre-processing : {}".format(new_dataset1.shape))

#Encoding data other than int, float and double
#le = preprocessing.LabelEncoder()
#enc = PandasLabelEncoder()
enc1  = {}

#Get Other Datatype Columns Only
n_columns1 = []
n_transform1 = []
n_name1 = []
count1 = 0;

#Function to convert string, objects to numeric ones
def numericsConverter1(columnName):
  
    #print("Argumnet: ", columnName) 
  
    #input data
    dff1 = objpandas.DataFrame({columnName: new_dataset1[columnName]})

    enc1[count1] = DataLabelEncoder()

    input1 = enc1[count1].fit_transform(dff1);

    #print("Input ", input)
  
    return input1;

#Method to find datatype of columns in csv file
for name1, dtype in new_dataset1.dtypes.iteritems():
    
    print("Data Type of Coulmn " + name1 + " is : ", dtype)
    
    if(dtype == 'object'):
        
        #print("Transform Coulmn " + name)	

        input1 = numericsConverter1(name1);
        
        new_dataset1[name1] = input1;
        
        n_columns1.append(objnumpy.array(count1))
        n_name1.append(objnumpy.array(name1))
        n_transform1.append(objnumpy.array(input1))
        
    count1 += 1;

n_columns1 = objnumpy.array(n_columns1);
n_name1 = objnumpy.array(n_name1);
n_transform1 = objnumpy.array(n_transform1);

#Get numeric Values Only
#new_dataset = new_dataset._get_numeric_data()
#new_dataset = new_dataset.iloc[ : , 0 : 10000]

#New dataset after encoding
#new_dataset = new_dataset.apply(le.fit_transform)

#Get the coulmn header
columnHeaders1 = new_dataset1.head(0)

#Get Feature Coulmns Only
n_features1 = len(new_dataset1.columns)
    
print("Features: ", n_features1)

A = range(0, n_features1);
  
new_dataset1.columns = list(A)

print("new_dataset.columns: ", new_dataset1.columns)

# you want all rows, and the feature_cols' columns
m = new_dataset1.iloc[:, 0: n_features1].values
n = new_dataset1.iloc[:, 2: 3].values

print(m)

synthetic.to_csv('/Output/DS/DSAdult10.csv', index=False)

endTime = time.time();

print("Processing Time In Seconds:::", (endTime-startTime))

df_real = objpandas.DataFrame(data = a,
          index = objnumpy.array(range(0, len(a))),
          columns = [columnHeaders])

df_synthetic = objpandas.DataFrame(data = m,
          index = objnumpy.array(range(0, len(m))),
          columns = [columnHeaders])
          
		
fig, ax = plt.subplots(figsize=(10,10))
		
sns_plot = sns.heatmap(df_real.corr() - df_synthetic.corr(), annot=True, fmt=".2f", ax=ax, cmap="pink_r")

plt.title('DS', fontsize = 20)
plt.xlabel("Real Data",fontweight='bold') 
plt.ylabel("Synthetic Data",fontweight='bold') 

sns_plot.figure.savefig("/Output/DS/DS_Heatmap200.png", dpi=1200)
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



##X = m
# create y
##Y = a
# split to train and test
##X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

##model = Sequential()

##print(X_train)
##print(Y_train)

##model.add(Dense(50, input_dim=8, activation='relu', kernel_initializer='he_uniform'))
##model.add(Dense(8, activation='softmax'))
#model.add(Embedding(max_features, 128, input_length=max_len, dropout=0.2))
#model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
#model.add(Dense(2))
#model.add(Activation('sigmoid'))

##model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



##model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, verbose=0)

#predictions = model.predict(Y_train)

##Train_acc = model.evaluate(X_train, Y_train, verbose=0)
##Test_acc = model.evaluate(X_test, Y_test, verbose=0)
##print("%s: %.2f%%" % (model.metrics_names[1], Train_acc[1]*100))
##print("%s: %.2f%%" % (model.metrics_names[1], Test_acc[1]*100))


#lr = LogisticRegression() 
#lr.fit(X_train, Y_train.astype(objnumpy.int64))

#predictions = lr.predict(X_train)

# print classification report 
#print(classification_report(X_train, Y_train.astype(objnumpy.int64)))
warnings.filterwarnings('ignore')
#print(classification_report(X_train, predictions))
eval_score  = evaluate(synthetic, dataset)
eval_roc = eval_score.mean()
eval_list = []
eval_list.append(eval_roc)
print('SD Metrics :', eval_score)

with open('/Output/ROCOutputAdult10.csv', 'a', newline='') as file:
	objwriter = csv.writer(file, delimiter=',')
	objwriter.writerows(map(lambda x: [x],eval_list))