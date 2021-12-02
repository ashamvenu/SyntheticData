#import libraries 
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import gc
from collections import defaultdict
import csv
import numpy as objnumpy
import pandas as objpandas
from random import randint
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
#from keras.layers import CuDNNLSTM 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
#from keras.utils import plot_model
from tensorflow.keras import backend as K
from multiprocessing.pool import ThreadPool as Pool
import time
import threading
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

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

    def inverse_transform(self, X):
        return X.apply(lambda x: objpandas.Categorical.from_codes(codes=x.values,
                       categories=self.label_dict[x.name]))
 
startTime = time.time();

#Load the data-set
dataset = objpandas.read_csv('F:/MS-CS/Thesis/Dataset/electronics_200.csv', encoding='unicode_escape', low_memory=False) 

#Print the count of rows and coulmns in csv file
print("Dimensions of Dataset: {}".format(dataset.shape))

# Dropped all the Null, Empty, NA values from csv file 
new_dataset = dataset.dropna(axis=0, how='any') 

print("Dimensions of Dataset after Pre-processing : {}".format(new_dataset.shape))

#Encoding data other than int, float and double
enc  = {}

#Get Other Datatype Columns Only
n_columns = []
n_transform = []
n_nantransform = []
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

#Method to find datatype of coulums in csv file
for name, dtype in new_dataset.dtypes.iteritems():
    
    print("Data Type of Coulmn " + name + " is : ", dtype)
    
    if(dtype == 'object'):
        
        #print("Transform Coulmn " + name)	
         
        n_nantransform.append(objnumpy.array(new_dataset[name]))
          
        input = numericsConverter(name);

        new_dataset[name] = input
        
        n_columns.append(objnumpy.array(count))
        n_name.append(objnumpy.array(name))
        n_transform.append(objnumpy.array(input))      
        
    count += 1;

n_columns = objnumpy.array(n_columns);
n_name = objnumpy.array(n_name);
n_transform = objnumpy.array(n_transform);

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

print("a: ", a)

print("Length of a: ", len(a))

encoded_X = []
encoded_XX = [columnHeaders]

def processInputData(a):
    
	for x in range(len(a)):

		gc.collect()

		K.clear_session()

		# define input sequence
		sequence = array(a[x])

		# reshape input into [samples, timesteps, features]
		n_in = len(sequence)
		sequence = sequence.reshape((1, n_in, 1))

		# define model
		model = Sequential()
		model.add(LSTM(100, activation = 'relu', input_shape = (n_in, 1)))
		model.add(RepeatVector(n_in))
		model.add(LSTM(100, activation = 'relu', return_sequences = True))
		model.add(TimeDistributed(Dense(1)))
		model.compile(optimizer = 'adam', loss = 'mse')

		# fit model
		model.fit(sequence, sequence, epochs = 300, verbose = 0)
		#plot_model(model, show_shapes=True, to_file='E:/GAN/reconstruct_lstm_autoencoder.png')

		# demonstrate recreation
		yhat = model.predict(sequence)

		encoded_X.append(yhat[0,:,0]);

		print(yhat[0,:,0])
   
t1 = threading.Thread(target=processInputData, args=(a,))     
t1.start()
t1.join()
        
#Print Coulmn Headers
print(columnHeaders)

data_list = [columnHeaders]

#Function to match output
def matcher(test, output):
        
    test.append([' '.join(map(str, output.values[i])) for i in range(len(output))][0])
    
    #for i in range(len(output)):

        # using list comprehension 
        #listToStr = ' '.join(map(str, output.values[i]))

        #print(listToStr) 

        #test.append(listToStr)
    
#Function to convert numeric to string and objects ones
def numericstoobjectConverter(k, z, t, test):
    
    y = 0;
    
    for i in range(len(n_transform[k])):
                
        #if t in n_transform[k].values[i]:
        if (round(t) in n_transform[k][i]):

            #print('Present', n_transform[k])  

            list1 = []
            list1.append(t)

            #input data
            dff = objpandas.DataFrame({n_name[k] : list1})

            output = enc[z].inverse_transform(dff)

            #print(output.values[0])

            matcher(test, output);
    
            y += 1; 

            break;  
            
    return y;
            
#Function to convert numeric to string and objects ones
def numericstoobjectConverterforNAN(k, test):

    value = randint(1, len(n_nantransform[k]));
            
    #print('len(n_nantransform[k]):', len(n_nantransform[k]))  
    #print('value:', value)  

    for q in range(len(n_nantransform[k])):

        if q == (value - 1):

            test.append(n_nantransform[k][q])     
                            
#Function to iterate objects
def objectItertaor(k, z, test, d):
    
    for t in d:  
        
        #print('t::',t) 
        
        if z in n_columns:
            
            #print('z::', z)
            
            #print('-----------------------')     
            
            #print(n_transform[k]);
            
            #rint('round:', abs(round(int(t))))  
                
            t = abs(round(int(t)))
                          
            y = numericstoobjectConverter(k, z, t, test)                                 

            if y == 0:
                
                numericstoobjectConverterforNAN(k, test)   
                
            #print('-----------------------') 
            
            k += 1;
            
        else:
        
            t = abs(round(int(t)))
            
            test.append(t)
        
        z += 1;
 
def processData(encoded_X):
    
    #for d in range(len(encoded_X)):    

        #print('-----------------------')     

        k = 0;
        z = 0;

        test = [];

        objectItertaor(k, z, test, encoded_X);

        #print('-----------------------')            

        data_list.append(test)
             
def writetofile(data_list):
    
    #write data to csv files
    with open('F:/MS-CS/Thesis/Dataset/VOutput200.csv', 'w', newline='') as file:
        objwriter = csv.writer(file, delimiter=',')
        objwriter.writerows(data_list)

pool = Pool(1000);
pool.map(processData, encoded_X);
pool.close()
pool.join()

#t4 = threading.Thread(target=processData, args=(encoded_X,))     
#t4.start()
#t4.join()

t5 = threading.Thread(target=writetofile, args=(data_list,))     
t5.start()
t5.join()

endTime = time.time();

print("Processin Time In Seconds:::", (endTime - startTime))

encoded_X = objnumpy.array(encoded_X)

df_real = objpandas.DataFrame(data = a,
          index = objnumpy.array(range(0, len(a))),
          columns = [columnHeaders])

#print(df_real)

df_synthetic = objpandas.DataFrame(data = encoded_X,
			   index = objnumpy.array(range(0, len(encoded_X))),
               columns = [columnHeaders])
		  
#print(df_synthetic)

fig, ax = plt.subplots(figsize=(10,10))

sns_plot = sns.heatmap(df_real.corr() - df_synthetic.corr(), annot=True, fmt=".2f", ax=ax, cmap="twilight")

plt.title('Variable Auto-Encoder', fontsize = 20)
plt.xlabel("Real Data",fontweight='bold') 
plt.ylabel("Synthetic Data",fontweight='bold') 

sns_plot.figure.savefig("F:/MS-CS/Thesis/Dataset/V_Heatmap200.png", dpi=1200)

plt.show()


diff = df_real.corr() - df_synthetic.corr();

meanValue = diff.mean()
  
print("Mean is :", meanValue)

meanList = []
meanList.append(meanValue.values)

#write data to csv files
with open('F:/MS-CS/Thesis/Dataset/Output200.csv', 'a', newline='') as file:
	objwriter = csv.writer(file, delimiter=',')
	objwriter.writerows(meanList)