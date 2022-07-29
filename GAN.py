import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import psutil ; 
print(list(psutil.virtual_memory())[0:2])
import numpy as objnumpy
import pandas as objpandas
#import xgboost as xgb
import pickle
import gc
import csv
gc.collect()
print(list(psutil.virtual_memory())[0:2])
import GAN_Models
import importlib
importlib.reload(GAN_Models) 
from GAN_Models import *
import threading
from multiprocessing.pool import ThreadPool as Pool
import os, multiprocessing, time
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from collections import defaultdict
from sdv.evaluation import evaluate
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


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

#Load the data-set
dataset = objpandas.read_csv('/home/mannara/SyntheticData/Input/adult10k.csv', encoding='unicode_escape', low_memory=False)

#Print the count of rows and coulmns in csv file
print("Dimensions of Dataset: {}".format(dataset.shape))

# Dropped all the Null, Empty, NA values from csv file 
new_dataset = dataset.dropna(axis=0, how='any') 
new_dataset1 = dataset.dropna(axis=0, how='any') 
new_dataset2 = dataset.dropna(axis=0, how='any')

print("Dimensions of Dataset after Pre-processing : {}".format(new_dataset.shape))

#Encoding data other than int, float and double
enc  = {}

#Get Other Datatype Columns Only
n_columns = []
n_transform = []
n_name = []
count = 0;

#Get the coulmn header
columnHeaders = new_dataset.head(0)

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

        input = numericsConverter(name);
        
        new_dataset[name] = input;
        new_dataset1[name] = input;
        new_dataset2[name] = input;
        
        n_columns.append(objnumpy.array(count))
        n_name.append(objnumpy.array(name))
        n_transform.append(objnumpy.array(input))
        
    count += 1;

n_columns = objnumpy.array(n_columns);
n_name = objnumpy.array(n_name);
n_transform = objnumpy.array(n_transform);

print('# new_dataset: ',new_dataset.shape)

# data columns will be all other columns except class
label_cols = ['rating']
#data_cols = list(new_dataset.columns[ new_dataset.columns != 'rating' ])
data_cols = list(new_dataset.columns)

print('# of data columns: ',len(data_cols))

# you want all rows, and the feature_cols' columns
a = new_dataset2.iloc[:, 0: len(data_cols)].values

new_dataset = new_dataset[ data_cols ].copy()

print('# new_dataset: ',new_dataset)

data_list = [data_cols]

# Add KMeans generated classes to fraud data - see classification section for more details on this
import sklearn.cluster as cluster

#train = new_dataset1.loc[ new_dataset1['rating']==5 ].copy()
train = new_dataset1;

algorithm = cluster.KMeans
args, kwds = (), {'n_clusters':2, 'random_state':0}
labels = algorithm(*args, **kwds).fit_predict(train[ data_cols ])

print( pd.DataFrame( [ [np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) ) )

fraud_w_classes = train.copy()
fraud_w_classes['Class'] = labels

# reloading the libraries and setting the parameters
import GAN_Models
import importlib
importlib.reload(GAN_Models) # For reloading after making changes
from GAN_Models import *

rand_dim = 9 # 32 # needs to be ~data_dim
base_n_count = 100 # 128

nb_steps = len(new_dataset) # 50000 # Add one for logging of the last interval
batch_size = 100 # 64

k_d = 1  # number of critic network updates per adversarial training step
k_g = 1  # number of generator network updates per adversarial training step
critic_pre_train_steps = 100 # 100  # number of steps to pre-train the critic before starting adversarial training
log_interval = 100 # 100  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
learning_rate = 5e-4 # 5e-5
data_dir = '/home/mannara/SyntheticData/Output/GAN/Cache/'
generator_model_path, discriminator_model_path, loss_pickle_path = None, None, None


# show = False
show = True 

# train = create_toy_spiral_df(1000)
# train = create_toy_df(n=1000,n_dim=2,n_classes=4,seed=0)
train = fraud_w_classes.copy().reset_index(drop=True) # fraud only with labels from classification

# train = pd.get_dummies(train, columns=['Class'], prefix='Class', drop_first=True)
label_cols = [ i for i in train.columns if 'Class' in i ]
data_cols = [ i for i in train.columns if i not in label_cols ]
#train[ data_cols ] = train[ data_cols ] / 10 # scale to random noise size, one less thing to learn
train_no_label = train[ data_cols ]
#print('# label_cols: ',len(label_cols))
#print('# data_cols: ',len(data_cols))
#print('# train_no_label: ',len(train_no_label))

# Training the vanilla GAN and CGAN architectures
k_d = 1 # number of critic network updates per adversarial training step

learning_rate = 5e-4 # 5e-5
arguments = [rand_dim, nb_steps, batch_size, 
             k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
            data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ]

try:
	x = adversarial_training_GAN(arguments, train_no_label, data_cols) # GAN
	print('# x: ',len(x))
except Exception as e:
	print("Main!", e.__class__, "occurred.")

#Function to match output
def matcher(test, output):
        
    test.append([' '.join(map(str, output.values[i])) for i in range(len(output))][0])
    
#Function to convert numeric to string and objects ones
def numericstoobjectConverter(k, z, t, test):
    
    if (round(t) in n_transform[k]):
	
        #print('round(t)', round(int(t)))
        #print('Present', n_transform[k])
        #print('n_name[k]::', n_name[k])
        
        list1 = [];
        list1.append(round(int(t)))
        
        #input data
        dff = objpandas.DataFrame({n_name[k]: list1})
        
        output = enc[z].inverseTransform(dff)
        
        #print(output.values[0])
        
        matcher(test, output);            
   
#Function to iterate objects
def objectItertaor(k, z, test, d):
    
    for t in d[0]:  
        
        #print('t::', t) 
        
        if (z in n_columns):
            
            #print('z::', z)
            
            #print('-----------------------')     
            
            numericstoobjectConverter(k, z, t, test);
            
            #print('-----------------------') 
            
            k += 1;
            
        else:
            t = abs(round(int(t)))
            
            test.append(t)
			
        z += 1;
	
def processData(*x):
	
	for d in range(len(x)):
		
		#print('-----------------------')

		k = 0;
		z = 0;

		test = [];    

		objectItertaor(k, z, test, x[d]);    

		#print('-----------------------')            

		data_list.append(test)
    
def writetofile(data_list):
    
    #write data to csv files
    with open('/home/mannara/SyntheticData/Output/GAN/GanAdult10.csv', 'w', newline='') as file:
        objwriter = csv.writer(file, delimiter=',')
        objwriter.writerows(data_list)

pool = Pool(10000);
pool.map(processData, x);
pool.close()
pool.join()

t5 = threading.Thread(target=writetofile, args=(data_list,))     
t5.start()
t5.join()

endTime = time.time();

print("Processing Time In Seconds:::", (endTime - startTime))

synthetic = objpandas.read_csv('/home/mannara/SyntheticData/Output/GAN/GanAdult10.csv', encoding='unicode_escape', low_memory=False)


x = objnumpy.array(x)

df_real = objpandas.DataFrame(data = a,
          index = objnumpy.array(range(0, len(a))),
          columns = [columnHeaders])
		  
#print(df_real)

df_synthetic = objpandas.DataFrame(data = x[0],
			   index = objnumpy.array(range(0, len(x[0]))),
               columns = [columnHeaders])
		  
#print(df_synthetic)

fig, ax = plt.subplots(figsize=(10,10))

sns_plot = sns.heatmap(df_real.corr() - df_synthetic.corr(), annot=True, fmt=".2f", ax=ax, cmap="pink_r")

plt.title('GAN', fontsize = 20)
plt.xlabel("Real Data",fontweight='bold') 
plt.ylabel("Synthetic Data",fontweight='bold') 

sns_plot.figure.savefig("/home/mannara/SyntheticData/Output/GAN/GanHeatmapAdult.png", dpi=1200)

plt.show()

diff = df_real.corr() - df_synthetic.corr();

meanValue = diff.mean()
  
print("Mean is :", meanValue)

meanList = []
meanList.append(meanValue.values)

#write data to csv files
with open('/home/mannara/SyntheticData/Output/OutputAdult10.csv', 'a', newline='') as file:
	objwriter = csv.writer(file, delimiter=',')
	objwriter.writerows(meanList)

eval_score = evaluate(synthetic, dataset)

eval_roc = eval_score.mean()
output = 'Output'
eval_list = [output]
eval_list.append(eval_roc)
print('SD Metrics:', eval_score)

with open('/home/mannara/SyntheticData/Output/ROCOutputAdult10.csv', 'w', newline='') as file:
	objwriter = csv.writer(file, delimiter=',')
	objwriter.writerows(map(lambda x: [x], eval_list))
