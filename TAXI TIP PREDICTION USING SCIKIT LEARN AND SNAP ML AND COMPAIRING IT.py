from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_squared_error
import time
import warnings
import gc,sys


#READING AND FILTERING THE DATA
raw_data=pd.read_csv('yellow_tripdata_2019-06.csv')
# print("There are "+str(len(raw_data))+" observation in the dataset.")
# print("There are " + str(len(raw_data.columns)) + " variables in the dataset.")
# print(raw_data.head())
#REDUCING THE DATASIZE TO 100000 RECORDS
raw_data=raw_data.head(100000)
gc.collect()
#removing the rows that have 0 as the tip
raw_data=raw_data[raw_data['tip_amount']>0]
#removing those whose tip was larger than total amount
raw_data=raw_data[(raw_data['tip_amount']<=raw_data['fare_amount'])]
#removing trips with very large fare cost
raw_data=raw_data[(raw_data['fare_amount']>=2)&(raw_data['fare_amount']<200)]
#removing the variable that include target amount in it #because of target variable
clean_data=raw_data.drop(['total_amount'],axis=1)
#clearing the garbage memory
del raw_data
gc.collect()
# print(clean_data.size,clean_data.shape)
# print("There are "+str(len(clean_data))+" observations in the dataset")
# print("There are "+str(len(clean_data.columns))+" variables in the dataset")
#VISUALINSING THE DATA
# plt.hist(clean_data.tip_amount.values,16,histtype='bar',facecolor='g')
# plt.show()
#FINDING THE MAX AND MIN AMOUNT
# print("Minimum amount value is ",np.min(clean_data.tip_amount.values))
# print("Maximum amount value is ", np.max(clean_data.tip_amount.values))
# print("90% of the trips have a tip amount less or equal than",np.percentile(clean_data.tip_amount.values,90))
# print(clean_data.head())

#DATASET PREPROSESSING
#converting the pickup and dropoff date and time to pandas date and time format
clean_data['tpep_dropoff_datetime']=pd.to_datetime(clean_data['tpep_dropoff_datetime'])
clean_data['tpep_pickup_datetime']=pd.to_datetime(clean_data['tpep_pickup_datetime'])
#extracting the pickup and dropoff hour
clean_data['pickup_hour']=clean_data['tpep_pickup_datetime'].dt.hour
clean_data['dropoff_hour']=clean_data['tpep_dropoff_datetime'].dt.hour
#extracting pickup and dropoff day of the week
clean_data['pickup_day']=clean_data['tpep_pickup_datetime'].dt.weekday
clean_data['dropoff_day']=clean_data['tpep_dropoff_datetime'].dt.weekday
#compute trip time in minutes
clean_data['trip_time']=(clean_data['tpep_dropoff_datetime']-clean_data['tpep_pickup_datetime'])/ np.timedelta64(1, 'm')
# print(clean_data.size,clean_data.shape)
# first_n_rows = 1000000
# clean_data = clean_data.head(first_n_rows)
#REMOVING THE PICKUP,DROPOFF DATE AND TIME COLUMNS FROM THE DATA
clean_data=clean_data.drop(['tpep_pickup_datetime','tpep_dropoff_datetime'],axis=1)
# some features are categorical, we need to encode them
# to encode them we use one-hot encoding from the Pandas package
get_dummy_col=["VendorID","RatecodeID","store_and_fwd_flag","PULocationID","DOLocationID","payment_type","pickup_hour","dropoff_hour","pickup_day","dropoff_day"]
processed_data=pd.get_dummies(clean_data,columns=get_dummy_col)
#releasing the garbage memory
del clean_data
gc.collect()
# extract the target labels from the dataframe
y=processed_data[['tip_amount']].values.astype('float32')
# drop the target variable from the feature matrix
processed_data=processed_data.drop(['tip_amount'],axis=1)
#get the feature matrix for training
X=processed_data.values
#normalising the feature matrix
X=normalize(X,axis=1,norm='l1',copy=False)
#print the shape of the features matrix an the labels vector
# print('X.shape=',X.shape,'y.shape=',y.shape)
#DATASET TRAIN/TEST SPLIT
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
# print('X_train.shape=',X_train.shape,'Y_train.shape=',y_train.shape)
# print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)




#BUILDING A DECISION TREE REGRESSOR MODEL WITH SCIKIT-LEARN
from sklearn.tree import DecisionTreeRegressor
# for reproducible output across multiple function calls, set random_state to a given integer value
sklearn_dt=DecisionTreeRegressor(max_depth=8,random_state=35)
#TRAINING A DECISION TREE REGRESSOR USING SCIKIT-LEARN
t0=time.time()
sklearn_dt.fit(X_train,y_train)
sklearn_time=time.time()-t0
print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

#BUILDING A DECISION TREE REGRESSOR MODEL WITH SNAP ML
from snapml import DecisionTreeRegressor

# in contrast to sklearn's Decision Tree, Snap ML offers multi-threaded CPU/GPU training
# to use the GPU, one needs to set the use_gpu parameter to True
# snapml_dt = DecisionTreeRegressor(max_depth=4, random_state=45, use_gpu=True)

# to set the number of CPU threads used at training time, one needs to set the n_jobs parameter
# for reproducible output across multiple function calls, set random_state to a given integer value
snapml_dt=DecisionTreeRegressor(max_depth=8,random_state=45,n_jobs=4)

#training the model using snap ml
t0=time.time()
snapml_dt.fit(X_train,y_train)
snapml_time=time.time()-t0
print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))

#EVALUATING THE SCIKIT-LEARN AND SNAP-MLDECISION TREE REGRESSOR MODELS
training_speedup=sklearn_time/snapml_time
print('[Decision Tree Regressor] Snap ML vs. Scikit-Learn speedup : {0:.2f}x '.format(training_speedup))

#PREDICTING
sklearn_pred=sklearn_dt.predict(X_test)

sklearn_mse=mean_squared_error(y_test,sklearn_pred)
print('[Scikit-Learn] MSE score: {0:.3f}'.format(sklearn_mse))

snapml_pred=snapml_dt.predict(X_test)
snapml_mse=mean_squared_error(y_test,snapml_pred)
print('[Snap ML] MSE score: {0:.3f}'.format(snapml_mse))



