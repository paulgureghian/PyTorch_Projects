""" Created by Paul A. Gureghian on 9/26/2018 """
""" This Python script uses keras to predict bitcoin price """
""" I picked a Recurrent Neural Network and a Bitcoin dataset """

### import packages 
#import os
import numpy as np 
import pandas as pd 
from statistics import mean 
from matplotlib import pyplot as plt 

from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import History 
history = History() 

import mlflow 
import mlflow.keras
mlflow.set_tracking_uri('/Users/paulgureghian/mlruns')   

from sklearn.preprocessing import MinMaxScaler

### read in the dataset to a dataframe 
pd.set_option('display.max_columns', 8)
pd.set_option('display.width', 1000)
df = pd.read_csv('/bitstamp.csv')

print(df.head())
print('')
print(df.shape) 
print('') 

### encode the date 
df['date'] = pd.to_datetime(df['Timestamp'], unit ='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()

print(Real_Price.head())
print('') 
print(Real_Price.shape)  
print('')

### split dataset into train and test sets 
prediction_days = 30 
df_train = Real_Price[:len(Real_Price) - prediction_days]
df_test = Real_Price[len(Real_Price) - prediction_days:]

print(df_train.head())
print('')
print(df_train.shape)
print('')
print(df_test.head())
print('')
print(df_test.shape)
print('')

### preprocess the data by reshaping it 
training_set = df_train.values 
training_set = np.reshape(training_set, (len(training_set),1))

print("Training set after reshaping:")
print('')
print(training_set)
print('')
print(training_set.shape)
print('')
 
### preprocess the data by scaling it 
sc = MinMaxScaler() 
training_set = sc.fit_transform(training_set) 
X_train = training_set[0 : len(training_set) -1]
y_train = training_set[1 : len(training_set)]
X_train = np.reshape(X_train, (len(X_train),1, 1)) 
                       
print("Scaled training set:")
print('')
print(training_set)
print('')
print("Define X_train")
print('')
print(X_train)
print('')
print(X_train.shape)
print("Define y_train:")
print('')
print(y_train)
print('')
print(y_train.shape)
print('')
print("X_train reshaped:")
print('')
print(X_train) 
print('') 
print(X_train.shape)
print('')
     
### instantiate the RNN model object 
regr = Sequential() 

### add the input and LSTM layers 
regr.add(LSTM(units =4, activation ='sigmoid', input_shape =(None, 1)))    

### add the output layer
regr.add(Dense(units =1))

### compile the RNN 
optimizer = 'RMSprop' 
regr.compile(optimizer, loss = 'mean_squared_error', metrics = ['mae']) 

### fit the model on the training set 
batch_size = 15
epochs = 20
history = regr.fit(X_train, y_train, batch_size, epochs, callbacks=[history]) 

loss = history.history['loss']  
loss = mean(loss)
print("loss_mean_squared_error: ", loss) 

metrics = history.history['mean_absolute_error']
metrics = mean(metrics) 
print("metrics_mae: ", metrics) 
print('')

### create predictions on the test set 
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1)) 
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = regr.predict(inputs) 
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)

print("Test set after reshaping:")
print('')
print(inputs)
print('')
print(inputs.shape)
print('')
print("Scaled inputs:")
print('')
print(inputs) 
print('')
print("Reshaped inputs:")
print('')
print(inputs)
print('')
print(inputs.shape)
print('')
print("Predicted BTC price: ", predicted_BTC_price)
print('')
print("Scaled predicted BTC price: ", predicted_BTC_price)
print('')

### get evaluation of the model predictions 
model_evaluation = regr.evaluate(inputs, predicted_BTC_price) 
model_evaluation = float(model_evaluation[0])
print("Model evaluation is: ", model_evaluation)   
print('') 
            
### visualize the results 
print("Visualize the results:") 
print('')

### plot the actual and predicted prices 
fig = plt.figure(figsize =(25, 20), dpi =80, facecolor ='w', edgecolor ='k')
ax = plt.gca() 
plt.plot(test_set, color = 'red', label = "Real BTC Price") 
plt.plot(predicted_BTC_price, color = 'blue', label = "Predicted BTC Price") 
plt.title("BTC Price Prediction", fontsize = 40)
plt.axis('tight') 

### reindex the 'df_test' dataframe 
df_test = df_test.reset_index() 
x = df_test.index

### set labels
labels = df_test['date']

### set xticks 
plt.xticks(x, labels, rotation = 'vertical')

### set fontsize for 'x' and 'y' ticks 
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)

for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
    
### set plot labels 
plt.xlabel('Time', fontsize = 40)
plt.ylabel('BTC Price(USD)', fontsize = 40)

### set plot legend
plt.legend(loc = 2, prop = {'size' : 25})  

### show the plot 
plt.show()    

### save the plot 
fig.savefig('btc_price_prediction_plot.png') 

### log params with mlflow
with mlflow.start_run() as run:
      mlflow.log_param("epochs", epochs) 
      mlflow.log_param("optimizer", optimizer)
      mlflow.log_param("batch_size", batch_size)
      
### log metrics with mlflow        
      mlflow.log_metric("loss_mse", loss) 
      mlflow.log_metric("metrics_mae", metrics)
      mlflow.log_metric("model_evaluation", model_evaluation)  

### log artifacts and model with mlflow      
      mlflow.log_artifact('btc_price_prediction_plot.png')  
     
      model_path = "models"
      mlflow.keras.log_model(regr, model_path)
        
      with open("info.txt", "w") as f:
          f.write("btc_price_prediction_plot")
      mlflow.log_artifact("info.txt")     

    



























  