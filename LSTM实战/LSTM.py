from pandas import read_csv
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,Input,merge,Convolution1D, MaxPooling1D,LSTM
def parse(x):
    return datetime.strptime(x,'%Y %m %d %H')


#dataset = read_csv('raw.csv',parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
#dataset.drop('No',axis=1,inplace=True)
#dataset.columns = ['pollution','dew','temp','press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
#dataset.index.name = 'date'
#dataset['pollution'].fillna(0,inplace=True)
#dataset = dataset[24:]
#print(dataset.head(5))
#dataset.to_csv('pollution.csv')



dataset = read_csv('pollution.csv',index_col=0,header=0)
values = dataset.values
group = [0,1,2,3,5,6,7]

m=1
#plt.figure()
#for i in group:
	#plt.subplot(7,1,m)
	#plt.plot(values[:,i])
	#plt.title(dataset.columns[i],loc='right')
	#m += 1
#plt.show()



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):    
	n_vars = 1 if type(data) is list else data.shape[1]    
	df = pd.DataFrame(data)    
	cols, names = list(), list()    # input sequence (t-n, ... t-1)    
	for i in range(n_in, 0, -1):        
		cols.append(df.shift(i))        
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]    # forecast sequence (t, t+1, ... t+n)    
	for i in range(0, n_out):        
		cols.append(df.shift(-i))        
		if i == 0:            
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]        
		else:            
			names += [('var%d(t+%d)' % (j+1, i)) 
	for j in range(n_vars)]    # put it all together    
		agg = pd.concat(cols, axis=1)    
		agg.columns = names    # drop rows with NaN values 
		#print(agg.head())   
		if dropnan:        
			agg.dropna(inplace=True)    
	return agg # load 

datasetdataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values# integer encode direction
encoder = preprocessing.LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])# ensure all data is float
values = values.astype('float32')# normalize features
print(values.shape)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)

print(reframed.head())



# split into train and test sets

values = reframed.values

n_train_hours = 365 * 24

train = values[:n_train_hours, :]

test = values[n_train_hours:, :]

# split into input and outputs

train_X, train_y = train[:, :-1], train[:, -1]

test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


model = Sequential()
model.add(LSTM(50,input_shape=(train_X.shape[1],train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae',optimizer='adam')
history = model.fit(train_X,train_y,epochs=50,batch_size=72,validation_data=(test_X,test_y),verbose=2,shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()




from sklearn.metrics import mean_squared_error
import numpy as np
import math
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]# calculate 
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
