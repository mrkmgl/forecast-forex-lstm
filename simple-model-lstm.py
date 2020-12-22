#Setting modules
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint

import joblib
import sklearn
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

#Importing data and creating database
eurusd = pd.read_csv('/Users/marco/Desktop/Thesis Data/eurusd_hour.csv')
eurusd.drop('Date', axis='columns', inplace=True)
eurusd.drop('Hour', axis='columns', inplace=True)
eurusd.head(2)

#Visualizing data
plt.figure(figsize=(15,6))
plt.plot(eurusd.BidClose)
plt.title('Euro vs USD')
plt.show()

eurusdout1 = eurusd.drop('BidChange', axis=1)
eurusdout = eurusdout1.drop('AskChange', axis=1)
eurusdout.boxplot()

#Generate features 
def generate_features(df):
    df_new = pd.DataFrame()
    
    #5 original features for Bid and Ask
    df_new['BidOpen'] = df['BidOpen']
    df_new['BidOpen_1'] = df['BidOpen'].shift(1)
    df_new['BidClose_1'] = df['BidClose'].shift(1)
    df_new['BidHigh_1'] = df['BidHigh'].shift(1)
    df_new['BidLow_1'] = df['BidLow'].shift(1)
    df_new['AskOpen'] = df['AskOpen']
    df_new['AskOpen_1'] = df['AskOpen'].shift(1)
    df_new['AskClose_1'] = df['AskOpen'].shift(1)
    df_new['AskHigh_1'] = df['AskHigh'].shift(1)
    df_new['AskLow_1'] = df['AskLow'].shift(1)
    
    #Average prices
    df_new['BidAvgPrice_5'] = df['BidClose'].rolling(window=5).mean().shift(1)
    df_new['BidAvgPrice_30'] = df['BidClose'].rolling(window=21).mean().shift(1)
    df_new['BidAvgPrice_90'] = df['BidClose'].rolling(window=63).mean().shift(1)
    df_new['BidAvgPrice_365'] = df['BidClose'].rolling(window=252).mean().shift(1)

    df_new['AskAvgPrice_5'] = df['AskClose'].rolling(window=5).mean().shift(1)
    df_new['AskAvgPrice_30'] = df['AskClose'].rolling(window=21).mean().shift(1)
    df_new['AskAvgPrice_90'] = df['AskClose'].rolling(window=63).mean().shift(1)
    df_new['AskAvgPrice_365'] = df['AskClose'].rolling(window=252).mean().shift(1)
    
    #average price ratio
    df_new['BidRatioAvg_5_30'] = df_new['BidAvgPrice_5'] / df_new['BidAvgPrice_30']
    df_new['BidRatioAvg_5_90'] = df_new['BidAvgPrice_5'] / df_new['BidAvgPrice_90']
    df_new['BidRatioAvg_5_365'] = df_new['BidAvgPrice_5'] / df_new['BidAvgPrice_365']
    df_new['BidRatioAvg_30_90'] = df_new['BidAvgPrice_30'] / df_new['BidAvgPrice_90']
    df_new['BidRatioAvg_30_365'] = df_new['BidAvgPrice_30'] / df_new['BidAvgPrice_365']
    df_new['BidRatioAvg_90_365'] = df_new['BidAvgPrice_90'] / df_new['BidAvgPrice_365']
    
    df_new['AskRatioAvg_5_30'] = df_new['AskAvgPrice_5'] / df_new['AskAvgPrice_30']
    df_new['AskRatioAvg_5_90'] = df_new['AskAvgPrice_5'] / df_new['AskAvgPrice_90']
    df_new['AskRatioAvg_5_365'] = df_new['AskAvgPrice_5'] / df_new['AskAvgPrice_365']
    df_new['AskRatioAvg_30_90'] = df_new['AskAvgPrice_30'] / df_new['AskAvgPrice_90']
    df_new['AskRatioAvg_30_365'] = df_new['AskAvgPrice_30'] / df_new['AskAvgPrice_365']
    df_new['AskRatioAvg_90_365'] = df_new['AskAvgPrice_90'] / df_new['AskAvgPrice_365']
    
    #standard deviation of prices
    df_new['std_BidPrice_5'] = df['BidClose'].rolling(window=5).std().shift(1)
    df_new['std_BidPrice_30'] = df['BidClose'].rolling(window=21).std().shift(1)
    df_new['std_BidPrice_90'] = df['BidClose'].rolling(window=63).std().shift(1)
    df_new['std_BidPrice_365'] = df['BidClose'].rolling(window=252).std().shift(1)    
    
    df_new['std_AskPrice_5'] = df['AskClose'].rolling(window=5).std().shift(1)
    df_new['std_AskPrice_30'] = df['AskClose'].rolling(window=21).std().shift(1)
    df_new['std_AskPrice_90'] = df['AskClose'].rolling(window=63).std().shift(1)
    df_new['std_AskPrice_365'] = df['AskClose'].rolling(window=252).std().shift(1)
    
    #standard deviation of ratios
    df_new['std_RatioBid_5_30'] = df_new['std_BidPrice_5'] / df_new['std_BidPrice_30']
    df_new['std_RatioBid_5_90'] = df_new['std_BidPrice_5'] / df_new['std_BidPrice_90']
    df_new['std_RatioBid_5_365'] = df_new['std_BidPrice_5'] / df_new['std_BidPrice_365']
    df_new['std_RatioBid_30_90'] = df_new['std_BidPrice_30'] / df_new['std_BidPrice_90']
    df_new['std_RatioBid_30_365'] = df_new['std_BidPrice_30'] / df_new['std_BidPrice_365']
    df_new['std_RatioBid_90_365'] = df_new['std_BidPrice_90'] / df_new['std_BidPrice_365']

    df_new['std_RatioAsk_5_30'] = df_new['std_AskPrice_5'] / df_new['std_AskPrice_30']
    df_new['std_RatioAsk_5_90'] = df_new['std_AskPrice_5'] / df_new['std_AskPrice_90']
    df_new['std_RatioAsk_5_365'] = df_new['std_AskPrice_5'] / df_new['std_AskPrice_365']
    df_new['std_RatioAsk_30_90'] = df_new['std_AskPrice_30'] / df_new['std_AskPrice_90']
    df_new['std_RatioAsk_30_365'] = df_new['std_AskPrice_30'] / df_new['std_AskPrice_365']
    df_new['std_RatioAsk_90_365'] = df_new['std_AskPrice_90'] / df_new['std_AskPrice_365']

    #return for Bid and Ask
    df_new['BidReturn_1'] = ((df['BidClose'] - df['BidClose'].shift(1)) / df['BidClose'].shift(1)).shift(1)
    df_new['BidReturn_5'] = ((df['BidClose'] - df['BidClose'].shift(5)) / df['BidClose'].shift(5)).shift(1)
    df_new['BidReturn_30'] = ((df['BidClose'] - df['BidClose'].shift(21)) / df['BidClose'].shift(21)).shift(1)
    df_new['BidReturn_90'] = ((df['BidClose'] - df['BidClose'].shift(63)) / df['BidClose'].shift(63)).shift(1)                                                
    df_new['BidReturn_365'] = ((df['BidClose'] - df['BidClose'].shift(252)) / df['BidClose'].shift(252)).shift(1)
    
    df_new['AskReturn_1'] = ((df['AskClose'] - df['AskClose'].shift(1)) / df['AskClose'].shift(1)).shift(1)
    df_new['AskReturn_5'] = ((df['AskClose'] - df['AskClose'].shift(5)) / df['AskClose'].shift(5)).shift(1)
    df_new['AskReturn_30'] = ((df['AskClose'] - df['AskClose'].shift(21)) / df['AskClose'].shift(21)).shift(1)
    df_new['AskReturn_90'] = ((df['AskClose'] - df['AskClose'].shift(63)) / df['AskClose'].shift(63)).shift(1)                                                
    df_new['AskReturn_365'] = ((df['AskClose'] - df['AskClose'].shift(252)) / df['AskClose'].shift(252)).shift(1)
    
    #average of return
    df_new['moving_BidAvg_5'] = df_new['BidReturn_1'].rolling(window=5).mean()
    df_new['moving_BidAvg_30'] = df_new['BidReturn_1'].rolling(window=21).mean()
    df_new['moving_BidAvg_30'] = df_new['BidReturn_1'].rolling(window=63).mean()
    df_new['moving_BidAvg_365'] = df_new['BidReturn_1'].rolling(window=252).mean()
    
    #spread generation
    df_new['Spread_Close'] = df['AskClose'] - df['BidClose']
    df_new['Spread_Open'] = df['AskOpen'] - df['BidOpen']
    
    #target
    df_new['BidClose'] = df['BidClose']
    df_new = df_new.dropna(axis=0)
    return df_new

datatraining = generate_features(eurusd)
    
type(datatraining)
    
#Create dataset for X,y 
def create_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
    
    # Scale and create datasets
target_index = datatraining.columns.tolist().index('BidClose')
high_index = datatraining.columns.tolist().index('BidHigh_1')
low_index = datatraining.columns.tolist().index('BidLow_1')
dataset = datatraining.values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create y_scaler to inverse it later
y_scaler = MinMaxScaler(feature_range=(0, 1))
t_y = datatraining['BidClose'].values.astype('float32')
t_y = np.reshape(t_y, (-1, 1))
y_scaler = y_scaler.fit(t_y)
    
# Set look_back to 5 which is 5 hours ()
X, y = create_dataset(dataset, look_back=1)
y = y[:,target_index]

# Set training data size
# We have a large enough dataset. So divid into 98% training / 1%  development / 1% test sets
train_size = int(len(X) * 0.99)
trainX = X[:train_size]
trainY = y[:train_size]
testX = X[train_size:]
testY = y[train_size:]

# create the LSTM network
model = Sequential()
model.add(LSTM(20, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(20, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(4, return_sequences=False))
model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mae', 'mse'])
print(model.summary())

#fitting the model
history = model.fit(trainX, trainY, epochs=200, batch_size=64, verbose=0, validation_split=0.1)
val_loss = model.evaluate(testX, testY) #mae and mse

#Plotting results
epoch = len(history.history['val_loss'])
plt.figure(figsize=(40,10))
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_loss'])
plt.title(val_loss)
plt.ylabel(val_loss)
plt.show()

#Visually
pred = model.predict(testX)

predictions = pd.DataFrame()
predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
predictions['actual'] = testY
predictions = predictions.astype(float)

predictions.plot(figsize=(20,10))
plt.show()

predictions['diff'] = predictions['predicted'] - predictions['actual']
plt.figure(figsize=(10,10))
sns.distplot(predictions['diff']);
plt.title('Distribution of differences between actual and prediction')
plt.show()

print("MSE : ", mean_squared_error(predictions['predicted'].values, predictions['actual'].values))
print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['actual'].values))
predictions['diff'].describe()
