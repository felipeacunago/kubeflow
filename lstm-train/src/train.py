import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras import optimizers

class LSTM_train:

    def __init__(
            self,
            data_has_lag=False, # if the data already contains the lag (t-n)
            n_features = 2,
            timesteps = 2,
            lstm_type='LSTM'
        ):

        self.data_has_lag = data_has_lag
        self.n_features = 0
        self.n_features = n_features
        self.timesteps = timesteps

        self.build_model(lstm_type)

    def build_model(self,lstm_type, n_features = 2):
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(timesteps, self.n_features)))
        self.model.add(Dense(1))

    def set_dataset(self,dataset):
        if self.data_has_lag:
            X, Y = dataset.values
        else:
            X, Y = self.series_lag(dataset, timesteps=self.timesteps, target_col='target', debug=True)

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        X = X.reshape((X.shape[0], self.timesteps, self.n_features))

        print(X)
        self.X, self.Y = X, Y

    def train(self, n_features = 2):
        optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

        self.model.compile(optimizer=optimizer, loss='mse')
        self.model.fit(self.X, self.Y, epochs=1000, verbose=1)

    def series_lag(self, data, timesteps=2, n_out=1, dropnan=True, target_col='target', debug=False):
        inputs = data.drop([target_col], axis=1)
        Y = data[target_col].shift(-timesteps)

        n_vars = 1 if type(inputs) is list else inputs.shape[1]
        columns = list(inputs.columns.values)

        cols, names = list(), list()

        for i in range(timesteps, 0, -1):
            cols.append(inputs.shift(i))
            names += [('%s%d(t-%d)' % (columns[j],j+1, i)) for j in range(n_vars)]

        X = pd.concat(cols, axis=1)
        names.sort()
        X.columns = names
        # drop rows with NaN values
        if dropnan:
            X.dropna(inplace=True)
        
        Y.dropna(inplace=True)

        if debug:
            print(X)
            print(Y)

        return X.values, Y.values

    def predict(self,X):
        return self.model.predict(X)

    

data = pd.DataFrame({'A': np.linspace(1, 10, num=10), 'B': np.linspace(1, 10, num=10)*2})

data['target'] = data['A']*2 + data['B']*5 + 1

timesteps = 2

lstm = LSTM_train(timesteps=timesteps)
lstm.set_dataset(data)
lstm.train()



# define model
test = np.random.rand(1,timesteps,2)
print(test)
yhat = lstm.predict(test)

print(yhat)