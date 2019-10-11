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
            lstm_type='LSTM',
            epochs=1000
        ):

        self.data_has_lag = data_has_lag
        self.n_features = n_features
        self.timesteps = timesteps
        self.epochs = epochs

        self.build_model(lstm_type)

    def build_model(self,lstm_type, n_features = 2):
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.timesteps, self.n_features)))
        self.model.add(Dense(1))

    def set_dataset(self,dataset, target_col='target'):
        if self.data_has_lag:
            X, Y = dataset.values
        else:
            X, Y = self.series_lag(dataset, timesteps=self.timesteps, target_col=target_col, debug=True)

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        X = X.reshape((X.shape[0], self.timesteps, self.n_features))

        print(X)
        self.X, self.Y = X, Y

    def train(self, n_features = 2, optimizer = None):
        if not optimizer:
            optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

        self.model.compile(optimizer=optimizer, loss='mse')
        self.model.fit(self.X, self.Y, epochs=1000, verbose=1)

    def series_lag(self, data, timesteps=1, n_out=1, dropnan=True, target_col='target', debug=False):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = data.shape[1]
        df = data.rename(columns={target_col: 'target'})

        columns = df.columns
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(timesteps, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # names.sort()
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s(t)' % (columns[j])) for j in range(n_vars)]
            else:
                names += [('%s(t+%d)' % (columns[j], i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        
        agg.columns = names

        cols_to_drop = [c for c in names if not 'target' in c.lower() and '(t)' in c.lower()]
        agg.drop(columns=cols_to_drop, inplace=True)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        X = agg.drop(['target(t)'], axis=1)
        Y = agg['target(t)']

        return X.values, Y.values

    def predict(self,X):
        return self.model.predict(X)
