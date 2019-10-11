from src.train import LSTM_train
import pandas as pd
import numpy as np

data = pd.DataFrame({'A': np.linspace(1, 10, num=1000)})

timesteps = 2
n_features = 1

lstm = LSTM_train(timesteps=timesteps, n_features=n_features)
lstm.set_dataset(data[['A']], target_col='A')
lstm.train()



# test series (must be 12)
test = np.array([[[1000],[1001]]])
print(test)
yhat = lstm.predict(test)

print(yhat)