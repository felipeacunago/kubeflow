from flask import Flask, request
import numpy as np
import pickle
import json
import pandas as pd
import os
import gcsfs
import environs

env = environs.Env()
env.read_env()

# enviroment variables
PROJECT_NAME = env.str("PROJECT_NAME")
MODELS_PATH = env.str("MODELS_PATH")
DICTIONARY_PATH = env.str("DICTIONARY_PATH")

app = Flask(__name__)


@app.route('/api/model/', methods=['POST'])
def model_request():
    """
    When sending POST with params model_name, periods and freq
    it loads the model, generates a prediction and send it back
    """
    print(request)
    data = request.get_json()

    print(data)
    model_name = data.get('model',None)
    periods = data.get('future_periods',None)
    freq = data.get('freq',None)
    date = data.get('date',None)

    prediction = predict(model_name, periods, freq, date).to_json()

    return prediction

@app.route('/api/grouping/',methods=['POST'])
def grouping_request():
    data = request.get_json()

    grouping = data.get('grouping_name',None)

    print(grouping)
    
    # get all the elements that are grouped by it
    name_list = get_list_dataframe(grouping)

    name_list['name'] = name_list['name'].astype(str)
    
    return name_list.to_json(orient='records')

@app.route('/api/grouping/list/', methods=['POST'])
def list_grouping():
    df = pd.read_csv(DICTIONARY_PATH)
    unique_grouping_names = list(df.grouping_name.unique())

    return json.dumps(unique_grouping_names)


def get_list_dataframe(grouping_name):
    """
    Reads dictionary dataframe and filters it by grouping_name 
    """
    df = pd.read_csv(DICTIONARY_PATH)
    df = df[(df['grouping_name'] == grouping_name)]

    return df

def predict(model_name, periods=None, freq=None, date=None):
    """
    Outputs a prediction from given pickle model
    """
    model_full_path = f'{MODELS_PATH}/{model_name}'

    if not model_name:
        return None

    fs = gcsfs.GCSFileSystem(project=PROJECT_NAME)
    with fs.open(f'{model_full_path}/{model_name}.pickle', 'rb') as f:
        model = pickle.load(f)

    if date:
        future = pd.DataFrame({'ds': [pd.to_datetime(date)]})
    elif not date and periods and freq:
        future = model.make_future_dataframe(periods=periods,freq=freq)
    else:
        return None

    prediction = model.predict(future)

    return prediction

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')