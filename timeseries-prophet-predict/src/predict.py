import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet
import argparse
import pickle
import os
import re

# python predict.py --predict-periods 10 --model-path ../../timeseries-prophet-train/models --predictions-path ../../predictions
parser = argparse.ArgumentParser(description='Docker that trains using prophet')
parser.add_argument('--predict-periods', type=int, help='Periods to predict')
parser.add_argument('--predict-freq', type=str, default='D', help='Frequency type argument to predict')
parser.add_argument('--model-path', type=str, help='Model path')
parser.add_argument('--predictions-path', type=str, help='Path where the predictions will be exported')
args = parser.parse_args()

# rename inputs
predict_periods = args.predict_periods
predict_freq = args.predict_freq
model_path = args.model_path
predictions_path = args.predictions_path

def get_bucket(path):
    """
        Returns bucket, bucket_name and prefix of path
    """
    from google.cloud import storage
    storage_client = storage.Client()
    bucket_name, prefix = re.search(r'gs:\/\/([\w.-]+)\/(.+)', path).group(1), re.search(r'gs:\/\/([\w.-]+)\/(.+)', path).group(2)
    bucket = storage_client.get_bucket(bucket_name)

    return bucket, bucket_name, prefix

def list_directories(path):
    """
        Returns a list of directories names inside path
    """
    if 'gs://' in path:
        from google.cloud import storage
        storage_client = storage.Client()
        bucket_name, prefix = re.search(r'gs:\/\/([\w.-]+)\/(.+)', path).group(1), re.search(r'gs:\/\/([\w.-]+)\/(.+)', path).group(2)
        bucket = storage_client.get_bucket(bucket_name)

        delimiter='/'

        blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

        list(blobs)
        directories = [(lambda x: x.replace(prefix,'').replace(delimiter,''))(x) for x in list(blobs.prefixes)]

        print('Directories found:')
        for directory in directories:
            print(directory)

    else:
        directories = next(os.walk(path))[1]

    return directories

def load_model(model_full_path):
    temp_filepath = './temp/model.pickle'
    if 'gs://' in model_full_path:
        bucket, _, prefix = get_bucket(model_full_path)
        print(f"Downloading file from {prefix}")
        blob = bucket.blob(prefix)
        blob.download_to_filename(temp_filepath)
    else:
        from shutil import copyfile

        copyfile(model_full_path, temp_filepath)
    with open(temp_filepath, 'rb') as f:
        model = pickle.load(f)

    return model


def predict(model_full_path,predict_periods,predict_freq):
    """
        Using the model located at model_full_path it predicts using predict_periods, and predict_freq
    """
    # read the Prophet model object
    print(f"Loading model file from {model_full_path}")
    model = load_model(model_full_path)
    print("Model loaded sucessfully")

    # make predictions dataset
    print("Generating prediction dataset from current dataset")
    future = model.make_future_dataframe(periods=predict_periods,freq=predict_freq)

    # there are duplicates for some reason, going to drop them meanwhile
    future.drop_duplicates(subset='ds', keep = 'first', inplace=True)

    # make prediction
    print("Predicting the future")
    forecast = model.predict(future)

    return forecast

def upload_folder_gcs(source_folder,gcs_output_path):
    """
        Goes through folder looking for files called data.csv
    """
    files = os.listdir(source_folder)
    bucket, _, prefix = get_bucket(gcs_output_path)

    for filename in files:
        gcs_file_path = f'{prefix}{filename}/data.csv'
        source_file_path = f'{source_folder}{filename}/data.csv'
        upload_file_gcs(source_file_path,gcs_file_path,bucket=bucket)

def upload_file_gcs(source_path,gcs_output_path, bucket=None):
    """
        Uploads file from source_path to gcs_output_path
        if the bucket is not given it tries to get access
    """
    if bucket == None:
        bucket, _, prefix = get_bucket(gcs_output_path)
    blob = bucket.blob(gcs_output_path)
    blob.upload_from_filename(source_path)

    print('File {} uploaded to {}.'.format(source_path,gcs_output_path))

# # create output directory if it doesn't exist
if not os.path.exists('./temp/'):
    os.makedirs('./temp/')
if not os.path.exists('./temp/predictions/'):
    os.makedirs('./temp/predictions/')

models = list_directories(model_path)

print("[LOG] Predictions started")
temp_folder = './temp/predictions/'
for model in models:
    model_full_path = f'{os.path.join(model_path,model)}/{model}.pickle'

    forecast = predict(model_full_path,predict_periods,predict_freq)
    print("Saving predictions to: {}".format(temp_folder))

    # output path deletes extension
    model_without_extension = model.split('.')[0]
    model_folder_path = f'./temp/predictions/{model_without_extension}'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    output_full_path = os.path.join(model_folder_path, 'data.csv')
    forecast.to_csv(output_full_path)
    print("Predictions Saved Locally")

if 'gs://' in predictions_path:
    upload_folder_gcs(temp_folder,predictions_path)
else:
    from shutil import copyfile

    copyfile(temp_folder, predictions_path)