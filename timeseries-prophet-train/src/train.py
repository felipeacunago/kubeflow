import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import seaborn as sns
# import matplotlib.pyplot as plt
from fbprophet import Prophet
# from sklearn.metrics import mean_squared_error, mean_absolute_error
from google.cloud import storage
import argparse
import pickle
import os
import glob
import re
import sys
import traceback
from retry import retry

# python train.py --dataset-path ../test-data/example_yosemite_temps.csv --changepoint-prior-scale 0.01 --model-output-path ./models/
parser = argparse.ArgumentParser(description='Docker that trains using prophet')
parser.add_argument('--dataset-path', type=str, help='Path of the file to load')
parser.add_argument('--changepoint-prior-scale', type=float, help='Change point prior scale')
parser.add_argument('--model-output-path', type=str, help='Model output path')
args = parser.parse_args()

# rename input variables
dataset_path = args.dataset_path
changepoint_prior_scale = args.changepoint_prior_scale
model_output_path = args.model_output_path

# times the retry method will try to do its job
retry_times = 3

@retry(Exception, tries=retry_times, delay=0)
def train(model_name, dataset_full_path, output_path):
    print("Loading dataset: {}".format(dataset_full_path))
    if 'gs://' in dataset_full_path:
        filepaths = [f'{dataset_full_path}/{filepath}' for filepath in list_gcs_files(dataset_full_path)]
        data = pd.concat([pd.read_csv(f) for f in filepaths], ignore_index = True)
    else:
        data = pd.concat([pd.read_csv(f) for f in glob.glob(f'{dataset_full_path}/*.csv')], ignore_index = True)

    # creates Prophet object and fits model with data loaded from previous step
    model = Prophet(changepoint_prior_scale=args.changepoint_prior_scale)
    model.fit(data)

    # generates the path where the model will be saved as a pickle file
    model_path = os.path.join(output_path)

    # save it using google cloud storage if the outpath starts with gs://, otherwise it's saved at local dir
    if 'gs://' in dataset_full_path:
        with open('local.pickle', "wb") as f:
            bucket, _, prefix = get_bucket(model_path)

            pickle.dump(model, f)
            blob = bucket.blob(f'{prefix}{model_name}/{model_name}.pickle')
            blob.upload_from_filename(filename='local.pickle')
    else:
        model_full_path = os.path.join(model_path,f'{model_name}.pickle')
        with open(model_full_path, "wb") as f:
            pickle.dump(model, f)
    print("Model saved")

def create_dir(directory_full_path):
    if not os.path.exists(directory_full_path) and not 'gs' in directory_full_path:
        os.makedirs(directory_full_path)

def print_error(ex):
    # Get current system exception
    ex_type, ex_value, ex_traceback = sys.exc_info()

    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(ex_traceback)

    # Format stacktrace
    stack_trace = list()

    for trace in trace_back:
        stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

    print("Exception type : %s " % ex_type.__name__)
    print("Exception message : %s" %ex_value)
    print("Stack trace : %s" %stack_trace)

def get_bucket(path):
    """
        Returns bucket, bucket_name and prefix of path
    """
    from google.cloud import storage
    storage_client = storage.Client()
    bucket_name, prefix = re.search(r'gs:\/\/([\w.-]+)\/(.+)', path).group(1), re.search(r'gs:\/\/([\w.-]+)\/(.+)', path).group(2)
    bucket = storage_client.get_bucket(bucket_name)

    return bucket, bucket_name, prefix

def list_directories(dataset_path):
    if 'gs://' in dataset_path:
        from google.cloud import storage
        storage_client = storage.Client()
        bucket_name, prefix = re.search(r'gs:\/\/([\w.-]+)\/(.+)', dataset_path).group(1), re.search(r'gs:\/\/([\w.-]+)\/(.+)', dataset_path).group(2)
        bucket = storage_client.get_bucket(bucket_name)

        delimiter='/'

        blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

        list(blobs)
        directories = [(lambda x: x.replace(prefix,'').replace(delimiter,''))(x) for x in list(blobs.prefixes)]

        print('Directories found:')
        for directory in directories:
            print(directory)

    else:
        directories = next(os.walk(dataset_path))[1]

    return directories

def list_gcs_files(path):
    """
        This function returns the filenames -without full path- from selected path as a list
    """
    from google.cloud import storage
    storage_client = storage.Client()
    bucket_name, prefix = re.search(r'gs:\/\/([\w.-]+)\/(.+)', path).group(1), re.search(r'gs:\/\/([\w.-]+)\/(.+)', path).group(2)
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)

    file_names = [(lambda x: x.name.split('/')[-1])(x) for x in blobs]

    return file_names


# create output directory if it doesn't exist
create_dir(model_output_path)

# list only directories in the dataset path
directories = list_directories(dataset_path)



# for every folder found in dataset path it tries to generate a model
for model_name in directories:
    dataset_full_path = os.path.join(dataset_path, model_name)
    try:
        train(model_name, dataset_full_path, model_output_path)
    except Exception as error:
        print(f"[WARNING] Error training model {model_name} after {retry_times} times, you can read the error below")
        print_error(error)
