import pandas as pd
import argparse
import os
import datetime
import re
import json

# logger
import logging
import logging.config

if os.path.exists('log.conf'):
    logging.config.fileConfig('log.conf')
else:
    logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s')
    logging.warning('log.conf file not found, basic configuration will be used')

# python task.py --predicted-ranking-path gcs-predicted-ranking-path --real-ranking-path gcs-real-ranking-path
parser = argparse.ArgumentParser(description='Docker that trains using prophet')
parser.add_argument('--predicted-ranking-path', type=str, help='Paths of the predicted ranking')
parser.add_argument('--real-ranking-path', type=str, help='Paths of the real ranking')
parser.add_argument('--eval-date', type=str, help='Date where the evaluation will be done')
parser.add_argument('--maximum-distance', type=int, default=0, help='Maximum distance of the prediction and real ranking to be considered as a valid prediction (default = 0 / Perfect match)')
parser.add_argument('--output', type=str, help='GCS evaluation output folder')
args = parser.parse_args()

# rename input variables
predicted_ranking_path = args.predicted_ranking_path
real_ranking_path = args.real_ranking_path
eval_date = args.eval_date
maximum_distance = args.maximum_distance
output = args.output

def list_gcs_files(path):
    """
    Returns the filenames -without full path- from selected path as a list

    Parameters
    ----------
    path : str 
        GCS path where the list will be done
    """
    from google.cloud import storage
    storage_client = storage.Client()
    bucket_name, prefix = re.search(r'gs:\/\/([\w.-]+)\/(.+)', path).group(1), re.search(r'gs:\/\/([\w.-]+)\/(.+)', path).group(2)
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)

    file_names = [(lambda x: x.name.split('/')[-1])(x) for x in blobs]

    return file_names

def generate_evaluation(real_ranking_path,predicted_ranking_path, eval_date):
    """
    Given two rankings it makes an evaluation on how good it was.
    It just calculates the distance between these two ranking for every object and counts how many of them are bigger
    than maximum_distance

    Parameters
    ----------
    real_ranking_path : str 
        Absolute gcs path to the real ranking file path
    predicted_ranking_path : str 
        Absolute gcs path to the predicted ranking file path
    eval_date : str (YYYY-MM-DD) 
        Date where the evaluation will be done so it only takes the ranking for given date
    """

    # generate rank
    real_ranking_df['rank'] = real_ranking_df[eval_date].rank(ascending=False)
    predicted_ranking_df['rank'] = predicted_ranking_df[eval_date].rank(ascending=False)

    comparison_df = pd.merge(real_ranking_df[['index','rank']],predicted_ranking_df[['index','rank']],on='index',suffixes=('_real','_pred'))
    comparison_df['dist'] = (comparison_df['rank_real']-comparison_df['rank_pred']).abs()

    return comparison_df

def top_k(comparison_df,k=1000, maximum_distance=0):
    """
    Generates the evaluation for the top k elements

    Parameters
    ----------
    comparison_df : Pandas Dataframe
        Comparison dataframe to get the top k evaluation
    k : int 
        Quantity of elements to consider for the evaluation
    maximum_distance : int 
        Maximum distance for the evaluation to consider it as a valid classification (it could be used in the future for ROC curve)
    """
    # crop data with real ranking > k
    cropped_df = comparison_df[(comparison_df['rank_real']<=k)]

    valid_pred_qty = len(cropped_df[(cropped_df['dist'] <= maximum_distance)].index)
    total_length = len(cropped_df.index)
    accuracy = valid_pred_qty/total_length*100
    minimum, maximum, avg_dist = cropped_df['dist'].min(), cropped_df['dist'].max(), cropped_df['dist'].abs().sum()/total_length

    return accuracy, minimum, maximum, avg_dist


# read the files
logging.info(f'Reading files from {real_ranking_path}')
rankings_to_eval = list_gcs_files(real_ranking_path)
logging.info(f'Found {len(rankings_to_eval)} files')

# initialize the output dataframe with the evaluation
eval_df = pd.DataFrame(columns=["name","accuracy", "minimum", "maximum", "avg_dist"])

for grouping_compare_name in rankings_to_eval:

    # paths to where the real and predicted ranking are located in gcs
    real_ranking_df = pd.read_csv(f'{real_ranking_path}{grouping_compare_name}')
    predicted_ranking_df = pd.read_csv(f'{predicted_ranking_path}{grouping_compare_name}')

    # calling the function to generate the comparison between both rankings
    comparison_df = generate_evaluation(real_ranking_df,predicted_ranking_df, eval_date)

    # get all the measurements for the given comparison and append it to the evaluation dataframe
    accuracy, minimum, maximum, avg_dist = top_k(comparison_df, k=20, maximum_distance=maximum_distance)
    row_dict = {'name': grouping_compare_name.replace('.csv',''), 'accuracy': accuracy, 'minimum': minimum, 'maximum': maximum, 'avg_dist': avg_dist}
    eval_df = eval_df.append(row_dict, ignore_index=True)

# pending to generate output as artifact
print(eval_df)
output_file = os.path.join(output,'table.csv')
eval_df.to_csv(output_file)
metadata = {
    'outputs' : [{
      'type': 'table',
      'storage': 'gcs',
      'format': 'csv',
      'header': list(eval_df.columns.values),
      'source': output_file
    }]
  }
print(metadata)
with open('/table.json', 'w') as f:
    json.dump(metadata, f)