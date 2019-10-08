import pandas as pd
import argparse
import os
import datetime
from sklearn import preprocessing
from pathlib import Path

# python task.py --input-dictionary ../data/dictionary_short.csv --input-path ../../predictions --input-path-names clicks --training-date 2019/09/12 --prediction-periods 0 --ranking-output-path ../../ranks/predictions
parser = argparse.ArgumentParser(description='Docker that trains using prophet')
parser.add_argument('--input-path', nargs='+', help='Paths of inputs for certain id to rank')
parser.add_argument('--input-path-names', nargs='+', help='Names of the input values')
parser.add_argument('--ranking-factors', nargs='+', help='Importance factors for ranking (the sum is 1)')
parser.add_argument('--input-dictionary', type=str, help='Grouping dictionary path')
parser.add_argument('--training-date', type=str, help='Date used on training')
parser.add_argument('--prediction-periods', type=int, help='Amount of days starting from training date to generate data')
parser.add_argument('--ranking-output-path', type=str, help='Path to output the ranking')
parser.add_argument('--ranking-output-path-file', type=str, help='Path of the file where output path will should be written')
args = parser.parse_args()

# rename input variables
input_path = args.input_path
input_path_names = args.input_path_names
ranking_factors = args.ranking_factors
input_dictionary = args.input_dictionary
training_date = args.training_date
ranking_output_path = args.ranking_output_path
prediction_periods = args.prediction_periods

def normalize_dataframe(dataframe):
    """
    Takes df and normalize it by column maximum value using sklearn MinMaxScaler
    A = 
    A     B   C
    1000  10  0.5
    765   5   0.35
    800   7   0.09
    normalize_dataframe(A)
    >> 
    A     B    C
    1     1    1
    0.765 0.5  0.7
    0.8   0.7  0.18

    Parameters
    ----------
    dataframe : pandas dataframe
        Dataframe to normalize
    """
    # keep index and columns names
    index = dataframe.reset_index()['index']
    columns = dataframe.columns

    # keep data related to useful information needed for the prediction and ranking
    x = dataframe.reset_index(drop=True).values

    # scales data from 0 to 1
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    # sets the index again with the saved index column
    df = pd.DataFrame(x_scaled, columns=columns).set_index(index)

    return df


if len(input_path) != len(ranking_factors):
    raise Exception(f'Input paths arguments {input_path} and ranking factors arguments {ranking_factors} must have the same length ({len(input_path)}!={len(ranking_factors)})')

# read dictionary from file
dictionary_df = pd.read_csv(input_dictionary)

# dictionary separated by column grouping_name
dictionary_array_df = [x for _, x in dictionary_df.groupby(dictionary_df['grouping_name'])]

i=0

# generate an empty dataframe with the column names
empty_df = pd.DataFrame(columns=['ds'])

# start date used to generate date array for initialization
historic_periods = 10
total_data_periods_in_days = historic_periods + prediction_periods + 1
start_date = (datetime.datetime.strptime(training_date, '%Y-%m-%d') - datetime.timedelta(days=historic_periods)).strftime('%Y/%m/%d')
data_ranges = pd.date_range(start=start_date, periods=total_data_periods_in_days, freq='d')
empty_df['ds'] = data_ranges

for dictionary in dictionary_array_df:
    grouping_name = dictionary.iloc[0]['grouping_name']
    print(f"Grouping -> {grouping_name}")
    output_df = empty_df
    for index, row in dictionary.iterrows():
        model_name = row['name']
        model_file_path = os.path.join(input_path[0],f'{model_name}/data.csv')
        print(f"Reading data from model {model_name} at {model_file_path}")
        try:
            model_df = pd.read_csv(model_file_path)

            # if it doesnt find yhat, try with y
            try:
                model_df = model_df[['ds','yhat']]
                model_df = model_df.rename(columns={'yhat':model_name})
            except KeyError:
                model_df = model_df[['ds','y']]
                model_df = model_df.rename(columns={'y':model_name})
            model_df['ds'] = pd.to_datetime(model_df['ds'])
            output_df = pd.merge(output_df,model_df, on=["ds"], how='left')
            
        except FileNotFoundError:
            print(f'[WARNING] File not found: The file for the model {model_name} was not found, it will be skipped')
    output_df = output_df.set_index('ds')
    
    print("Generating output file")
    output_df = normalize_dataframe(output_df.T)
    if 'gs://' in ranking_output_path:
        output_full_path = f'{ranking_output_path}{grouping_name}.csv'
    else:
        output_full_path = os.path.join(f'{ranking_output_path}', f'{grouping_name}.csv')
    print(f"Saving file to {output_full_path}")
    output_df.to_csv(output_full_path)

# Writing ranking_output_path to a file so that it will be passed to downstream tasks
Path(args.ranking_output_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.ranking_output_path_file).write_text(args.ranking_output_path)