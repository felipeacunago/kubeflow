# importing pandas module  
import pandas as pd
import argparse
import os

# python task.py --dataset-path https://media.geeksforgeeks.org/wp-content/uploads/nba.csv --output-path ./output/ --split-column Team --ds-column Name --y-column Age --minimum-length 10
parser = argparse.ArgumentParser(description='Docker that trains using prophet')
parser.add_argument('--dataset-path', type=str, help='Path of the file to load')
parser.add_argument('--output-path', type=str, help='Path where the split will be generated')
parser.add_argument('--split-column', type=str, help='Split column name')
parser.add_argument('--ds-column', type=str, help='DS column name')
parser.add_argument('--y-column', type=str, help='y column name')
parser.add_argument('--minimum-length', type=int, default=0, help='Minimum amount of rows to be considered in the output (default=0)')
parser.add_argument('--order-ds', type=str, default='none', help='Order ds column output. Options: asc, desc, none')
parser.add_argument('--training-date', type=str, help='Maximum date to consider from training data')
args = parser.parse_args()

# rename input variables
dataset_path = args.dataset_path
output_path = args.output_path
split_column = args.split_column
ds_column = args.ds_column
y_column = args.y_column
min_length = args.minimum_length
training_date = args.training_date

# reading csv file from url  
df = pd.read_csv(dataset_path)

df = df.rename(columns={ds_column: 'ds', y_column: 'y', split_column: 'split_col'})
df = df[['ds','y','split_col']]

print(args)
print(df.head())

if training_date:
    df = df[(df['ds'] <= training_date)]

df_arrays = [x for _, x in df.groupby(df['split_col'])]

for data in df_arrays:
    if len(data.index) >= min_length:
        dataset_name = str(data.iloc[0]['split_col']) # make sure that is used as a string
        path = os.path.join(output_path,dataset_name)
        print(path)

        if not os.path.exists(path) and not 'gs://' in output_path:
            os.makedirs(path)

        if args.order_ds != 'none':
            if args.order_ds == 'asc':
                data = data.sort_values(by='ds', ascending=True)
            elif args.order_ds == 'desc':
                data = data.sort_values(by='ds', ascending=False)
            else:
                raise Exception('{} is not a correct argument for --order-ds command'.format(args.order_ds))

        data.to_csv(os.path.join(path,'data.csv'),index=False)