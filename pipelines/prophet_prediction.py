"""
Kubeflow Pipelines for timeseries prediction using fbprophet
Run this script to compile pipeline
"""


import kfp.dsl as dsl
import kfp.components as comp
import kfp.gcp as gcp
import json

bigquery_query_op = comp.load_component_from_url(
    'https://raw.githubusercontent.com/kubeflow/pipelines/e598176c02f45371336ccaa819409e8ec83743df/components/gcp/bigquery/query/component.yaml')

ranker_op = comp.load_component_from_url(
    'https://raw.githubusercontent.com/membrilloski/kubeflow/master/ranker/component.yaml'
)

@dsl.pipeline(
  name='Prophet',
  description='A pipeline to train and serve the MNIST example.'
)
def prophet_pipeline(dataset_query='',
                   dictionary_query='',
                   val_dataset_query='',
                   minimum_length=10,
                   training_date='2019-09-12',
                   changepoint_prior_scale=0.01,
                   evaluation_date='',
                   evaluation_maximum_distance=1
                    ):

    """
      Pipeline with three stages:
        1. Query data from Bigquery and save to storage
        2. Generate a timeseries prediction for every distinct element in split_column
        3. Generate a ranking based on prediction data and compare it to a ranking made with real data from that date
    """

    rank_path_names=['']
    ranking_factors=[]
    project_id=''
    split_column=''
    gcs_root='gs://bucket/folder'
    prediction_y='y'
    predict_periods=10
    dataset_location='US'
    ds_column='ds'
    predict_freq='D'
    order_ds = 'asc'

    original_dataset_path='{}/input/dataset.csv'.format(gcs_root)
    val_dataset_path='{}/input/val_dataset.csv'.format(gcs_root)
    val_split_output_path='{}/output/validation/'.format(gcs_root)
    preprocess_output_path='{}/output/training/'.format(gcs_root)
    model_output_path='{}/models/'.format(gcs_root)
    predictions_path='{}/predictions/'.format(gcs_root)
    dictionary_file_path = '{}/dictionary/dictionary.csv'.format(gcs_root)
    prophet_rank_output_path= '{}/rankings/prediction/'.format(gcs_root)
    validation_rank_output='{}/rankings/validation/'.format(gcs_root)

    

    dataset_query_op = bigquery_query_op(
        query=dataset_query, 
        project_id=project_id,
        output_gcs_path=original_dataset_path, 
        dataset_location=dataset_location, 
        job_config='').apply(gcp.use_gcp_secret('user-gcp-sa'))

    preprocess = dsl.ContainerOp(
        name='preprocess-split',
        image='docker.io/felipeacunago/dataset-preprocess:latest',
        arguments=[
            "task.py",
            "--dataset-path", dataset_query_op.outputs['output_gcs_path'],
            "--output-path", preprocess_output_path,
            "--split-column", split_column,
            "--ds-column", ds_column,
            "--y-column", prediction_y,
            "--minimum-length", minimum_length,
            "--order-ds", 'asc',
            "--training-date", training_date
            ]
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

    timeseries_prophet_train = dsl.ContainerOp(
        name='prophet-train',
        image='docker.io/felipeacunago/timeseries-prophet-train:latest',
        arguments=[
            "train.py",
            "--dataset-path", preprocess_output_path,
            "--changepoint-prior-scale", changepoint_prior_scale,
            "--model-output-path", model_output_path
            ]
    ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(preprocess)

    timeseries_prophet_predict = dsl.ContainerOp(
        name='prophet-predict',
        image='docker.io/felipeacunago/timeseries-prophet-predict:latest',
        arguments=[
            "predict.py",
            "--predict-periods", predict_periods,
            "--predict-freq", predict_freq,
            "--model-path", model_output_path,
            "--predictions-path", predictions_path
            ]
    ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(timeseries_prophet_train)

    dataset_query_op = bigquery_query_op(
        query=dictionary_query, 
        project_id=project_id,
        output_gcs_path=dictionary_file_path, 
        dataset_location=dataset_location, 
        job_config='').apply(gcp.use_gcp_secret('user-gcp-sa'))

    prophet_rank = ranker_op(
        input_path=predictions_path,
        input_path_names=rank_path_names,
        ranking_factors=ranking_factors,
        input_dictionary=dataset_query_op.outputs['output_gcs_path'],
        training_date=training_date,
        ranking_output_path=prophet_rank_output_path,
        prediction_periods=predict_periods
    ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(timeseries_prophet_predict)

    val_query_op = bigquery_query_op(
        query=val_dataset_query, 
        project_id=project_id,
        output_gcs_path=val_dataset_path, 
        dataset_location=dataset_location, 
        job_config='').apply(gcp.use_gcp_secret('user-gcp-sa'))

    val_preprocess = dsl.ContainerOp(
        name='validation-preprocess-split',
        image='docker.io/felipeacunago/dataset-preprocess:latest',
        arguments=[
            "task.py",
            "--dataset-path", val_query_op.outputs['output_gcs_path'],
            "--output-path", val_split_output_path,
            "--split-column", split_column,
            "--ds-column", ds_column,
            "--y-column", prediction_y,
            "--minimum-length", 1,
            "--order-ds", order_ds,
            "--training-date", evaluation_date
            ]
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

    val_rank = ranker_op(
        input_path=val_split_output_path,
        input_path_names=rank_path_names,
        ranking_factors=ranking_factors,
        input_dictionary=dataset_query_op.outputs['output_gcs_path'],
        training_date=evaluation_date,
        ranking_output_path=validation_rank_output,
        prediction_periods=predict_periods
    ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(val_preprocess)

    rank_evaluation = dsl.ContainerOp(
        name='rank-evaluation',
        image='docker.io/felipeacunago/ranker-eval:latest',
        arguments=[
            "task.py",
            "--predicted-ranking-path", prophet_rank.outputs['ranking_output_path_file'],
            "--real-ranking-path", val_rank.outputs['ranking_output_path_file'],
            "--eval-date", training_date,
            "--maximum-distance", evaluation_maximum_distance
            ]
    ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(prophet_rank)

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(prophet_pipeline, __file__ + '.tar.gz')