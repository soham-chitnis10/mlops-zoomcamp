#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os

S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', '')
options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}


def prepare_data(df, categorical):
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def save_data(df, output_file):
    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )
    print('data saved to', output_file)

def main(year, month):
    input_file = os.getenv('INPUT_FILE_PATTERN', 's3://nyc-duration/yellow_tripdata_{year:04d}-{month:02d}.parquet').format(year=year, month=month)
    output_file = os.getenv('OUTPUT_FILE_PATTERN', 's3://nyc-duration/yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet').format(year=year, month=month)
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    categorical = ['PULocationID', 'DOLocationID']
    print('input file:', input_file)
    print('output file:', output_file)
    df = pd.read_parquet(input_file, storage_options=options)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred


    save_data(df_result, output_file)
if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)