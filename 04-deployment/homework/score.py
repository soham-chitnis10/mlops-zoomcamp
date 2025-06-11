import pickle
import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

categorical = ['PULocationID', 'DOLocationID']

# @task(retries=3, retry_delay_seconds=2, log_prints=True)
def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

# @task(retries=3, retry_delay_seconds=2, log_prints=True)
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame and filter it"""
    print(f'Reading data from {filename}')
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# @task(retries=3, retry_delay_seconds=2, log_prints=True)
def predict(df: pd.DataFrame, dv: DictVectorizer, model: LinearRegression):
    """Score the data using the model"""
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(f'Mean prediction: {np.mean(y_pred):.2f}')
    print(f'Standard deviation of predictions: {np.std(y_pred):.2f}')

    return y_pred

# @task(retries=3, retry_delay_seconds=2, log_prints=True)
def save_results(df: pd.DataFrame, y_pred: np.ndarray, output_file: str, year: int, month: int ):
    """Save the predictions to a CSV file"""
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(f'Results saved to {output_file}')

def run(args):
    """Main function to run the scoring process"""
    year = args.year
    month = args.month
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'predictions_{year:04d}-{month:02d}.parquet'

    dv, model = load_model()
    df = read_data(input_file)
    y_pred = predict(df, dv, model)
    save_results(df, y_pred, output_file, year, month)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Score taxi trip data')
    parser.add_argument('--year', type=int, default=2023, help='Year of the data')
    parser.add_argument('--month', type=int, default=3, help='Month of the data')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
    print("Scoring completed successfully.")
