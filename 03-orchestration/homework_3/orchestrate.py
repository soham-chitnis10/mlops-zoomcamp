import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
from prefect import flow, task
from sklearn.linear_model import LinearRegression
from mlflow.tracking import MlflowClient


@task(retries=3, retry_delay_seconds=2, log_prints=True)
def read_data_unfiltered(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(filename)
    print(f'Total records: {len(df)}')
    return df

@task(retries=3, retry_delay_seconds=2, log_prints=True)
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    print(f'Total records after filtering: {len(df)}')

    return df

@task
def process_data(df) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    """Add features to the model"""

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    train_dicts = df[categorical+numerical].to_dict(orient="records")

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    y_train = df["duration"].values

    return X_train, y_train, dv


@task(retries=3, retry_delay_seconds=2, log_prints=True)
def train_model(
    X_train: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> LinearRegression:
    """Train the model"""
    with mlflow.start_run():
        mlflow.set_tag("model", "linear_regression")
        mlflow.set_tag("version", "1.0")

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save the model
        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.sklearn.log_model(model, "models")
    print(f"Model intercept: {model.intercept_}")
    return model

@task(retries=3, retry_delay_seconds=2, log_prints=True)
def register_model():
    client = MlflowClient("http://127.0.0.1:5000")
    run_id = client.search_runs(experiment_ids='1')[0].info.run_id
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/models",
        name='nyc-taxi-duration-predictor'
    )

@flow
def main_flow_new(
    file_path: str = "/workspaces/mlops-zoomcamp/03-orchestration/data/yellow_tripdata_2023-03.parquet",
) -> None:
    """The main training pipeline"""

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    # Load
    df_train = read_dataframe(file_path)
    X_train, y_train, dv = process_data(df_train)

    # Train
    model = train_model(X_train, y_train, dv)

    # Register
    register_model()


if __name__ == "__main__":
    main_flow_new()
