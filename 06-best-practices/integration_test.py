import pandas as pd
from datetime import datetime
import os


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_save():
    S3_ENDPOINT_URL = 'http://localhost:4566'  # LocalStack S3 endpoint
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL  # LocalStack S3 endpoint
        }
    }
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
        ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    df.to_parquet(
        's3://nyc-duration/yellow_tripdata_2023-01.parquet',
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
)
    os.system('python batch.py 2023 1')

if __name__ == '__main__':
    test_save()
    print("Test completed successfully.")
    
