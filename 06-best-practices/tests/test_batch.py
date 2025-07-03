
import batch
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)



def test_prepare_data():
    data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    actual = batch.prepare_data(df, ['PULocationID', 'DOLocationID'])
    expected = pd.DataFrame({
        'PULocationID': ['-1', '1'],
        'DOLocationID': ['-1', '1'],
        'tpep_pickup_datetime': [dt(1, 1), dt(1, 2)],
        'tpep_dropoff_datetime': [dt(1, 10), dt(1, 10)],
        'duration': [9.0, 8.0]
    })
    assert actual.equals(expected), f"Expected:\n{expected}\n\nActual:\n{actual}"

