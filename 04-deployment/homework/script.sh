#!/bin/bash

# Start the Prefect server
prefect server start --host 0.0.0.0 &

# Wait for the server to start

prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api

sleep 10

# Run the score script
python score.py --year 2023 --month 5