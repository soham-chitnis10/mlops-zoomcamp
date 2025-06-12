#!/bin/bash

# Start the Prefect server
prefect server start --host 0.0.0.0 &

# Wait for the server to start
sleep 10

prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api

# Run the score script
python score.py