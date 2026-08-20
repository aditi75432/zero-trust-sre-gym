#!/bin/bash

set -e

echo "[START] Zero Trust SRE Gym — Enterprise Security RL Environment"
echo "[START] Launching microservice layer..."

# Start services in background
python frontend_service.py &
FRONTEND_PID=$!
python payment_service.py &
PAYMENT_PID=$!
python hr_db_service.py &
HRDB_PID=$!

echo "[START] Waiting for microservices to initialise..."
sleep 8

echo "[START] Checking service health..."
for port in 5003 5004 5005; do
    for attempt in 1 2 3 4 5; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[START] Port ${port} ready."
            break
        fi
        echo "[WAIT] Port ${port} not ready yet..."
        sleep 1
    done
done

echo "[START] Microservice layer ready."

# Start FastAPI backend on port 8000 (internal)
echo "[START] Starting FastAPI backend on port 8000..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a moment for FastAPI to start
sleep 2

# Start Streamlit dashboard on the public port 7860
echo "[START] Starting Streamlit dashboard on port 7860..."
streamlit run dashboard.py --server.port 7860 --server.address 0.0.0.0

# Keep all processes alive
wait $FRONTEND_PID $PAYMENT_PID $HRDB_PID $API_PID