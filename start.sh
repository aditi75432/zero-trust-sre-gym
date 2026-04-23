#!/bin/bash

set -e

echo "[START] Zero Trust SRE Gym — Enterprise Security RL Environment"
echo "[START] Launching microservice layer..."

pip install flask --quiet 2>/dev/null || true

python frontend_service.py &
FRONTEND_PID=$!

python payment_service.py &
PAYMENT_PID=$!

python hr_db_service.py &
HRDB_PID=$!

echo "[START] Waiting for microservices to initialise..."
sleep 3

for port in 5003 5004 5005; do
    for attempt in 1 2 3 4 5; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[START] Port ${port} ready."
            break
        fi
        sleep 1
    done
done

echo "[START] Microservice layer ready."
echo "[START] Starting Zero Trust Gym API on port 7860..."

uvicorn server.app:app --host 0.0.0.0 --port 7860

wait $FRONTEND_PID $PAYMENT_PID $HRDB_PID