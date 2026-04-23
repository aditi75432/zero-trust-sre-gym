#!/bin/bash

set -e

echo "[START] Zero Trust SRE Gym — Enterprise Security RL Environment"
echo "[START] Launching microservice layer..."

pip install flask --quiet 2>/dev/null || true

# Start services with nohup so HF doesn't kill them
nohup python frontend_service.py > frontend.log 2>&1 &
FRONTEND_PID=$!

nohup python payment_service.py > payment.log 2>&1 &
PAYMENT_PID=$!

nohup python hr_db_service.py > hr.log 2>&1 &
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
echo "[START] Starting Zero Trust Gym API on port 7860..."

uvicorn server.app:app --host 0.0.0.0 --port 7860

# Keep processes alive
wait $FRONTEND_PID $PAYMENT_PID $HRDB_PID