#!/bin/bash
set -e

echo "[START] Zero Trust SRE Gym — Enterprise Security RL Environment"
echo "[START] Launching microservice layer..."

# Start the three Flask microservices in the background
python frontend_service.py &
FRONTEND_PID=$!
python payment_service.py &
PAYMENT_PID=$!
python hr_db_service.py &
HRDB_PID=$!

echo "[START] Waiting for microservices to initialise..."
sleep 8

# Function to check if a service port is responding
wait_for_port() {
    local port=$1
    local max_attempts=10
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        # Use Python to make a quick HTTP request – no curl dependency
        if python -c "import requests; requests.get('http://localhost:${port}/health', timeout=2)" 2>/dev/null; then
            echo "[START] Port ${port} ready."
            return 0
        fi
        echo "[WAIT] Port ${port} not ready yet... (attempt $attempt)"
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "[ERROR] Port ${port} failed to start after ${max_attempts} attempts."
    return 1
}

# Check each microservice
wait_for_port 5003
wait_for_port 5004
wait_for_port 5005
echo "[START] Microservice layer ready."

# Start FastAPI backend on port 8000 (internal)
echo "[START] Starting FastAPI backend on port 8000..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for FastAPI to be ready before starting Streamlit
echo "[START] Waiting for FastAPI to be ready..."
until python -c "import requests; requests.get('http://localhost:8000/')" 2>/dev/null; do
    echo "[WAIT] FastAPI not ready yet..."
    sleep 2
done
echo "[START] FastAPI backend ready."

# Start Streamlit dashboard on the public port 7860
echo "[START] Starting Streamlit dashboard on port 7860..."
streamlit run dashboard.py --server.port 7860 --server.address 0.0.0.0

# Keep all processes alive (streamlit runs in foreground)
wait $FRONTEND_PID $PAYMENT_PID $HRDB_PID $API_PID