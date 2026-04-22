#!/bin/bash

echo "Starting Zero Trust Enterprise-in-a-Box..."

# 1. Start the Rogue Service (Nginx) in the background
nginx &

# 2. Start the Mock Jira ITSM on port 8080 in the background
cat << 'EOF' > jira_app.py
from fastapi import FastAPI
import uuid

app = FastAPI()
tickets = {}

@app.post("/ticket")
def create(ip: str):
    ticket_id = str(uuid.uuid4())[:8]
    tickets[ticket_id] = ip
    return {"id": ticket_id}
    
@app.get("/ticket/{ticket_id}")
def get(ticket_id: str):
    return {"approved": True}
EOF

uvicorn jira_app:app --host 0.0.0.0 --port 8080 &

# 3. Start the OpenEnv Zero Trust Server on the mandatory port 7860
uvicorn server.app:app --host 0.0.0.0 --port 7860