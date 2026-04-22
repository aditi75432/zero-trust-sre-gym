import requests
import time

BASE_URL = "http://localhost:7860"

def print_step(title, res):
    print(f"\n=== {title} ===")
    if res.status_code != 200:
        print(f"CRITICAL ERROR: Server returned {res.status_code}")
        print(res.text)
        exit(1)
        
    data = res.json()
    print(f"Reward:  {data['reward']['value']}")
    print(f"Message: {data['reward']['message']}")
    print(f"Console Output: {data['observation']['command_output']}")
    if data['done']:
        print(">>> EPISODE TERMINATED <<<")

print("Testing ADVANCED Open-Ended Zero Trust Environment...")

requests.post(f"{BASE_URL}/reset", json={"task_id": "test"})

# Step 1: Raw Bash to read logs
res = requests.post(f"{BASE_URL}/step", json={
    "tool_name": "execute_bash", 
    "payload": {"command": "cat /var/log/syslog | grep error"},
    "justification": "Finding the threat IP"
})
print_step("Step 1: Execute Bash (Read Logs)", res)

# Extract the IP from the console output (simulating the LLM parsing text)
output = res.json()['observation']['command_output']
threat_ip = output.split("Failed auth from ")[1].split(" -")[0]

# Step 2: Raw HTTP to Jira
res = requests.post(f"{BASE_URL}/step", json={
    "tool_name": "execute_http_request", 
    "payload": {
        "method": "POST",
        "url": "http://jira-mock:8080/ticket",
        "body": {"ip": threat_ip}
    },
    "justification": "Filing incident ticket for the threat IP."
})
print_step("Step 2: Execute HTTP (File Ticket)", res)

# Extract Ticket ID
ticket_id = res.json()['observation']['command_output'].split('\"id\":\"')[1].replace('\"}', '')

# Step 3: Raw HTTP to check status
res = requests.post(f"{BASE_URL}/step", json={
    "tool_name": "execute_http_request", 
    "payload": {
        "method": "GET",
        "url": f"http://jira-mock:8080/ticket/{ticket_id}"
    },
    "justification": "Checking cryptographic approval."
})
print_step("Step 3: Execute HTTP (Check Approval)", res)

# Step 4: Raw Bash to Remedite
res = requests.post(f"{BASE_URL}/step", json={
    "tool_name": "execute_bash", 
    "payload": {"command": "docker pause rogue-service"},
    "justification": "Executing approved isolation."
})
print_step("Step 4: Execute Bash (Isolate Container)", res)