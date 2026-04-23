import time
import random
import threading
from collections import deque
from flask import Flask, jsonify, request

app = Flask(__name__)

_attack_state: dict = {}
_log_buffer: deque = deque(maxlen=200)
_lock = threading.Lock()


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _emit(msg: str) -> None:
    with _lock:
        _log_buffer.append(f"[{_ts()}] {msg}")


@app.route("/health")
def health():
    with _lock:
        active = bool(_attack_state.get("active"))
    if active:
        _emit(f"SERVICE=hr_db EVENT=HEALTH_CHECK STATUS=COMPROMISED")
        return jsonify({"status": "compromised", "latency_ms": random.randint(500, 1500)}), 200
    return jsonify({"status": "healthy", "latency_ms": random.randint(10, 30)}), 200


@app.route("/query", methods=["POST"])
def query():
    with _lock:
        state = dict(_attack_state)
    if state.get("active"):
        ip   = state.get("ip", "10.0.0.1")
        role = state.get("role", "hr-reader-svc")
        _emit(f"SERVICE=hr_db EVENT=UNAUTHORIZED_BULK_READ SOURCE_IP={ip} IAM_ROLE={role} ROWS={random.randint(10000, 1000000)} TABLE=employee_pii")
        _emit(f"SERVICE=hr_db EVENT=PRIVILEGE_ESCALATION ROLE={role} ATTEMPTED_ROLE=admin-full-access ATTEMPTS=12 STATUS=DENIED_BUT_CONTINUING")
    return jsonify({"rows": random.randint(1, 50)})


@app.route("/attack", methods=["POST"])
def attack():
    data = request.json or {}
    with _lock:
        _attack_state.update({
            "active":       True,
            "threat_type":  data.get("threat_type", "privilege_escalation"),
            "ip":           data.get("ip", "10.0.0.1"),
            "role":         data.get("role", "hr-reader-svc"),
            "cve_context":  data.get("cve_context", ""),
            "started_at":   time.time(),
        })
    ip   = data.get("ip", "10.0.0.1")
    role = data.get("role", "hr-reader-svc")
    tt   = data.get("threat_type", "privilege_escalation")
    _emit(f"SERVICE=hr_db EVENT=ATTACK_TRIGGERED THREAT={tt} SOURCE_IP={ip} IAM_ROLE={role}")
    _emit(f"SERVICE=hr_db EVENT=IAM_ASSUME ROLE={role} FROM={ip} OUTSIDE_BUSINESS_HOURS=true RESULT=SUCCESS")
    _emit(f"SERVICE=hr_db EVENT=BULK_DATA_READ TABLE=employee_pii ROWS={random.randint(50000, 500000)} DEST_IP={ip}")
    _emit(f"SERVICE=hr_db EVENT=PRIVILEGE_ESCALATION ROLE={role} ATTEMPTED_ROLE=admin-full-access AssumeRole_CALLS=12")
    _emit(f"SERVICE=hr_db EVENT=OUTBOUND_EXFIL DEST=203.0.113.42 BYTES={random.randint(50000000, 300000000)}")
    cve = data.get("cve_context", "")
    if cve:
        _emit(f"SERVICE=hr_db EVENT=CVE_INDICATOR CONTEXT={cve[:120]}")
    return jsonify({"status": "attack_triggered"}), 200


@app.route("/isolate", methods=["POST"])
def isolate():
    with _lock:
        _attack_state["active"] = False
    _emit("SERVICE=hr_db EVENT=NODE_ISOLATED REASON=SECURITY_RESPONSE")
    return jsonify({"status": "isolated"}), 200


@app.route("/reset_attack", methods=["POST"])
def reset_attack():
    with _lock:
        _attack_state.clear()
    _emit("SERVICE=hr_db EVENT=ATTACK_STATE_RESET")
    return jsonify({"status": "reset"}), 200


@app.route("/logs")
def logs():
    with _lock:
        recent = list(_log_buffer)[-20:]
    return jsonify({"recent_logs": recent, "total": len(_log_buffer)}), 200


@app.route("/metrics")
def metrics():
    with _lock:
        active = bool(_attack_state.get("active"))
    return jsonify({
        "service":         "hr_db",
        "under_attack":    active,
        "query_count":     random.randint(200, 800),
        "error_rate":      random.uniform(0.10, 0.50) if active else random.uniform(0.0, 0.003),
        "p99_latency_ms":  random.randint(500, 1500) if active else random.randint(10, 35),
        "rows_accessed":   random.randint(100_000, 1_000_000) if active else random.randint(10, 500),
    }), 200


if __name__ == "__main__":
    _emit("SERVICE=hr_db EVENT=STARTUP STATUS=OK")
    app.run(host="0.0.0.0", port=5005, threaded=True)