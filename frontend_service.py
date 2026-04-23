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
        latency = random.uniform(0.8, 2.5)
        time.sleep(latency)
        _emit(f"SERVICE=frontend EVENT=HEALTH_CHECK LATENCY={latency:.2f}s STATUS=DEGRADED")
        return jsonify({"status": "degraded", "latency_ms": int(latency * 1000)}), 200
    return jsonify({"status": "healthy", "latency_ms": random.randint(12, 40)}), 200


@app.route("/process", methods=["POST"])
def process():
    with _lock:
        state = dict(_attack_state)
    if state.get("active"):
        ip   = state.get("ip", "10.0.0.1")
        role = state.get("role", "unknown")
        tt   = state.get("threat_type", "lateral_movement")
        _emit(f"SERVICE=frontend EVENT=UNAUTHORIZED_INTERNAL_CALL THREAT={tt} SOURCE_IP={ip} IAM_ROLE={role} ENDPOINT=/internal/admin")
        _emit(f"SERVICE=frontend EVENT=ANOMALOUS_OUTBOUND DEST_IP=203.0.113.42 BYTES={random.randint(50000, 5000000)}")
    return jsonify({"processed": True})


@app.route("/attack", methods=["POST"])
def attack():
    data = request.json or {}
    with _lock:
        _attack_state.update({
            "active":       True,
            "threat_type":  data.get("threat_type", "lateral_movement"),
            "ip":           data.get("ip", "10.0.0.1"),
            "role":         data.get("role", "frontend-webapp-svc"),
            "cve_context":  data.get("cve_context", ""),
            "started_at":   time.time(),
        })
    ip   = data.get("ip", "10.0.0.1")
    role = data.get("role", "frontend-webapp-svc")
    tt   = data.get("threat_type", "lateral_movement")
    _emit(f"SERVICE=frontend EVENT=ATTACK_TRIGGERED THREAT={tt} SOURCE_IP={ip} IAM_ROLE={role}")
    _emit(f"SERVICE=frontend EVENT=IAM_ASSUME ROLE={role} FROM={ip} RESULT=SUCCESS")
    _emit(f"SERVICE=frontend EVENT=LATERAL_PROBE TARGET=payment PORT=5004 ATTEMPTS=23")
    _emit(f"SERVICE=frontend EVENT=LATERAL_PROBE TARGET=hr_db PORT=5005 ATTEMPTS=17")
    _emit(f"SERVICE=frontend EVENT=UNAUTHORIZED_API_CALL ENDPOINT=/internal/db_dump STATUS=403")
    cve = data.get("cve_context", "")
    if cve:
        _emit(f"SERVICE=frontend EVENT=CVE_INDICATOR CONTEXT={cve[:120]}")
    return jsonify({"status": "attack_triggered"}), 200


@app.route("/isolate", methods=["POST"])
def isolate():
    with _lock:
        _attack_state["active"] = False
    _emit("SERVICE=frontend EVENT=NODE_ISOLATED REASON=SECURITY_RESPONSE")
    return jsonify({"status": "isolated"}), 200


@app.route("/reset_attack", methods=["POST"])
def reset_attack():
    with _lock:
        _attack_state.clear()
    _emit("SERVICE=frontend EVENT=ATTACK_STATE_RESET")
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
        "service":        "frontend",
        "under_attack":   active,
        "request_count":  random.randint(1000, 5000),
        "error_rate":     random.uniform(0.15, 0.40) if active else random.uniform(0.001, 0.01),
        "p99_latency_ms": random.randint(800, 2500) if active else random.randint(15, 50),
    }), 200


if __name__ == "__main__":
    _emit("SERVICE=frontend EVENT=STARTUP STATUS=OK")
    app.run(host="0.0.0.0", port=5003, threaded=True)