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
        latency = random.uniform(1.2, 3.0)
        time.sleep(latency)
        _emit(f"SERVICE=payment EVENT=HEALTH_CHECK LATENCY={latency:.2f}s STATUS=DEGRADED")
        return jsonify({"status": "degraded", "latency_ms": int(latency * 1000)}), 200
    return jsonify({"status": "healthy", "latency_ms": random.randint(15, 55)}), 200


@app.route("/charge", methods=["POST"])
def charge():
    with _lock:
        state = dict(_attack_state)
    if state.get("active"):
        ip   = state.get("ip", "10.0.0.1")
        role = state.get("role", "payment-processor-svc")
        tt   = state.get("threat_type", "data_exfiltration")
        _emit(f"SERVICE=payment EVENT=DATA_EXFILTRATION THREAT={tt} SOURCE_IP={ip} IAM_ROLE={role} BYTES={random.randint(100000, 50000000)} DEST=external-203.0.113.42")
        _emit(f"SERVICE=payment EVENT=UNAUTHORIZED_DB_DUMP ENDPOINT=/internal/transactions STATUS=200 ROWS={random.randint(5000, 200000)}")
    return jsonify({"charged": True, "transaction_id": f"TXN-{random.randint(10000, 99999)}"})


@app.route("/attack", methods=["POST"])
def attack():
    data = request.json or {}
    with _lock:
        _attack_state.update({
            "active":       True,
            "threat_type":  data.get("threat_type", "data_exfiltration"),
            "ip":           data.get("ip", "10.0.0.1"),
            "role":         data.get("role", "payment-processor-svc"),
            "cve_context":  data.get("cve_context", ""),
            "started_at":   time.time(),
        })
    ip   = data.get("ip", "10.0.0.1")
    role = data.get("role", "payment-processor-svc")
    tt   = data.get("threat_type", "data_exfiltration")
    bytes_val = random.randint(100 * 1024 * 1024, 500 * 1024 * 1024)
    _emit(f"SERVICE=payment EVENT=ATTACK_TRIGGERED THREAT={tt} SOURCE_IP={ip} IAM_ROLE={role}")
    _emit(f"SERVICE=payment EVENT=IAM_ASSUME ROLE={role} FROM={ip} RESULT=SUCCESS OUTSIDE_BUSINESS_HOURS=true")
    _emit(f"SERVICE=payment EVENT=OUTBOUND_TRANSFER DEST=203.0.113.42 BYTES={bytes_val} PROTOCOL=HTTPS")
    _emit(f"SERVICE=payment EVENT=INTERNAL_API_SCAN ENDPOINTS_PROBED=/admin/users,/internal/db_dump,/metrics ATTEMPTS=47")
    _emit(f"SERVICE=payment EVENT=DATA_EXFILTRATION ROWS_ACCESSED={random.randint(10000, 500000)} TABLE=transactions")
    cve = data.get("cve_context", "")
    if cve:
        _emit(f"SERVICE=payment EVENT=CVE_INDICATOR CONTEXT={cve[:120]}")
    return jsonify({"status": "attack_triggered"}), 200


@app.route("/isolate", methods=["POST"])
def isolate():
    with _lock:
        _attack_state["active"] = False
    _emit("SERVICE=payment EVENT=NODE_ISOLATED REASON=SECURITY_RESPONSE")
    return jsonify({"status": "isolated"}), 200


@app.route("/reset_attack", methods=["POST"])
def reset_attack():
    with _lock:
        _attack_state.clear()
    _emit("SERVICE=payment EVENT=ATTACK_STATE_RESET")
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
        "service":        "payment",
        "under_attack":   active,
        "transaction_count": random.randint(500, 2000),
        "error_rate":     random.uniform(0.20, 0.60) if active else random.uniform(0.001, 0.005),
        "p99_latency_ms": random.randint(1200, 3000) if active else random.randint(20, 70),
        "data_transfer_bytes": random.randint(100_000_000, 500_000_000) if active else random.randint(1000, 50000),
    }), 200


if __name__ == "__main__":
    _emit("SERVICE=payment EVENT=STARTUP STATUS=OK")
    app.run(host="0.0.0.0", port=5004, threaded=True)