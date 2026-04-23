import time
import random
import requests
import threading

SERVICE_PORTS = {
    "frontend": 5003,
    "payment":  5004,
    "hr_db":    5005,
}

PROPAGATION_GRAPH = {
    "auth_service": ["frontend", "payment"],
    "frontend":     ["hr_db"],
    "payment":      [],
    "hr_db":        [],
    "api_gateway":  ["frontend", "payment"],
}

_attack_state: dict[str, dict] = {}
_state_lock = threading.Lock()


def execute_attack(scenario: dict) -> dict[str, bool]:
    results = {}
    for i, node in enumerate(scenario.get("compromised_nodes", [])):
        if node not in SERVICE_PORTS:
            results[node] = False
            continue

        ip   = scenario["threat_ips"][i] if i < len(scenario["threat_ips"]) else f"10.0.{random.randint(1,10)}.{random.randint(10,99)}"
        role = scenario["iam_roles"][i]  if i < len(scenario["iam_roles"])  else "unknown-svc"

        payload = {
            "threat_type": scenario.get("threat_type", "data_exfiltration"),
            "ip":          ip,
            "role":        role,
            "cve_context": scenario.get("cve_context", ""),
        }

        try:
            r = requests.post(
                f"http://localhost:{SERVICE_PORTS[node]}/attack",
                json=payload,
                timeout=3,
            )
            ok = r.status_code == 200
        except Exception:
            ok = False

        with _state_lock:
            _attack_state[node] = {**payload, "active": ok, "started_at": time.time()}

        if ok:
            _propagate(node, payload)

        results[node] = ok

    return results


def isolate_node(node: str) -> bool:
    if node not in SERVICE_PORTS:
        return False
    try:
        r = requests.post(
            f"http://localhost:{SERVICE_PORTS[node]}/isolate",
            timeout=3,
        )
        with _state_lock:
            if node in _attack_state:
                _attack_state[node]["active"] = False
        return r.status_code == 200
    except Exception:
        return False


def get_node_attack_state(node: str) -> dict:
    with _state_lock:
        return dict(_attack_state.get(node, {}))


def reset_all_attacks() -> None:
    with _state_lock:
        nodes_to_reset = list(_attack_state.keys())

    for node in nodes_to_reset:
        if node in SERVICE_PORTS:
            try:
                requests.post(
                    f"http://localhost:{SERVICE_PORTS[node]}/reset_attack",
                    timeout=2,
                )
            except Exception:
                pass

    with _state_lock:
        _attack_state.clear()


def _propagate(source: str, base_payload: dict) -> None:
    targets = PROPAGATION_GRAPH.get(source, [])
    for target in targets:
        if target not in SERVICE_PORTS:
            continue
        propagated = {
            **base_payload,
            "threat_type":  "lateral_movement",
            "propagated_from": source,
        }
        try:
            requests.post(
                f"http://localhost:{SERVICE_PORTS[target]}/attack",
                json=propagated,
                timeout=2,
            )
            with _state_lock:
                _attack_state[target] = {**propagated, "active": True, "started_at": time.time()}
        except Exception:
            pass