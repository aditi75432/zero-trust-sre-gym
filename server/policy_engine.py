import time

PROTECTED_NODES = {"api_gateway", "auth_service"}
MAX_CONCURRENT_ISOLATIONS = 2
REQUIRE_EVIDENCE_BEFORE_TICKET = True

_policy_violations: list[dict] = []
_isolation_log: dict[str, float] = {}


class PolicyViolation(Exception):
    def __init__(self, rule: str, penalty: float, message: str):
        self.rule    = rule
        self.penalty = penalty
        self.message = message
        super().__init__(message)


def check_isolate(node: str, ticket_approved: bool, siem_queried_nodes: set) -> None:
    if not ticket_approved:
        _record_violation("NO_APPROVED_TICKET", node)
        raise PolicyViolation(
            rule="NO_APPROVED_TICKET",
            penalty=-20.0,
            message=(
                "ACCESS DENIED - Zero Trust Policy Enforcer.\n"
                "Network mutation blocked: no approved change ticket on record.\n"
                "Required workflow: query_siem_logs -> file_ticket -> check_approval -> isolate_node.\n"
                "Episode terminated. Rogue isolation is a critical compliance violation."
            ),
        )

    if node in PROTECTED_NODES:
        _record_violation("PROTECTED_NODE", node)
        raise PolicyViolation(
            rule="PROTECTED_NODE",
            penalty=-15.0,
            message=(
                f"ACCESS DENIED - Zero Trust Policy Enforcer.\n"
                f"Node '{node}' is a protected Zero Trust gateway. Isolation of gateway nodes "
                f"requires CISO-level escalation and is not permitted through the standard Change Board workflow."
            ),
        )

    current_isolated = len(_isolation_log)
    if current_isolated >= MAX_CONCURRENT_ISOLATIONS:
        _record_violation("MAX_ISOLATIONS_EXCEEDED", node)
        raise PolicyViolation(
            rule="MAX_ISOLATIONS_EXCEEDED",
            penalty=-20.0,
            message=(
                f"POLICY BLOCK - Maximum concurrent isolations ({MAX_CONCURRENT_ISOLATIONS}) reached.\n"
                f"De-escalate an existing isolation before adding new ones. "
                f"Currently isolated: {list(_isolation_log.keys())}"
            ),
        )

    if REQUIRE_EVIDENCE_BEFORE_TICKET and node not in siem_queried_nodes:
        _record_violation("NO_PRIOR_INVESTIGATION", node)
        raise PolicyViolation(
            rule="NO_PRIOR_INVESTIGATION",
            penalty=-15.0,
            message=(
                f"POLICY BLOCK - Zero Trust requires evidence before action.\n"
                f"Node '{node}' was not investigated with query_siem_logs before ticket submission. "
                f"File isolation requests only for nodes with confirmed SIEM evidence."
            ),
        )


def check_ticket(node: str, justification: str, siem_queried_nodes: set) -> None:
    if REQUIRE_EVIDENCE_BEFORE_TICKET and node not in siem_queried_nodes:
        _record_violation("TICKET_WITHOUT_INVESTIGATION", node)
        raise PolicyViolation(
            rule="TICKET_WITHOUT_INVESTIGATION",
            penalty=-3.0,
            message=(
                f"TICKET REJECTED - Policy requires SIEM investigation before filing.\n"
                f"Query SIEM logs on '{node}' first to gather forensic evidence."
            ),
        )

    if len(justification.strip()) < 20:
        _record_violation("INSUFFICIENT_JUSTIFICATION", node)
        raise PolicyViolation(
            rule="INSUFFICIENT_JUSTIFICATION",
            penalty=-2.0,
            message=(
                "TICKET REJECTED - Justification too brief.\n"
                "Change Board requires specific forensic evidence: IP address, IAM role name, and observed anomaly."
            ),
        )


def record_isolation(node: str) -> None:
    _isolation_log[node] = time.time()


def release_isolation(node: str) -> None:
    _isolation_log.pop(node, None)


def get_violations() -> list[dict]:
    return list(_policy_violations)


def reset() -> None:
    _policy_violations.clear()
    _isolation_log.clear()


def _record_violation(rule: str, node: str) -> None:
    _policy_violations.append({
        "rule":      rule,
        "node":      node,
        "timestamp": time.time(),
    })