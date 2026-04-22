---
title: Autonomous Zero Trust SRE
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
- openenv

---

# Autonomous Zero Trust SRE Environment

**Theme Alignment:** Theme 3.1 (World Modeling - Professional Tasks) 
**Bonus Sub-Theme Target:** Scaler AI Labs (Multi-App RL Environment for Enterprise Workflows)

### The Core Problem
Modern enterprise cloud systems are deeply interconnected and highly fragile. When a structural anomaly or cyberattack occurs, Level 1 Security Operations Center (SOC) analysts face massive alert fatigue, sifting through thousands of false positives under extreme time pressure. 

Currently, the industry trains autonomous AI agents as "cowboys." If an agent detects a threat, it mitigates it aggressively. However, in a real enterprise setting, if an engineer or an agent blindly isolates a core database without diagnostic proof, a tracking ticket, and formal change approval, it causes a self-inflicted global production outage. We must train frontier models to perform safe anomaly detection and remediation strictly within the bounds of standard ITIL workflows and Zero Trust architecture.

### The Solution and Objective
This project provides a real-world, multi-app benchmark for Cloud Security operations. It features a dynamic, Partially Observable Markov Decision Process (POMDP) that simulates a multi-tier Zero Trust enterprise. 

The primary objective is to evaluate and train frontier models to neutralize threats while maintaining strict compliance and service uptime. The environment forces the language model to prioritize system integrity, active diagnostic reasoning, and sequential workflow adherence over aggressive, blind remediation.

### Unique Selling Proposition and Novelty
Most benchmark submissions rely on single-system, static puzzle boxes where the answers are readily available. This environment acts as a living, adversarial enterprise simulation. The agent does not get the answers handed to it. It must actively query tools to uncover the truth, ignore competing distractor alerts, and push its findings through an administrative workflow before the infrastructure degrades. 

### Key Innovations (Top-Tier Differentiators)
* **Multi-App Workflow Orchestration:** The agent does not just interact with infrastructure. It must orchestrate actions across three distinct simulated systems: Diagnostic Tools (SIEM/eBPF), a Service Desk (Ticketing), and a Change Management Engine (Compliance).
* **Strict Partial Observability and Ambiguity:** Critical threat targets are hidden. Querying the SIEM logs only provides low-confidence subnets and ambiguous decoy data. The agent must synthesize this partial knowledge and execute secondary eBPF traces to pinpoint the exact malicious IP or compromised IAM role.
* **Competing Priorities (Signal vs. Noise):** The environment actively tests triage reasoning by injecting P3 distractors (e.g., routine memory leaks) alongside P1 Critical data exfiltrations. The agent must learn to ignore the noise and focus on the primary threat.
* **Zero Trust Policy Enforcement:** The environment itself acts as an unforgiving compliance overseer. If the agent attempts a destructive action without first generating an incident ticket and securing cryptographic approval based on forensic evidence, the Zero Trust engine intercepts the action, logs a fatal compliance failure, and terminates the episode.
* **Cascading Infrastructure Failures:** Services are mapped via a Directed Acyclic Graph (DAG). Isolating a core dependency causes downstream services to immediately degrade, penalizing the agent for collateral damage.

### Task Difficulty Gradient
* **Level 1 (Brute Force):** The agent must utilize tools to confirm a malicious external IP address and block it without disrupting legitimate internal service traffic.
* **Level 2 (Lateral Movement):** The agent must track internal network flow and isolate a compromised frontend pod before the threat reaches the core HR database.
* **Level 3 (Insider Threat):** The true test of reasoning. The agent receives an ambiguous network anomaly alert alongside a noisy distractor. It must gather information to reveal the true target, open an incident ticket, secure change approval, and safely revoke the compromised IAM role.

### Reward Model and Evaluation Logic
We implemented a highly shaped, dense, multi-factor cost matrix designed specifically to stabilize Proximal Policy Optimization (PPO) algorithms.
* **Operational Costs:** Every step incurs a fractional time penalty to simulate SLA pressure, alongside dedicated administrative costs for tool usage and ticket generation.
* **Positive Reinforcement:** Points are awarded for sequential logic, successfully uncovering evidence, and neutralizing the threat.
* **Zero Trust Constraints:** Massive point penalties and immediate episode terminations are applied for blind actions, guessing, taking the bait on distractor alerts, or attempting to alter production infrastructure without compliance approval. 

### System Architecture
The environment is built on a robust, four-tier architecture designed for rapid reinforcement learning and live telemetry.
* **The OpenEnv Server:** A FastAPI-powered backend implementing the full OpenEnv specification, ensuring repeatable containerized execution.
* **The Simulation Engine:** A complex Python state machine managing the microservice DAG, injecting enterprise noise, managing multi-app state, and calculating the multi-factor reward score.
* **Live Telemetry Dashboard:** A real-time Streamlit observer panel that visualizes the microservice DAG, active tickets, compliance approvals, and reward trajectories for observability and oversight.
* **The Reinforcement Learning Pipeline:** A highly optimized training loop utilizing the Hugging Face TRL library. We use Proximal Policy Optimization (PPO) with dynamic prompting to teach small, fast models to navigate the complex environment constraints.

### Post-Training and Self-Improvement Strategy
To align the agent, we run a continuous reinforcement learning loop. Because our environment provides immediate, strict, mathematically shaped rewards rather than delayed or ambiguous feedback, the algorithm rapidly accumulates accurate gradients. Over successive epochs, the policy updates to stop hallucinating destructive shortcuts, avoids the compliance traps, and consistently converges on the optimal "gather evidence -> ticket -> approve -> remediate" enterprise sequence.