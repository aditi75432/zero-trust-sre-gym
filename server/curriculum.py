"""
curriculum.py — Progressive difficulty controller for Zero Trust SRE Gym.

Tracks per-threat-type resolution rates across episodes and escalates
difficulty as the agent improves. The key insight: a static curriculum
wastes training steps on scenarios the agent already knows how to handle.
An adaptive curriculum keeps the agent at its learning edge.

Escalation ladder:
  warmup       → single fault, 15 steps, Junior judge
  beginner     → single fault with red herrings, 13 steps, Senior judge
  intermediate → harder faults, 11 steps, Senior judge
  advanced     → multi-fault, 9 steps, Principal judge  
  expert       → adversarial multi-fault targeting weak spots, 7 steps, Principal judge

The adversarial designer reads get_weakness_profile() to decide what to generate.
The environment reads get_difficulty() to set SLA and judge persona.
"""

import json
from pathlib import Path
from typing import Optional


DIFFICULTY_LEVELS = ["warmup", "beginner", "intermediate", "advanced", "expert"]

# Mastery thresholds for escalation.
# Episode count gates prevent premature escalation on lucky early wins.
ESCALATION_THRESHOLDS = {
    "warmup":       {"min_episodes": 2, "avg_mastery": 0.0},
    "beginner":     {"min_episodes": 4, "avg_mastery": 0.20},
    "intermediate": {"min_episodes": 8, "avg_mastery": 0.40},
    "advanced":     {"min_episodes": 12, "avg_mastery": 0.60},
    "expert":       {"min_episodes": 16, "avg_mastery": 0.75},
}


class CurriculumController:
    """
    Single instance that persists across episodes within a server session.
    Optionally writes to disk so training can resume after crashes.
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        self.mastery: dict[str, float] = {
            "data_exfiltration": 0.0,
            "lateral_movement": 0.0,
            "privilege_escalation": 0.0,
            "supply_chain": 0.0,
            "multi_fault": 0.0
        }
        self.episode_count: int = 0
        self.episode_log: list[dict] = []
        self.persistence_path = persistence_path
        
        if persistence_path and Path(persistence_path).exists():
            self._load()
            print(f"[Curriculum] Loaded from disk: episode {self.episode_count}, difficulty {self.get_difficulty()}")
    
    # ─── PUBLIC API ──────────────────────────────────────────────────────────
    
    def get_difficulty(self) -> str:
        """
        Returns current difficulty level based on mastery and episode count.
        Called by environment.py at every reset.
        """
        avg_mastery = self._avg_mastery()
        
        for level in reversed(DIFFICULTY_LEVELS):
            threshold = ESCALATION_THRESHOLDS[level]
            if (self.episode_count >= threshold["min_episodes"] and 
                avg_mastery >= threshold["avg_mastery"]):
                return level
        
        return "warmup"
    
    def get_weakness_profile(self) -> dict[str, float]:
        """
        Returns failure rates per threat type.
        Higher value = agent fails more at this = adversarial designer targets it.
        Called by adversarial_designer.py at scenario generation.
        """
        return {
            threat: round(1.0 - mastery, 3)
            for threat, mastery in self.mastery.items()
        }
    
    def update(
        self,
        threat_type: str,
        resolved: bool,
        total_reward: float,
        steps_taken: int
    ) -> None:
        """
        Call this at the end of every episode. Updates mastery and logs.
        
        Mastery delta logic:
        - Resolved: +0.10 base, +efficiency_bonus for fast resolution
        - Failed: -0.05 decay (slow decay prevents catastrophic forgetting)
        - multi_fault: extra bonus/decay because it's the hardest category
        """
        self.episode_count += 1
        
        # Normalize threat_type (LLM sometimes returns variants)
        normalized = self._normalize_threat_type(threat_type)
        
        if resolved:
            # Efficiency bonus: up to +0.08 extra for resolving quickly
            max_steps = 15.0
            efficiency = max(0.0, (max_steps - steps_taken) / max_steps)
            delta = 0.10 + (efficiency * 0.08)
            self.mastery[normalized] = min(1.0, self.mastery[normalized] + delta)
            
            # Multi-fault resolution also improves multi_fault mastery
            if normalized != "multi_fault" and self.episode_count > 0:
                difficulty = self.get_difficulty()
                if difficulty in ["advanced", "expert"]:
                    self.mastery["multi_fault"] = min(1.0, self.mastery["multi_fault"] + 0.05)
        else:
            self.mastery[normalized] = max(0.0, self.mastery[normalized] - 0.05)
        
        # Log this episode
        self.episode_log.append({
            "episode": self.episode_count,
            "threat_type": normalized,
            "resolved": resolved,
            "reward": round(total_reward, 2),
            "steps": steps_taken,
            "difficulty": self.get_difficulty(),
            "mastery_snapshot": dict(self.mastery)
        })
        
        # Trim log to last 100 episodes to avoid memory growth
        if len(self.episode_log) > 100:
            self.episode_log = self.episode_log[-100:]
        
        if self.persistence_path:
            self._save()
    
    def get_summary(self) -> dict:
        """Returns curriculum state for /curriculum endpoint and dashboard."""
        recent = self.episode_log[-10:] if self.episode_log else []
        return {
            "episode_count": self.episode_count,
            "difficulty": self.get_difficulty(),
            "mastery": {k: round(v, 3) for k, v in self.mastery.items()},
            "weakness_profile": self.get_weakness_profile(),
            "recent_rewards": [e["reward"] for e in recent],
            "resolution_rate": self._resolution_rate(recent),
            "avg_mastery": round(self._avg_mastery(), 3)
        }
    
    def reset_mastery(self) -> None:
        """Hard reset — for debugging and demo restarts."""
        self.mastery = {k: 0.0 for k in self.mastery}
        self.episode_count = 0
        self.episode_log.clear()
        if self.persistence_path:
            self._save()
    
    # ─── PRIVATE ─────────────────────────────────────────────────────────────
    
    def _avg_mastery(self) -> float:
        if not self.mastery:
            return 0.0
        return sum(self.mastery.values()) / len(self.mastery)
    
    def _resolution_rate(self, episodes: list) -> float:
        if not episodes:
            return 0.0
        return round(sum(1 for e in episodes if e["resolved"]) / len(episodes), 2)
    
    def _normalize_threat_type(self, threat_type: str) -> str:
        """Maps LLM-returned threat type variants to our standard keys."""
        threat_type = threat_type.lower().replace("-", "_").replace(" ", "_")
        
        mapping = {
            "exfiltration": "data_exfiltration",
            "data_theft": "data_exfiltration",
            "lateral": "lateral_movement",
            "pivoting": "lateral_movement",
            "privilege": "privilege_escalation",
            "privesc": "privilege_escalation",
            "supply": "supply_chain",
            "multi": "multi_fault",
            "compound": "multi_fault",
        }
        
        # Try exact match first
        if threat_type in self.mastery:
            return threat_type
        
        # Try partial match
        for key, value in mapping.items():
            if key in threat_type:
                return value
        
        # Default
        return "data_exfiltration"
    
    def _save(self) -> None:
        """Persist state to disk for training restarts."""
        data = {
            "mastery": self.mastery,
            "episode_count": self.episode_count,
            "episode_log": self.episode_log[-50:]
        }
        Path(self.persistence_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load(self) -> None:
        """Load persisted state from disk."""
        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)
            self.mastery.update(data.get("mastery", {}))
            self.episode_count = data.get("episode_count", 0)
            self.episode_log = data.get("episode_log", [])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Curriculum] Failed to load state: {e}. Starting fresh.")