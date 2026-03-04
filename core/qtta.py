"""
Quantum Tunneling Threshold Adapter (QTTA)
==========================================

The centrepiece novel module of Q-SAND. This module replaces static anomaly
score thresholds with a dynamic function inspired by the WKB (Wentzel–Kramers–
Brillouin) quantum tunneling probability approximation.

Quantum Tunneling Analogy for Anomaly Detection
------------------------------------------------

In quantum mechanics, tunneling is the phenomenon where a particle passes 
through a potential barrier that it classically could not surmount. The WKB
approximation gives the tunneling probability:

    T(E, V0, d) = exp( -2·d · sqrt(2·m·(V0 - E)) / ℏ )

where:
    E   = particle energy
    V0  = barrier height (potential energy)
    d   = barrier width
    m   = particle mass
    ℏ   = reduced Planck's constant

Mapping to Anomaly Detection
-----------------------------
    E   → current_alert_pressure:  rolling anomaly rate in recent packets (0–1)
          Analogous to the particle's energy — higher alert pressure means the
          "particle" (detection system) has more energy to penetrate the barrier.

    V0  → noise_floor:  baseline false-positive rate estimated from history.
          Analogous to the barrier height — represents the "resistance" the system
          has against false positives.

    d   → temporal_window:  a configurable sensitivity parameter controlling how
          quickly the threshold adapts. Wider d = more conservative adaptation.

    m, ℏ → normalized to 1 for simplicity.

Behavior
--------
    When E << V0 (few anomalies):
        T is very small → κ stays near base_threshold (conservative)
        → Avoids false positives in quiet periods

    When E → V0 (attack building):
        T increases → κ decreases (threshold lowers)
        → Catches low-and-slow attacks that a static threshold would miss

    When E >= V0 (full attack):
        T = 1.0 → κ drops to minimum
        → Maximum sensitivity during active attacks

The final detection threshold is:
    κ(t) = base_threshold × (1 - α × T(t))

where α (alpha) is a sensitivity parameter (default 0.3) that controls how
much the tunneling probability can lower the threshold.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


class QTTAThreshold:
    """
    Quantum Tunneling-inspired Threshold Adaptation.

    Replaces static anomaly score thresholds with a dynamic function
    inspired by the WKB quantum tunneling probability:

        T(E, V0, d) = exp(-2 * d * sqrt(2m(V0 - E)) / hbar)

    In our analogy:
        E  = current_alert_pressure  (recent anomaly rate, 0-1)
        V0 = noise_floor             (baseline false positive rate)
        d  = temporal_window         (detection sensitivity, configurable)
        m  = 1 (normalized)
        hbar = 1 (normalized)

    When E << V0 (few anomalies): T is small → threshold stays HIGH
        (conservative, avoid false positives)
    When E approaches V0 (attack underway): T rises → threshold LOWERS
        (aggressive, catch low-and-slow attacks)

    The detection threshold kappa is then:
        kappa(t) = base_threshold * (1 - alpha * T(t))
    where alpha is a sensitivity parameter (0.3 default).
    """

    def __init__(self, base_threshold: float = 0.5, alpha: float = 0.3, d: float = 1.0):
        """
        Initialize the QTTA module.

        Args:
            base_threshold: The static baseline threshold κ₀. Scores above this
                            are considered anomalous in the static regime.
            alpha: Sensitivity parameter controlling how much tunneling can
                   lower the threshold. Range [0, 1]. Higher = more adaptive.
            d: Temporal window / barrier width. Controls adaptation speed.
               Larger d = more conservative (slower adaptation).
        """
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.d = d
        self.history: List[float] = []          # Rolling window of recent anomaly scores
        self.window_size: int = 50              # Last N packets used for alert pressure calculation
        self.threshold_history: List[Dict[str, Any]] = []
        self.current_threshold: float = base_threshold

    def _compute_tunneling_probability(self, E: float, V0: float) -> float:
        """
        WKB approximation of quantum tunneling probability.

        Implements: T = exp(-2 * d * sqrt(2 * (V0 - E)))

        When E >= V0, the particle has enough energy to classically overcome
        the barrier, so T = 1.0 (certain transmission).

        When E < V0, the probability decreases exponentially with the
        square root of the energy deficit (V0 - E), modulated by the
        barrier width d.

        Args:
            E: Alert pressure (particle energy analog), range [0, 1]
            V0: Noise floor (barrier height analog), range [0, 1]

        Returns:
            Tunneling probability T ∈ [0, 1]
        """
        if E >= V0:
            return 1.0  # Particle is above barrier — always passes through
        # WKB formula: T = exp(-2·d·√(2·(V0 - E)))
        # The factor √(2·(V0-E)) represents the evanescent wave decay rate
        exponent = -2 * self.d * np.sqrt(2 * max(V0 - E, 0))
        return float(np.exp(exponent))

    def _compute_alert_pressure(self) -> float:
        """
        Compute alert pressure E: rolling anomaly rate in recent history.

        E represents the fraction of recent packets that exceeded the
        base threshold — analogous to the particle's kinetic energy.

        High E means many recent anomalies → system should be more
        sensitive (lower threshold).

        Returns:
            E ∈ [0, 1]
        """
        if len(self.history) == 0:
            return 0.0
        recent = self.history[-self.window_size:]
        return float(np.mean([1 if score > self.base_threshold else 0
                              for score in recent]))

    def _compute_noise_floor(self) -> float:
        """
        Compute noise floor V0: baseline false positive estimate from full history.

        V0 is estimated as the 20th percentile of all historical scores,
        clipped to [0.05, 0.6]. This represents the "normal" background
        noise level — the potential barrier that the alert pressure must
        overcome to trigger threshold adaptation.

        Returns:
            V0 ∈ [0.05, 0.6]
        """
        if len(self.history) < 10:
            return 0.3  # Default noise floor when insufficient history
        scores = np.array(self.history)
        return float(np.clip(np.percentile(scores, 20), 0.05, 0.6))

    def update(self, new_score: float) -> Tuple[bool, float, float]:
        """
        Ingest a new anomaly score, update internal QTTA state,
        and compute the new adaptive threshold.

        The threshold adaptation follows:
            1. Append score to history
            2. Compute E (alert pressure) from recent window
            3. Compute V0 (noise floor) from full history
            4. Compute T (tunneling probability) via WKB formula
            5. Update κ = base_threshold × (1 - α × T)
            6. Classify: anomaly if score > κ

        Args:
            new_score: Raw anomaly probability from the ML model

        Returns:
            Tuple of (is_anomaly, current_threshold, tunneling_probability)
        """
        self.history.append(float(new_score))

        E = self._compute_alert_pressure()
        V0 = self._compute_noise_floor()
        T = self._compute_tunneling_probability(E, V0)

        # κ(t) = κ₀ × (1 - α × T(t))
        # When T is high (attack), threshold drops; when T is low (quiet), stays near κ₀
        self.current_threshold = self.base_threshold * (1 - self.alpha * T)
        self.current_threshold = float(np.clip(self.current_threshold, 0.1, 0.95))

        self.threshold_history.append({
            "score": float(new_score),
            "threshold": self.current_threshold,
            "tunneling_prob": T,
            "alert_pressure": E,
            "noise_floor": V0,
            "is_anomaly": bool(new_score > self.current_threshold)
        })

        return (
            bool(new_score > self.current_threshold),
            self.current_threshold,
            T
        )

    def batch_update(self, scores: List[float]) -> List[Tuple[bool, float, float]]:
        """
        Process a batch of anomaly scores sequentially.

        Each score is processed in order, maintaining the rolling window
        state for accurate alert pressure and noise floor estimation.

        Args:
            scores: List of anomaly probability scores

        Returns:
            List of (is_anomaly, threshold, tunneling_prob) tuples
        """
        return [self.update(s) for s in scores]

    def get_state_series(self) -> List[Dict[str, Any]]:
        """
        Returns the complete threshold history for frontend charting.

        Each entry contains: score, threshold, tunneling_prob,
        alert_pressure, noise_floor, is_anomaly.

        Returns:
            List of state dictionaries for visualization
        """
        return self.threshold_history

    def reset(self) -> None:
        """Reset the QTTA to initial state, clearing all history."""
        self.history = []
        self.threshold_history = []
        self.current_threshold = self.base_threshold
