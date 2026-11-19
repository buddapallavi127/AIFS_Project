"""
Multimodal Fusion Engine — Combining Vision AI + Numerical ML
==============================================================
Implements Dempster-Shafer inspired evidence fusion to combine
two independent AI modalities (image analysis and physicochemical
ML model) into a single unified water-quality assessment.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModalityEvidence:
    """Evidence from a single modality."""
    source: str               # "Vision AI" or "Numerical ML"
    probability_safe: float   # P(safe) from this modality
    confidence: float         # 0–1, how much to trust this modality
    details: Dict             # modality-specific details


@dataclass
class FusionResult:
    """Unified multimodal assessment."""
    # Individual modality results
    vision_evidence: Optional[ModalityEvidence]
    numerical_evidence: Optional[ModalityEvidence]

    # Fused decision
    fused_probability: float       # combined P(safe)
    fused_decision: str            # "SAFE", "UNSAFE", "INCONCLUSIVE"
    fused_confidence: float        # combined confidence
    fused_risk_level: str          # "Low", "Medium", "High", "Critical"

    # Agreement analysis
    modality_agreement: str        # "Strong Agreement", "Weak Agreement",
                                   # "Disagreement", "Strong Disagreement"
    agreement_score: float         # -1 (oppose) to +1 (agree)
    disagreement_details: List[str]  # what specifically disagrees

    # Fusion metadata
    fusion_method: str             # "Weighted Dempster-Shafer"
    vision_weight: float           # actual weight used for vision
    numerical_weight: float        # actual weight used for numerical

    # Decision thresholds
    safe_threshold: float
    uncertain_low: float
    uncertain_high: float

    # Human-readable
    summary: str
    recommendation: str
    action_items: List[str]


# ═══════════════════════════════════════════════════════════════════════
# FUSION LOGIC
# ═══════════════════════════════════════════════════════════════════════

# Base weights: Numerical ML is more reliable than vision for this domain
BASE_WEIGHT_NUMERICAL = 0.70
BASE_WEIGHT_VISION = 0.30

# Decision thresholds
SAFE_THRESHOLD = 0.55
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65


def _adjusted_weights(vision_confidence: float,
                      numerical_confidence: float) -> Tuple[float, float]:
    """Adjust weights based on per-modality confidence.

    If vision confidence is very low (e.g., ambiguous image), reduce
    its influence. If numerical data has missing features, reduce its weight.
    """
    w_v = BASE_WEIGHT_VISION * vision_confidence
    w_n = BASE_WEIGHT_NUMERICAL * numerical_confidence

    # Normalize
    total = w_v + w_n
    if total < 1e-6:
        return 0.5, 0.5
    return w_v / total, w_n / total


def _dempster_shafer_combine(p_safe_v: float, conf_v: float,
                              p_safe_n: float, conf_n: float) -> Tuple[float, float]:
    """Simplified Dempster-Shafer combination of two evidence sources.

    Each source provides:
      - belief in SAFE:   m(safe)   = confidence * p_safe
      - belief in UNSAFE: m(unsafe) = confidence * (1 - p_safe)
      - uncertainty:      m(Θ)      = 1 - confidence

    Returns combined (belief_safe, belief_unsafe) normalized.
    """
    # Vision mass assignments
    m1_safe = conf_v * p_safe_v
    m1_unsafe = conf_v * (1 - p_safe_v)
    m1_theta = 1 - conf_v  # uncertainty

    # Numerical mass assignments
    m2_safe = conf_n * p_safe_n
    m2_unsafe = conf_n * (1 - p_safe_n)
    m2_theta = 1 - conf_n

    # Dempster's combination rule
    # Agreeing on safe
    combined_safe = (m1_safe * m2_safe +
                     m1_safe * m2_theta +
                     m1_theta * m2_safe)

    # Agreeing on unsafe
    combined_unsafe = (m1_unsafe * m2_unsafe +
                       m1_unsafe * m2_theta +
                       m1_theta * m2_unsafe)

    # Conflict mass (safe vs unsafe)
    conflict = (m1_safe * m2_unsafe + m1_unsafe * m2_safe)

    # Remaining uncertainty
    combined_theta = m1_theta * m2_theta

    # Normalize (Dempster's rule divides by 1 - conflict)
    normalization = 1.0 - conflict
    if normalization < 1e-6:
        # Total conflict — maximum uncertainty
        return 0.5, 0.5

    combined_safe /= normalization
    combined_unsafe /= normalization

    # Convert back to probability
    total_belief = combined_safe + combined_unsafe
    if total_belief < 1e-6:
        p_fused = 0.5
    else:
        p_fused = combined_safe / total_belief

    # Combined confidence
    fused_confidence = 1.0 - (combined_theta / normalization)
    fused_confidence = max(0.0, min(1.0, fused_confidence))

    return p_fused, fused_confidence


def _assess_agreement(p_v: float, p_n: float,
                       decision_v: bool, decision_n: bool) -> Tuple[str, float, List[str]]:
    """Evaluate how well the two modalities agree."""
    # Agreement score: 1 if both agree strongly, -1 if strongly opposed
    diff = abs(p_v - p_n)
    same_decision = decision_v == decision_n

    details = []

    if same_decision and diff < 0.15:
        label = "Strong Agreement"
        score = 1.0 - diff
    elif same_decision and diff < 0.30:
        label = "Moderate Agreement"
        score = 0.6 - diff
    elif same_decision:
        label = "Weak Agreement"
        score = 0.3
        details.append(f"Both predict {'Safe' if decision_v else 'Unsafe'}, "
                       f"but probability gap is {diff:.0%}.")
    elif diff < 0.20:
        label = "Borderline Disagreement"
        score = -0.2
        details.append("Modalities give opposite decisions but probabilities "
                        "are close to the threshold.")
    else:
        label = "Strong Disagreement"
        score = -(0.5 + diff * 0.5)
        v_label = "Safe" if decision_v else "Unsafe"
        n_label = "Safe" if decision_n else "Unsafe"
        details.append(f"Vision AI says {v_label} ({p_v:.0%}), "
                       f"Numerical ML says {n_label} ({p_n:.0%}).")
        details.append("⚠️ RECOMMEND: Laboratory retest to resolve modality conflict.")

    return label, round(score, 3), details


def _build_recommendation(fused_prob: float, decision: str,
                           agreement: str, risk: str,
                           vision_safe: bool, numerical_safe: bool) -> Tuple[str, List[str]]:
    """Generate actionable recommendation and items."""
    actions = []

    if decision == "SAFE" and agreement.startswith("Strong"):
        rec = ("Both AI modalities concur: this water sample shows strong indicators "
               "of potability. Standard periodic monitoring recommended.")
        actions.append("✅ Clear for consumption under standard monitoring protocol.")

    elif decision == "SAFE" and "Disagree" in agreement:
        rec = ("The numerical model indicates safety, but vision analysis raises concerns. "
               "While the fused assessment leans safe, additional verification is advised.")
        actions.append("🔬 Schedule laboratory confirmation within 24 hours.")
        actions.append("📸 Re-capture water sample image under standardized lighting.")

    elif decision == "INCONCLUSIVE":
        rec = ("The multimodal assessment is inconclusive — the sample falls in the "
               "uncertainty zone between safe and unsafe thresholds. "
               "Do NOT use for drinking without laboratory confirmation.")
        actions.append("🚫 Do not consume until laboratory test confirms safety.")
        actions.append("🔬 Dispatch sample for microbial + chemical panel immediately.")
        actions.append("📊 Cross-reference with upstream water source data.")

    elif decision == "UNSAFE" and agreement.startswith("Strong"):
        rec = ("CRITICAL: Both AI modalities agree this water is unsafe. "
               "Immediate intervention required.")
        actions.append("🚫 Do NOT consume. Issue public advisory if municipal supply.")
        actions.append("🏭 Activate emergency treatment protocol.")
        actions.append("🔬 Rush laboratory analysis for specific contaminant identification.")

    elif decision == "UNSAFE":
        rec = ("The fused assessment indicates this water is unsafe for consumption. "
               "Treatment or alternative supply required.")
        actions.append("🚫 Do not consume without treatment.")
        actions.append("🔬 Laboratory confirmation recommended.")
        if not vision_safe:
            actions.append("📸 Visual anomalies detected — inspect source for contamination.")

    else:
        rec = "Assessment complete. Review individual modality details for specifics."
        actions.append("📋 Review detailed metrics below.")

    return rec, actions


# ═══════════════════════════════════════════════════════════════════════
# MAIN FUSION FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def fuse_assessments(
    vision_prob_safe: float,
    vision_confidence: float,
    vision_details: Dict,
    numerical_prob_safe: float,
    numerical_confidence: float = 0.85,
    numerical_details: Optional[Dict] = None,
    numerical_threshold: float = 0.38,
) -> FusionResult:
    """
    Fuse Vision AI and Numerical ML assessments into a single decision.

    Parameters
    ----------
    vision_prob_safe : float
        Vision AI's estimate of P(potable). Derived from visual quality score.
    vision_confidence : float
        Vision AI's self-reported confidence (0–1).
    vision_details : dict
        Full vision assessment details for display.
    numerical_prob_safe : float
        ML model's predict_proba output for the positive (potable) class.
    numerical_confidence : float
        Confidence in numerical model (default 0.85, can be reduced if
        input had many missing features).
    numerical_details : dict, optional
        ML prediction details.
    numerical_threshold : float
        Original ML decision threshold.

    Returns
    -------
    FusionResult
    """
    if numerical_details is None:
        numerical_details = {}

    # Vision modality evidence
    vision_ev = ModalityEvidence(
        source="Vision AI (Computer Vision)",
        probability_safe=vision_prob_safe,
        confidence=vision_confidence,
        details=vision_details,
    )

    # Numerical modality evidence
    num_ev = ModalityEvidence(
        source="Numerical ML (Soft-Voting Ensemble)",
        probability_safe=numerical_prob_safe,
        confidence=numerical_confidence,
        details=numerical_details,
    )

    # Individual decisions
    vision_safe = vision_prob_safe >= 0.55
    numerical_safe = numerical_prob_safe >= numerical_threshold

    # Compute adjusted weights
    w_v, w_n = _adjusted_weights(vision_confidence, numerical_confidence)

    # Dempster-Shafer fusion
    fused_prob, fused_conf = _dempster_shafer_combine(
        vision_prob_safe, vision_confidence,
        numerical_prob_safe, numerical_confidence,
    )

    # Decision with uncertainty band
    if fused_prob >= UNCERTAIN_HIGH:
        decision = "SAFE"
    elif fused_prob <= UNCERTAIN_LOW:
        decision = "UNSAFE"
    else:
        # In the gray zone — check agreement
        if vision_safe and numerical_safe:
            decision = "SAFE"
        elif not vision_safe and not numerical_safe:
            decision = "UNSAFE"
        else:
            decision = "INCONCLUSIVE"

    # Risk level
    if fused_prob >= 0.75:
        risk = "Low"
    elif fused_prob >= 0.50:
        risk = "Medium"
    elif fused_prob >= 0.30:
        risk = "High"
    else:
        risk = "Critical"

    # Agreement analysis
    agreement_label, agreement_score, disagree_details = _assess_agreement(
        vision_prob_safe, numerical_prob_safe, vision_safe, numerical_safe
    )

    # Recommendation
    rec, actions = _build_recommendation(
        fused_prob, decision, agreement_label, risk, vision_safe, numerical_safe
    )

    # Summary
    summary = (
        f"Multimodal fusion ({w_v:.0%} Vision + {w_n:.0%} Numerical) yields "
        f"a fused potability probability of {fused_prob:.1%} "
        f"(Decision: {decision}, Confidence: {fused_conf:.0%}). "
        f"Modality agreement: {agreement_label}."
    )

    return FusionResult(
        vision_evidence=vision_ev,
        numerical_evidence=num_ev,
        fused_probability=round(fused_prob, 4),
        fused_decision=decision,
        fused_confidence=round(fused_conf, 3),
        fused_risk_level=risk,
        modality_agreement=agreement_label,
        agreement_score=agreement_score,
        disagreement_details=disagree_details,
        fusion_method="Weighted Dempster-Shafer Evidence Fusion",
        vision_weight=round(w_v, 3),
        numerical_weight=round(w_n, 3),
        safe_threshold=SAFE_THRESHOLD,
        uncertain_low=UNCERTAIN_LOW,
        uncertain_high=UNCERTAIN_HIGH,
        summary=summary,
        recommendation=rec,
        action_items=actions,
    )
