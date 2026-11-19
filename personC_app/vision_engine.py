"""
Vision AI Engine — Water Sample Image Analysis
================================================
Extracts visual biomarkers from water sample photographs using
classical computer-vision pipelines (colour analysis, clarity metrics,
contaminant detection) and maps them to water-quality indicators.

Works with Pillow + NumPy only (no OpenCV dependency).
"""

import numpy as np
from PIL import Image, ImageFilter, ImageStat
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import io, math


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ColourProfile:
    """HSV-space colour analysis of a water sample."""
    dominant_hue: float          # 0–360
    dominant_saturation: float   # 0–1
    mean_value: float            # 0–1  (brightness)
    blue_ratio: float            # fraction of blue-ish pixels
    green_ratio: float           # fraction of green-ish pixels (algae indicator)
    brown_ratio: float           # fraction of brown/yellow pixels (sediment)
    colourless_ratio: float      # near-grey / transparent pixels
    colour_label: str            # human-readable: "Clear", "Turbid Brown", etc.


@dataclass
class ClarityMetrics:
    """Image-based turbidity and clarity estimation."""
    sharpness: float             # Laplacian variance (higher = sharper)
    contrast: float              # RMS contrast (0–1)
    entropy: float               # Shannon entropy of luminance
    edge_density: float          # fraction of strong-edge pixels
    estimated_turbidity_ntu: float  # mapped NTU estimate
    clarity_label: str           # "Crystal Clear", "Slightly Hazy", etc.


@dataclass
class ContaminantIndicators:
    """Visual anomaly / contaminant markers."""
    algae_risk: float            # 0–1  (green pigment intensity)
    sediment_risk: float         # 0–1  (brown/ochre pigment)
    foam_risk: float             # 0–1  (white cluster detection)
    discolouration_risk: float   # 0–1  (deviation from clear-water norm)
    particle_count_estimate: int # detected high-contrast particles
    anomalies: List[str]         # human-readable anomaly list


@dataclass
class VisionAssessment:
    """Full Vision AI assessment result for a water sample image."""
    colour: ColourProfile
    clarity: ClarityMetrics
    contaminants: ContaminantIndicators
    visual_quality_score: float  # 0–100  composite score
    visual_potability: bool      # True if vision says safe
    confidence: float            # 0–1 confidence in visual assessment
    risk_level: str              # "Low", "Medium", "High", "Critical"
    summary: str                 # human-readable summary paragraph
    warnings: List[str]          # list of warnings


# ═══════════════════════════════════════════════════════════════════════
# COLOUR HELPERS (pure NumPy, no OpenCV)
# ═══════════════════════════════════════════════════════════════════════

def _rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) uint8 RGB array → (H, W, 3) float HSV.
    H in [0, 360], S in [0, 1], V in [0, 1].
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.max(rgb_f, axis=-1)
    cmin = np.min(rgb_f, axis=-1)
    delta = cmax - cmin

    # Hue
    hue = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    hue[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Saturation
    sat = np.where(cmax > 0, delta / cmax, 0.0)

    return np.stack([hue, sat, cmax], axis=-1)


def _classify_hue_ranges(hsv: np.ndarray, sat_thresh: float = 0.08):
    """Classify pixels into colour buckets.  Returns ratio dict."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    n = h.size

    colourless = (s < sat_thresh) | (v < 0.10)
    chromatic = ~colourless

    # Blue: H in [190, 260]
    blue = chromatic & (h >= 190) & (h <= 260)
    # Green: H in [75, 170]
    green = chromatic & (h >= 75) & (h <= 170)
    # Brown/Yellow/Ochre: H in [15, 50]
    brown = chromatic & (h >= 15) & (h <= 50)
    # Red: H < 15 or H > 340
    red = chromatic & ((h < 15) | (h > 340))

    return {
        "blue": blue.sum() / n,
        "green": green.sum() / n,
        "brown": brown.sum() / n,
        "red": red.sum() / n,
        "colourless": colourless.sum() / n,
    }


def _dominant_hue_sat(hsv: np.ndarray, sat_thresh: float = 0.08):
    """Find dominant hue among chromatic pixels via circular mean."""
    s = hsv[..., 1].ravel()
    h = hsv[..., 0].ravel()
    mask = s >= sat_thresh
    if mask.sum() < 10:
        return 0.0, 0.0  # essentially colourless

    h_rad = np.deg2rad(h[mask])
    sin_mean = np.mean(np.sin(h_rad))
    cos_mean = np.mean(np.cos(h_rad))
    dominant_h = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360
    dominant_s = float(np.mean(s[mask]))
    return dominant_h, dominant_s


# ═══════════════════════════════════════════════════════════════════════
# CLARITY / TURBIDITY (Pillow filters)
# ═══════════════════════════════════════════════════════════════════════

def _compute_sharpness(grey_arr: np.ndarray) -> float:
    """Laplacian variance as sharpness proxy (3×3 kernel via convolution)."""
    # Approximate Laplacian via second-order differences
    lap_x = np.diff(grey_arr.astype(np.float32), n=2, axis=1)
    lap_y = np.diff(grey_arr.astype(np.float32), n=2, axis=0)
    # Variance of the Laplacian
    var_x = np.var(lap_x) if lap_x.size > 0 else 0
    var_y = np.var(lap_y) if lap_y.size > 0 else 0
    return float(var_x + var_y)


def _rms_contrast(grey_arr: np.ndarray) -> float:
    """RMS contrast in [0, 1]."""
    f = grey_arr.astype(np.float32) / 255.0
    return float(np.std(f))


def _shannon_entropy(grey_arr: np.ndarray) -> float:
    """Shannon entropy of luminance histogram."""
    hist, _ = np.histogram(grey_arr, bins=256, range=(0, 255))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def _edge_density(img_pil: Image.Image) -> float:
    """Fraction of strong-edge pixels (FIND_EDGES filter)."""
    edges = img_pil.convert("L").filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges)
    return float((arr > 60).sum() / arr.size)


def _map_to_ntu(sharpness: float, contrast: float, edge_density: float,
                colour_ratios: dict = None) -> float:
    """Heuristic mapping of image metrics → estimated NTU (0–100+).

    Uses both image-clarity metrics AND colour cues (brown/green content
    strongly correlates with real-world turbidity).  This dual approach
    works for both real photographs and synthetic demo images.

    Calibration assumptions (tuneable):
      - Crystal clear water: low colour contamination, any sharpness → NTU ≈ 0.5–3
      - Moderately turbid:   some brown/green tint                   → NTU ≈ 5–20
      - Very turbid:         heavy brown/green, low contrast         → NTU ≈ 30–100+
    """
    # --- Image-metric channel ---
    clarity_norm = min(sharpness / 500.0, 1.0)
    contrast_norm = min(contrast / 0.20, 1.0)
    edge_norm = min(edge_density / 0.10, 1.0)
    metric_clarity = 0.40 * clarity_norm + 0.35 * contrast_norm + 0.25 * edge_norm

    # --- Colour channel (most discriminative for water) ---
    colour_clarity = 1.0  # assume clear if no info
    if colour_ratios is not None:
        brown = colour_ratios.get("brown", 0)
        green = colour_ratios.get("green", 0)
        red = colour_ratios.get("red", 0)
        colourless = colour_ratios.get("colourless", 0)
        blue = colour_ratios.get("blue", 0)

        # Clear water is colourless or blue-tinted, with minimal brown/green
        contamination = brown + green * 0.9 + red * 0.8
        clearness = colourless + blue * 0.8
        colour_clarity = max(0.0, min(1.0, clearness - contamination * 1.5))

    # Blend both channels (colour dominates for water assessment)
    combined = 0.35 * metric_clarity + 0.65 * colour_clarity
    # Map to NTU: 1 = crystal clear (0.1 NTU), 0 = opaque (100 NTU)
    ntu = max(0.1, 100.0 * (1.0 - combined) ** 1.8)
    return round(ntu, 1)


def _clarity_label(ntu: float) -> str:
    if ntu < 1.0:
        return "Crystal Clear"
    elif ntu < 5.0:
        return "Clear"
    elif ntu < 15.0:
        return "Slightly Hazy"
    elif ntu < 30.0:
        return "Turbid"
    elif ntu < 60.0:
        return "Very Turbid"
    else:
        return "Opaque / Heavily Turbid"


# ═══════════════════════════════════════════════════════════════════════
# CONTAMINANT DETECTION
# ═══════════════════════════════════════════════════════════════════════

def _detect_foam(rgb: np.ndarray) -> float:
    """Detect white/foam clusters: pixels that are bright AND achromatic.
    True foam is near-white (all channels close, all high).
    Blue-tinted clear water (R≈215, G≈230, B≈250) must NOT trigger this.
    """
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    brightness = np.max(rgb, axis=-1)
    # Pairwise channel differences
    rg_diff = np.abs(r - g)
    rb_diff = np.abs(r - b)
    gb_diff = np.abs(g - b)
    max_diff = np.maximum(np.maximum(rg_diff, rb_diff), gb_diff)
    # Foam: very bright AND all channels within 8 of each other (truly white/grey)
    white_mask = (brightness > 230) & (max_diff < 8)
    ratio = white_mask.sum() / brightness.size

    if ratio > 0.03:
        return min((ratio - 0.03) * 8, 1.0)
    return 0.0


def _detect_particles(grey_arr: np.ndarray) -> int:
    """Very rough dark-particle count via thresholding."""
    # Look for small very-dark spots against a lighter background
    median = np.median(grey_arr)
    if median < 80:
        return 0  # image is overall dark, can't reliably detect

    dark_thresh = max(30, median - 80)
    dark_pixels = (grey_arr < dark_thresh).astype(np.uint8)
    # Count connected regions roughly via run-length in rows
    transitions = np.diff(dark_pixels, axis=1)
    starts = (transitions == 1).sum()
    return int(starts)


# ═══════════════════════════════════════════════════════════════════════
# COLOUR LABEL
# ═══════════════════════════════════════════════════════════════════════

def _colour_label(ratios: dict, dom_hue: float, dom_sat: float) -> str:
    """Human-readable colour description."""
    if ratios["colourless"] > 0.70:
        return "Clear / Colourless"
    if ratios["blue"] > 0.30:
        return "Blue-tinted (mineral or copper trace)"
    if ratios["green"] > 0.25:
        return "Green-tinted (possible algae / biological)"
    if ratios["brown"] > 0.25:
        return "Brown / Ochre (sediment or iron)"
    if ratios["red"] > 0.15:
        return "Reddish (iron oxide / contamination)"
    if dom_sat < 0.12:
        return "Near-colourless (slight tint)"
    return f"Tinted (dominant hue ≈ {dom_hue:.0f}°)"


# ═══════════════════════════════════════════════════════════════════════
# COMPOSITE SCORING
# ═══════════════════════════════════════════════════════════════════════

def _compute_visual_score(colour: ColourProfile,
                          clarity: ClarityMetrics,
                          contam: ContaminantIndicators) -> Tuple[float, bool, float, str]:
    """Compute composite visual quality score (0–100), potability, confidence, risk level."""

    # Clarity sub-score (40 points)
    ntu = clarity.estimated_turbidity_ntu
    if ntu < 1:
        clarity_score = 40.0
    elif ntu < 5:
        clarity_score = 35.0
    elif ntu < 15:
        clarity_score = 25.0
    elif ntu < 30:
        clarity_score = 15.0
    elif ntu < 60:
        clarity_score = 7.0
    else:
        clarity_score = 0.0

    # Colour sub-score (30 points)
    colour_score = 30.0
    if colour.green_ratio > 0.20:
        colour_score -= 15 * min(colour.green_ratio / 0.40, 1.0)
    if colour.brown_ratio > 0.15:
        colour_score -= 15 * min(colour.brown_ratio / 0.40, 1.0)
    colour_score = max(0, colour_score)

    # Contaminant sub-score (30 points)
    contam_score = 30.0
    contam_score -= 10 * contam.algae_risk
    contam_score -= 10 * contam.sediment_risk
    contam_score -= 5 * contam.foam_risk
    contam_score -= 5 * contam.discolouration_risk
    contam_score = max(0, contam_score)

    total = clarity_score + colour_score + contam_score
    total = max(0, min(100, total))

    # Confidence: based on how much colour info we have
    if colour.colourless_ratio > 0.90:
        confidence = 0.50  # mostly grey/white → less visual info
    elif colour.colourless_ratio > 0.60:
        confidence = 0.65
    else:
        confidence = 0.80

    # Risk level
    if total >= 75:
        risk = "Low"
    elif total >= 50:
        risk = "Medium"
    elif total >= 30:
        risk = "High"
    else:
        risk = "Critical"

    potable = total >= 55

    return total, potable, confidence, risk


def _build_summary(colour: ColourProfile,
                   clarity: ClarityMetrics,
                   contam: ContaminantIndicators,
                   score: float, risk: str) -> Tuple[str, List[str]]:
    """Generate human-readable summary and warnings list."""
    parts = []
    warnings = []

    # Clarity
    parts.append(f"Visual turbidity estimated at ~{clarity.estimated_turbidity_ntu:.1f} NTU "
                 f"({clarity.clarity_label}).")

    if clarity.estimated_turbidity_ntu > 5:
        warnings.append(f"⚠️ Elevated visual turbidity ({clarity.estimated_turbidity_ntu:.1f} NTU) — "
                        "WHO guideline is < 5 NTU for drinking water.")

    # Colour
    parts.append(f"Colour profile: {colour.colour_label}.")
    if colour.green_ratio > 0.20:
        warnings.append("🟢 Significant green pigmentation detected — possible algal bloom or "
                        "biological contamination.")
    if colour.brown_ratio > 0.20:
        warnings.append("🟤 Brown/ochre colouration detected — possible sediment, iron, or "
                        "organic matter.")

    # Contaminants
    if contam.algae_risk > 0.3:
        warnings.append(f"🦠 Algae contamination risk: {contam.algae_risk:.0%}")
    if contam.sediment_risk > 0.3:
        warnings.append(f"🏔️ Sediment contamination risk: {contam.sediment_risk:.0%}")
    if contam.foam_risk > 0.2:
        warnings.append(f"🫧 Surface foam / surfactant detected (risk: {contam.foam_risk:.0%})")
    if contam.particle_count_estimate > 50:
        warnings.append(f"🔬 ~{contam.particle_count_estimate} suspended particles detected in image.")

    parts.append(f"Overall visual quality score: {score:.0f}/100 (Risk: {risk}).")

    summary = " ".join(parts)
    return summary, warnings


# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def analyse_water_image(image: Image.Image,
                        target_size: int = 512) -> VisionAssessment:
    """
    Full Vision AI analysis pipeline for a water sample image.

    Parameters
    ----------
    image : PIL.Image.Image
        The uploaded water sample photograph.
    target_size : int
        Resize longest edge to this for consistent processing.

    Returns
    -------
    VisionAssessment
        Complete visual assessment including colour, clarity,
        contaminants, composite score, and human-readable summary.
    """
    # ── Preprocessing ────────────────────────────────────────────────
    img = image.convert("RGB")
    # Resize for consistent analysis
    w, h = img.size
    scale = target_size / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    rgb = np.array(img)
    grey = np.array(img.convert("L"))

    # ── 1. Colour Analysis ───────────────────────────────────────────
    hsv = _rgb_to_hsv_array(rgb)
    ratios = _classify_hue_ranges(hsv)
    dom_hue, dom_sat = _dominant_hue_sat(hsv)
    mean_v = float(np.mean(hsv[..., 2]))

    colour = ColourProfile(
        dominant_hue=round(dom_hue, 1),
        dominant_saturation=round(dom_sat, 3),
        mean_value=round(mean_v, 3),
        blue_ratio=round(ratios["blue"], 3),
        green_ratio=round(ratios["green"], 3),
        brown_ratio=round(ratios["brown"], 3),
        colourless_ratio=round(ratios["colourless"], 3),
        colour_label=_colour_label(ratios, dom_hue, dom_sat),
    )

    # ── 2. Clarity / Turbidity ───────────────────────────────────────
    sharpness = _compute_sharpness(grey)
    contrast = _rms_contrast(grey)
    entropy = _shannon_entropy(grey)
    edge_d = _edge_density(img)
    ntu = _map_to_ntu(sharpness, contrast, edge_d, colour_ratios=ratios)

    clarity = ClarityMetrics(
        sharpness=round(sharpness, 2),
        contrast=round(contrast, 4),
        entropy=round(entropy, 2),
        edge_density=round(edge_d, 4),
        estimated_turbidity_ntu=ntu,
        clarity_label=_clarity_label(ntu),
    )

    # ── 3. Contaminant Indicators ────────────────────────────────────
    algae_risk = min(colour.green_ratio / 0.35, 1.0)
    sediment_risk = min(colour.brown_ratio / 0.35, 1.0)
    foam = _detect_foam(rgb)
    particles = _detect_particles(grey)

    # Discolouration: how far from "ideal clear water" (high V, low S)
    ideal_v, ideal_s = 0.75, 0.05
    discol = math.sqrt((mean_v - ideal_v) ** 2 + (dom_sat - ideal_s) ** 2)
    discol_risk = min(discol / 0.5, 1.0)

    contam = ContaminantIndicators(
        algae_risk=round(algae_risk, 3),
        sediment_risk=round(sediment_risk, 3),
        foam_risk=round(foam, 3),
        discolouration_risk=round(discol_risk, 3),
        particle_count_estimate=particles,
        anomalies=[],
    )

    # Build anomaly list
    if algae_risk > 0.3:
        contam.anomalies.append("Algal pigmentation")
    if sediment_risk > 0.3:
        contam.anomalies.append("Sediment/iron colouration")
    if foam > 0.2:
        contam.anomalies.append("Surface foam / surfactants")
    if particles > 50:
        contam.anomalies.append(f"~{particles} suspended particles")

    # ── 4. Composite Score ───────────────────────────────────────────
    score, potable, confidence, risk = _compute_visual_score(colour, clarity, contam)

    # ── 5. Summary ───────────────────────────────────────────────────
    summary, warnings = _build_summary(colour, clarity, contam, score, risk)

    return VisionAssessment(
        colour=colour,
        clarity=clarity,
        contaminants=contam,
        visual_quality_score=round(score, 1),
        visual_potability=potable,
        confidence=round(confidence, 3),
        risk_level=risk,
        summary=summary,
        warnings=warnings,
    )
