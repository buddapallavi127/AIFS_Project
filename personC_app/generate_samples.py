"""
Generate synthetic water sample images for demonstration.
Creates realistic-looking water sample photographs using
Pillow drawing primitives — no external images needed.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os, pathlib


def _add_noise(arr: np.ndarray, sigma: float = 5) -> np.ndarray:
    noise = np.random.normal(0, sigma, arr.shape).astype(np.int16)
    return np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def generate_clear_water(size=(400, 400)) -> Image.Image:
    """Crystal-clear water sample — light blue, high clarity."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    # Light blue base with subtle gradient
    for y in range(size[0]):
        t = y / size[0]
        arr[y, :, 0] = int(200 + 30 * t)   # R
        arr[y, :, 1] = int(220 + 20 * t)   # G
        arr[y, :, 2] = int(240 + 10 * t)   # B
    arr = _add_noise(arr, sigma=3)
    img = Image.fromarray(arr)
    # Add subtle refraction pattern
    draw = ImageDraw.Draw(img)
    for _ in range(15):
        x = np.random.randint(20, size[1] - 20)
        y = np.random.randint(20, size[0] - 20)
        r = np.random.randint(5, 20)
        draw.ellipse([x-r, y-r, x+r, y+r],
                     fill=None,
                     outline=(210, 230, 250, 60))
    return img.filter(ImageFilter.GaussianBlur(0.5))


def generate_turbid_water(size=(400, 400)) -> Image.Image:
    """Turbid/murky water — brownish, low clarity."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for y in range(size[0]):
        t = y / size[0]
        arr[y, :, 0] = int(160 + 40 * t)   # R (brown)
        arr[y, :, 1] = int(120 + 30 * t)   # G
        arr[y, :, 2] = int(70 + 20 * t)    # B
    arr = _add_noise(arr, sigma=12)
    img = Image.fromarray(arr)
    # Add sediment particles
    draw = ImageDraw.Draw(img)
    for _ in range(80):
        x = np.random.randint(5, size[1] - 5)
        y = np.random.randint(5, size[0] - 5)
        r = np.random.randint(1, 4)
        c = np.random.randint(80, 140)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(c, c-20, c-40))
    return img.filter(ImageFilter.GaussianBlur(1.5))


def generate_algae_water(size=(400, 400)) -> Image.Image:
    """Algae-contaminated water — green tint, particles."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for y in range(size[0]):
        t = y / size[0]
        arr[y, :, 0] = int(80 + 20 * t)
        arr[y, :, 1] = int(150 + 50 * t)   # Green dominant
        arr[y, :, 2] = int(60 + 20 * t)
    arr = _add_noise(arr, sigma=8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    # Algae clusters
    for _ in range(40):
        x = np.random.randint(10, size[1] - 10)
        y = np.random.randint(10, size[0] - 10)
        r = np.random.randint(3, 12)
        g_val = np.random.randint(100, 200)
        draw.ellipse([x-r, y-r, x+r, y+r],
                     fill=(30, g_val, 20), outline=(20, g_val-20, 10))
    return img.filter(ImageFilter.GaussianBlur(1.0))


def generate_iron_water(size=(400, 400)) -> Image.Image:
    """Iron/rust contaminated — reddish-orange tint."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for y in range(size[0]):
        t = y / size[0]
        arr[y, :, 0] = int(190 + 40 * t)   # Red dominant
        arr[y, :, 1] = int(100 + 20 * t)
        arr[y, :, 2] = int(50 + 15 * t)
    arr = _add_noise(arr, sigma=10)
    img = Image.fromarray(arr)
    # Rust particles
    draw = ImageDraw.Draw(img)
    for _ in range(50):
        x = np.random.randint(5, size[1] - 5)
        y = np.random.randint(5, size[0] - 5)
        r = np.random.randint(1, 5)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(160, 60, 20))
    return img.filter(ImageFilter.GaussianBlur(1.2))


def generate_foam_water(size=(400, 400)) -> Image.Image:
    """Water with surface foam / detergent contamination."""
    # Base: slightly cloudy blue-grey
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for y in range(size[0]):
        t = y / size[0]
        arr[y, :, 0] = int(170 + 30 * t)
        arr[y, :, 1] = int(180 + 25 * t)
        arr[y, :, 2] = int(195 + 20 * t)
    arr = _add_noise(arr, sigma=5)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    # Foam bubbles in upper third
    for _ in range(60):
        x = np.random.randint(5, size[1] - 5)
        y = np.random.randint(5, size[0] // 3)
        r = np.random.randint(5, 18)
        draw.ellipse([x-r, y-r, x+r, y+r],
                     fill=(240, 240, 245), outline=(220, 220, 225))
    return img.filter(ImageFilter.GaussianBlur(0.8))


SAMPLE_GENERATORS = {
    "clear_water.png": ("💧 Clear / Potable Water", generate_clear_water),
    "turbid_water.png": ("🟤 Turbid / Sediment-laden Water", generate_turbid_water),
    "algae_water.png": ("🟢 Algae-contaminated Water", generate_algae_water),
    "iron_water.png": ("🔴 Iron/Rust-contaminated Water", generate_iron_water),
    "foam_water.png": ("🫧 Foam / Detergent-contaminated Water", generate_foam_water),
}


def generate_all_samples(output_dir: str) -> dict:
    """Generate all sample images and return {filename: description}."""
    os.makedirs(output_dir, exist_ok=True)
    result = {}
    for fname, (desc, gen_fn) in SAMPLE_GENERATORS.items():
        path = os.path.join(output_dir, fname)
        img = gen_fn()
        img.save(path)
        result[fname] = desc
    return result


if __name__ == "__main__":
    out = pathlib.Path(__file__).parent / "sample_images"
    info = generate_all_samples(str(out))
    for fname, desc in info.items():
        print(f"  ✓ {fname}: {desc}")
    print(f"\nGenerated {len(info)} sample images in {out}")
