"""Standalone low-light generator matching the surviving original algorithm.

Input may be one image or a dataset root containing train/high and test/high.
The default settings reproduce the original synthesis and its continuous RNG.
Use --size-multiple 1 and --seed-mode per-image for later full-image variants.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


BASE_CONFIG = {
    "gamma_range": (1.55, 2.15),
    "brightness_range": (0.38, 0.70),
    "illumination_base_range": (0.38, 0.62),
    "illumination_spot_count": (1, 3),
    "illumination_spot_gain_range": (0.10, 0.22),
    "illumination_lamp_prob": 0.8,
    "illumination_lamp_gain_range": (0.14, 0.30),
    "illumination_sigma_x_range": (0.12, 0.30),
    "illumination_sigma_y_range": (0.12, 0.34),
    "illumination_blur_sigma_range": (9.0, 18.0),
    "illumination_clip_range": (0.16, 0.82),
    "vignette_strength_range": (0.24, 0.46),
    "gaussian_luma_noise_std_range": (0.0025, 0.0100),
    "gaussian_chroma_noise_std_range": (0.00015, 0.00080),
    "gaussian_dark_scale_range": (0.6, 1.4),
    "poisson_levels_range": (36.0, 68.0),
    "poisson_dark_scale_range": (0.35, 1.0),
    "noise_darkness_gamma_range": (1.2, 1.8),
    "red_speckle_max_excess": 0.05,
    # OpenCV channel order: B, G, R.
    "color_profiles": {
        "cold": ((1.00, 1.08), (0.95, 1.02), (0.86, 0.96)),
        "greenish": ((0.90, 0.99), (1.01, 1.09), (0.87, 0.96)),
        "yellow_green": ((0.84, 0.94), (1.00, 1.08), (0.94, 1.03)),
    },
    "blur_prob": 0.25,
    "blur_kernel_choices": (3, 5),
    "blur_sigma_range": (0.4, 1.1),
    "jpeg_quality": 95,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def make_config(brightness_scale: float, noise_scale: float, blur_prob: float) -> dict:
    cfg = {**BASE_CONFIG, "blur_prob": blur_prob}
    cfg["brightness_range"] = tuple(v * brightness_scale for v in BASE_CONFIG["brightness_range"])
    for key in ("gaussian_luma_noise_std_range", "gaussian_chroma_noise_std_range"):
        cfg[key] = tuple(v * noise_scale for v in BASE_CONFIG[key])
    cfg["poisson_levels_range"] = tuple(v / noise_scale**2 for v in BASE_CONFIG["poisson_levels_range"])
    return cfg


def sample_color_gains(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    names = list(cfg["color_profiles"])
    ranges = cfg["color_profiles"][names[int(rng.integers(0, len(names)))]]
    return np.array([rng.uniform(lo, hi) for lo, hi in ranges], dtype=np.float32)


def generate_illumination_map(height: int, width: int, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= max(width - 1, 1)
    yy /= max(height - 1, 1)
    illumination = np.full((height, width), rng.uniform(*cfg["illumination_base_range"]), dtype=np.float32)

    center_x, center_y = rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65)
    radius_x, radius_y = rng.uniform(0.75, 1.05), rng.uniform(0.75, 1.10)
    distance = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
    strength = rng.uniform(*cfg["vignette_strength_range"])
    illumination *= np.clip(1.0 - strength * distance, 0.55, 1.0)

    count = int(rng.integers(cfg["illumination_spot_count"][0], cfg["illumination_spot_count"][1] + 1))
    for idx in range(count):
        lamp = idx == 0 and rng.random() < cfg["illumination_lamp_prob"]
        if lamp:
            gain = rng.uniform(*cfg["illumination_lamp_gain_range"])
            mu_x, mu_y = rng.uniform(0.25, 0.75), rng.uniform(0.12, 0.68)
        else:
            gain = rng.uniform(*cfg["illumination_spot_gain_range"])
            mu_x, mu_y = rng.uniform(0.10, 0.90), rng.uniform(0.10, 0.90)
        sigma_x = rng.uniform(*cfg["illumination_sigma_x_range"])
        sigma_y = rng.uniform(*cfg["illumination_sigma_y_range"])
        gaussian = np.exp(-(((xx - mu_x) ** 2 / (2.0 * sigma_x**2)) +
                            ((yy - mu_y) ** 2 / (2.0 * sigma_y**2))))
        illumination += gain * gaussian.astype(np.float32)

    blur_sigma = rng.uniform(*cfg["illumination_blur_sigma_range"])
    illumination = cv2.GaussianBlur(illumination, (0, 0), blur_sigma)
    return np.clip(illumination, *cfg["illumination_clip_range"])[..., None]


def suppress_red_speckles(image: np.ndarray, cfg: dict) -> np.ndarray:
    result = np.clip(image, 0.0, 1.0).copy()
    red_limit = np.maximum(result[..., 0], result[..., 1]) + cfg["red_speckle_max_excess"]
    result[..., 2] = np.minimum(result[..., 2], red_limit)
    return result


def add_dark_dependent_noise(image: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    luminance = 0.114 * image[..., 0] + 0.587 * image[..., 1] + 0.299 * image[..., 2]
    darkness = np.clip(1.0 - luminance, 0.0, 1.0)
    darkness_weight = darkness ** rng.uniform(*cfg["noise_darkness_gamma_range"])

    luma_std = rng.uniform(*cfg["gaussian_luma_noise_std_range"])
    chroma_std = rng.uniform(*cfg["gaussian_chroma_noise_std_range"])
    gaussian_scale = rng.uniform(*cfg["gaussian_dark_scale_range"])
    noise_strength = 0.35 + gaussian_scale * darkness_weight[..., None]
    height, width = image.shape[:2]
    luma_noise = rng.normal(0.0, 1.0, (height, width, 1)).astype(np.float32)
    luma_noise *= luma_std * noise_strength
    chroma_noise = np.zeros_like(image, dtype=np.float32)
    chroma_noise[..., 0] = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    chroma_noise[..., 1] = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    chroma_noise[..., 2] = 0.0
    chroma_noise *= chroma_std * noise_strength

    levels = rng.uniform(*cfg["poisson_levels_range"])
    poisson_scale = rng.uniform(*cfg["poisson_dark_scale_range"])
    poisson_base = np.clip(luminance, 0.0, 1.0)
    poisson_sample = rng.poisson(poisson_base * levels).astype(np.float32)
    poisson_sample /= levels
    poisson_noise = (poisson_sample - poisson_base)[..., None]
    poisson_noise *= 0.25 + poisson_scale * darkness_weight[..., None]

    noisy = image + luma_noise + chroma_noise + poisson_noise
    return np.clip(suppress_red_speckles(noisy, cfg), 0.0, 1.0)


def synthesize_lowlight(image_bgr: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    image = image_bgr.astype(np.float32) / 255.0
    gamma = rng.uniform(*cfg["gamma_range"])
    lowlight = np.power(np.clip(image, 0.0, 1.0), gamma)
    lowlight *= rng.uniform(*cfg["brightness_range"])
    lowlight *= generate_illumination_map(image.shape[0], image.shape[1], rng, cfg)
    lowlight *= sample_color_gains(rng, cfg).reshape(1, 1, 3)
    lowlight = np.clip(lowlight, 0.0, 1.0)
    if rng.random() < cfg["blur_prob"]:
        kernels = cfg["blur_kernel_choices"]
        size = int(kernels[int(rng.integers(0, len(kernels)))])
        lowlight = cv2.GaussianBlur(lowlight, (size, size), rng.uniform(*cfg["blur_sigma_range"]))
    lowlight = add_dark_dependent_noise(lowlight, rng, cfg)
    return np.clip(lowlight * 255.0, 0.0, 255.0).astype(np.uint8)


def natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def align_image(image: np.ndarray, multiple: int) -> tuple[np.ndarray, list[int]]:
    height, width = image.shape[:2]
    target_h, target_w = height // multiple * multiple, width // multiple * multiple
    if target_h == 0 or target_w == 0:
        raise ValueError(f"Image dimensions must be at least {multiple} pixels")
    top, left = (height - target_h) // 2, (width - target_w) // 2
    return image[top:top + target_h, left:left + target_w], [left, top, left + target_w, top + target_h]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, cfg: dict) -> None:
    options = [cv2.IMWRITE_JPEG_QUALITY, cfg["jpeg_quality"]] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(path), image, options):
        raise OSError(f"Cannot write image: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="One image or dataset root with train/high and test/high")
    parser.add_argument("--output", type=Path, required=True, help="New output image or dataset root")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--seed-mode", choices=("stream", "per-image"), default="stream")
    parser.add_argument("--size-multiple", type=int, default=16, help="1 preserves original dimensions")
    parser.add_argument("--brightness-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--blur-prob", type=float, default=0.25)
    args = parser.parse_args()
    if args.seed < 0 or args.size_multiple < 1:
        parser.error("--seed must be nonnegative and --size-multiple must be positive")
    if not np.isfinite([args.brightness_scale, args.noise_scale, args.blur_prob]).all():
        parser.error("Scale and probability values must be finite")
    if args.brightness_scale <= 0 or args.noise_scale <= 0 or not 0 <= args.blur_prob <= 1:
        parser.error("Scales must be positive and blur probability must be in [0, 1]")
    source, output = args.source.resolve(), args.output.resolve()
    if not source.exists() or output.exists() or source == output:
        parser.error("Source must exist and output must not exist or equal source")
    if source.is_dir() and (source in output.parents or output in source.parents):
        parser.error("Source and output directories must not be nested")
    if source.is_file() and output.suffix.lower() not in IMAGE_EXTENSIONS:
        parser.error("Single-image output must end in .png, .jpg, or .jpeg")
    if source.is_file() and output.with_name(output.stem + "_generation.json").exists():
        parser.error("Single-image manifest already exists")

    cv2.setNumThreads(2)
    cfg = make_config(args.brightness_scale, args.noise_scale, args.blur_prob)
    stream_rng = np.random.default_rng(args.seed)
    records = []
    if source.is_file():
        image = read_image(source)
        aligned, box = align_image(image, args.size_multiple)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_image(output, synthesize_lowlight(aligned, stream_rng, cfg), cfg)
        records.append({"source": str(source), "output": str(output), "seed": args.seed,
                        "original_size_wh": [image.shape[1], image.shape[0]], "crop_box_ltrb": box})
        manifest = output.with_name(output.stem + "_generation.json")
    else:
        images = []
        for split in ("train", "test"):
            folder = source / split / "high"
            if not folder.is_dir():
                parser.error(f"Missing input directory: {folder}")
            files = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key)
            if not files:
                parser.error(f"No PNG/JPEG images in: {folder}")
            images.extend((split, p) for p in files)
        output.mkdir(parents=True)
        for index, (split, path) in enumerate(images):
            image = read_image(path)
            aligned, box = align_image(image, args.size_multiple)
            high_path = output / split / "high" / path.name
            low_path = output / split / "low" / path.name
            high_path.parent.mkdir(parents=True, exist_ok=True)
            low_path.parent.mkdir(parents=True, exist_ok=True)
            if aligned.shape == image.shape:
                shutil.copy2(path, high_path)
            else:
                write_image(high_path, aligned, cfg)
            rng = stream_rng if args.seed_mode == "stream" else np.random.default_rng(args.seed + index)
            write_image(low_path, synthesize_lowlight(aligned, rng, cfg), cfg)
            records.append({"split": split, "filename": path.name,
                            "seed": args.seed + index if args.seed_mode == "per-image" else None,
                            "original_size_wh": [image.shape[1], image.shape[0]],
                            "output_size_wh": [aligned.shape[1], aligned.shape[0]], "crop_box_ltrb": box})
            print(f"{index + 1}/{len(images)} {split}/{path.name}", flush=True)
        manifest = output / "generation_manifest.json"

    payload = {"config": cfg, "seed": args.seed, "seed_mode": args.seed_mode,
               "size_multiple": args.size_multiple, "images": records}
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(records)} image(s); manifest: {manifest}")


if __name__ == "__main__":
    main()
