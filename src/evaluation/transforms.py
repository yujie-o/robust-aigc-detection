"""
Robustness transformations for evaluation.

Implements the 6 challenge transformation categories at the respective severities. 
These are applied to raw PIL images BEFORE the model's AutoProcessor, 
so preprocessing/normalisation stays identical to what the model saw during training.
"""

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def identity(image: Image.Image) -> Image.Image:
    return image


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    small = image.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC
    )
    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def color_jitter(image: Image.Image, factor_range: float = 0.2) -> Image.Image:
    img = image.convert("RGB")
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-factor_range, factor_range)
        img = enhancer_cls(img).enhance(factor)
    return img


def center_crop(image: Image.Image, crop_fraction: float = 0.8) -> Image.Image:
    w, h = image.size
    new_w, new_h = int(w * crop_fraction), int(h * crop_fraction)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BICUBIC)


TRANSFORM_CONDITIONS = {
    "clean": identity,

    "jpeg_q90": lambda img: jpeg_compress(img, 90),
    "jpeg_q70": lambda img: jpeg_compress(img, 70),
    "jpeg_q50": lambda img: jpeg_compress(img, 50),
    "jpeg_q30": lambda img: jpeg_compress(img, 30),

    "blur_s0.5": lambda img: gaussian_blur(img, 0.5),
    "blur_s1.0": lambda img: gaussian_blur(img, 1.0),
    "blur_s2.0": lambda img: gaussian_blur(img, 2.0),

    "resize_0.5x": lambda img: resize_roundtrip(img, 0.5),
    "resize_0.25x": lambda img: resize_roundtrip(img, 0.25),

    "noise_s0.02": lambda img: gaussian_noise(img, 0.02),
    "noise_s0.05": lambda img: gaussian_noise(img, 0.05),
    "noise_s0.10": lambda img: gaussian_noise(img, 0.10),

    "color_jitter": lambda img: color_jitter(img, 0.2),

    "crop_80": lambda img: center_crop(img, 0.8),
}


CONDITION_GROUPS = {
    "clean": ["clean"],
    "jpeg": ["jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30"],
    "blur": ["blur_s0.5", "blur_s1.0", "blur_s2.0"],
    "resize": ["resize_0.5x", "resize_0.25x"],
    "noise": ["noise_s0.02", "noise_s0.05", "noise_s0.10"],
    "color": ["color_jitter"],
    "crop": ["crop_80"],
}