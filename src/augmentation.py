from PIL import Image, ImageFilter
import numpy as np
import io
import torchvision.transforms as T

def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def resize_downup(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)

def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 1) * 255
    return Image.fromarray(noisy.astype(np.uint8))

def color_jitter(img: Image.Image, strength: float = 0.2) -> Image.Image:
    return T.ColorJitter(brightness=strength, contrast=strength, saturation=strength)(img)

def center_crop_pct(img: Image.Image, pct: float = 0.8) -> Image.Image:
    w, h = img.size
    return T.CenterCrop((int(h*pct), int(w*pct)))(img)


import random

def apply_augmentation_strategy(img: Image.Image, strategy: str) -> Image.Image:
    transforms = {
        "jpeg":  lambda im: jpeg_compress(im, random.choice([90,70,50,30])),
        "blur":  lambda im: gaussian_blur(im, random.choice([0.5,1.0,2.0])),
        "resize":lambda im: resize_downup(im, random.choice([0.5,0.25])),
        "noise": lambda im: gaussian_noise(im, random.choice([0.02,0.05,0.10])),
        "color": lambda im: color_jitter(im, 0.2),
        "crop":  lambda im: center_crop_pct(im, 0.8),
    }
    names = list(transforms.keys())

    # Allow direct calls like apply_augmentation_strategy(img, "jpeg")
    if strategy in transforms:
        return transforms[strategy](img)

    # Allow case-insensitive transform names as well
    normalized = strategy.lower()
    if normalized in transforms:
        return transforms[normalized](img)

    if strategy == "R0":
        return img  # no augmentation — control
    elif strategy == "R1":
        return transforms[random.choice(names)](img)  # exactly one, random type
    elif strategy == "R2":
        chosen = ["jpeg", "blur", "resize"]  # your fixed pick — reasonable "real-world" combo
        for name in chosen:
            img = transforms[name](img)
        return img
    elif strategy == "R3":
        k = random.choice([1, 2])
        for name in random.sample(names, k):
            img = transforms[name](img)
        return img
    else:
        raise ValueError(strategy)