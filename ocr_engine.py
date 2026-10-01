import os
import hashlib
import logging

logging.getLogger("ppocr").setLevel(logging.ERROR)

from paddleocr import PaddleOCR
import cv2
import numpy as np

# Configurable via environment variables 
MAX_DIM = int(os.getenv("OCR_MAX_DIM", "800"))   # lower = faster, less accurate on tiny text
MIN_DIM = int(os.getenv("OCR_MIN_DIM", "600"))    # below this we upscale
HIGH_RES_MAX_DIM = int(os.getenv("OCR_HIGH_RES_MAX_DIM", "1400"))  # used only on retry pass
OCR_DEBUG = os.getenv("OCR_DEBUG", "0") == "1"    # gate verbose per-box logging

#  Singleton OCR instance 
_ocr_instance = None

def is_ocr_ready() -> bool:
    return _ocr_instance is not None

def get_ocr() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            device="cpu",
            enable_mkldnn=False,
            cpu_threads=1,
            det_limit_side_len=640,
            show_log=False
        )
    return _ocr_instance

#  In-process OCR result cache (keyed on fast structural hash) 
_ocr_cache: dict = {}
_cache_hits: int = 0
_CACHE_MAX = 50


def _rotate_np(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def correct_orientation(image):
    if image is None:
        return image

    h, w = image.shape[:2]
    scale = 400 / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image

    ocr = get_ocr()
    best_angle, best_score = 0, -1.0

    for angle in (0, 90, 180, 270):
        rotated = _rotate_np(small, angle)
        result = ocr.ocr(rotated, cls=False)
        lines = result[0] if result and result[0] else []
        score = len(lines)
        if angle == 0:
            score += 0.5
        if score > best_score:
            best_score = score
            best_angle = angle

    return _rotate_np(image, best_angle)

def get_cache_stats() -> dict:
    return {"size": len(_ocr_cache), "hits": _cache_hits}

def _fast_img_key(image: np.ndarray, max_dim: int):
    
    h, w = image.shape[:2]
    try:
        samples = (
            image[0, 0].tobytes(),
            image[h // 2, w // 2].tobytes(),
            image[-1, -1].tobytes(),
            image[0, -1].tobytes(),
            image[-1, 0].tobytes(),
        )
        return (h, w, image.dtype.str, max_dim, *samples)
    except Exception:
        return None

def run_ocr(image: np.ndarray, max_dim: int = None):
    
    import time

    total_start = time.time()
    global _cache_hits

    if image is None:
        return [], []

    effective_max_dim = max_dim if max_dim else MAX_DIM

    # Cache lookup 
    img_key = _fast_img_key(image, effective_max_dim)
    if img_key is not None and img_key in _ocr_cache:
        _cache_hits += 1
        return _ocr_cache[img_key]

    #  Resolution normalisation 
    h, w = image.shape[:2]
    upscale_factor = 1.0

    if h < MIN_DIM or w < MIN_DIM:
        upscale_factor = 800.0 / h if h < w else 800.0 / w
        if upscale_factor > 1.0:
            image = cv2.resize(image, None, fx=upscale_factor, fy=upscale_factor,
                               interpolation=cv2.INTER_CUBIC)
    elif h > effective_max_dim or w > effective_max_dim:
        downscale_factor = effective_max_dim / h if h > w else effective_max_dim / w
        if downscale_factor < 1.0:
            image = cv2.resize(image, None, fx=downscale_factor, fy=downscale_factor,
                               interpolation=cv2.INTER_AREA)
            upscale_factor = downscale_factor   # reuse to scale boxes back

    # Run OCR 
    ocr = get_ocr()
    result = ocr.ocr(image, cls=False)
   
    if not result or result[0] is None:
        return [], []

    texts = []
    raw_data = []

    for line in result[0]:
        if len(line) < 2:
            continue

        box = line[0]
        text_info = line[1]

        if upscale_factor != 1.0:
            new_box = []
            if isinstance(box, (list, tuple)):
                for pt in box:
                    try:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            new_box.append([float(pt[0]) / upscale_factor,
                                            float(pt[1]) / upscale_factor])
                        else:
                            new_box.append([0.0, 0.0])
                    except (ValueError, TypeError, IndexError):
                        new_box.append([0.0, 0.0])
                box = new_box
            else:
                box = [[0, 0], [0, 0], [0, 0], [0, 0]]

        if not isinstance(text_info, (list, tuple)) or len(text_info) < 1:
            continue

        text = str(text_info[0])
        if OCR_DEBUG:
            print(f"{text} ---> {box}")
        conf = 0.5
        if len(text_info) > 1:
            try:
                conf = float(text_info[1])
            except (ValueError, TypeError):
                pass

        texts.append(text)
        raw_data.append({"text": text, "box": box, "conf": conf})

    output = (texts, raw_data)

    # Store in cache 
    if img_key is not None:
        if len(_ocr_cache) >= _CACHE_MAX:
            # Evict oldest half when limit reached
            keys = list(_ocr_cache.keys())
            for k in keys[:len(keys) // 2]:
                del _ocr_cache[k]
        _ocr_cache[img_key] = output

    return output