import os
import io
import re
import uuid
import time
import asyncio
import logging
import difflib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ocr_engine import correct_orientation

from image_preprocessing import load_image
from ocr_engine import run_ocr, is_ocr_ready, get_cache_stats, HIGH_RES_MAX_DIM
from doc_classifier import classify_document

from image_assets import (
    extract_aadhaar_face,
    crop_embedded_aadhaar_card,
    extract_pan_face,
    extract_pan_signature,
    extract_dl_face,
    extract_dl_signature,
    extract_marksheet_face,
)
from field_extractors import aadhaar, pan, dl, marksheet
from schemas import OCRResponse

def _is_new_style_aadhaar(image, texts=None):
    if image is None:
        return False
    h, w = image.shape[:2]
    if h > w * 1.15:
        return True
    if texts:
        joined = " ".join(texts).upper()
        has_to_line = any(t.strip().upper() in ("TO", "TO,") for t in texts)
        has_enrolment = "ENROLMENT" in joined
        if has_to_line and has_enrolment:
            return True
    return False

# Logging 
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ocr_api")

# App 
app = FastAPI(title="Real-Time OCR Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For debugging, allow all while testing connectivity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "90"))   # seconds

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Simple in-process metrics 
_metrics = {
    "requests_total": 0,
    "requests_error": 0,
    "total_latency_ms": 0.0,
}


# Startup warm-up 
async def startup_event():
    """Warm up OCR engine. Called by FastAPI handler and directly by tests."""
    from ocr_engine import get_ocr
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=4))
    await loop.run_in_executor(None, get_ocr)
    logger.info("OCR Engine warmed up and ready.")


@app.on_event("startup")
async def _startup_handler():
    await startup_event()


# Thread-pool helper 
def run_in_thread(func, *args):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args)


# Routes 
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/health")
def health():
    ready = is_ocr_ready()
    cache = get_cache_stats()
    return {
        "status": "ok",
        "ocr_ready": ready,
        "message": "OCR system is ready" if ready else "OCR system is warming up...",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "cache": cache,
    }


@app.get("/metrics")
def metrics():
    total = _metrics["requests_total"]
    avg_ms = (_metrics["total_latency_ms"] / total) if total > 0 else 0.0
    return {
        "requests_total": total,
        "requests_error": _metrics["requests_error"],
        "avg_latency_ms": round(avg_ms, 1),
        "cache": get_cache_stats(),
    }


@app.get("/")
def root():
    return {"message": "OCR API is running"}


def _core_subject_key(s):
    """Longest matching core keyword, so e.g. SCIENCE vs SOCIAL SCIENCE aren't conflated."""
    s_u = s.upper()
    candidates = [c for c in marksheet.CORE_10TH_SUBJECTS if c in s_u]
    return max(candidates, key=len) if candidates else s_u


def _subjects_match(a, b):
    """Same subject across OCR passes, tolerating rewording/misspelling."""
    a, b = a.upper(), b.upper()
    if a == b:
        return True
    key_a, key_b = _core_subject_key(a), _core_subject_key(b)
    if key_a == key_b:
        return True
    if key_a not in marksheet.CORE_10TH_SUBJECTS and key_b not in marksheet.CORE_10TH_SUBJECTS:
        if a in b or b in a:
            return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


def _recover_from(subjects, other_subjects):
   
    recovered = False
    for s in subjects:
        if s.get("marks") is None:
            candidate = next(
                (r.get("marks") for r in other_subjects
                 if r.get("marks") is not None and _subjects_match(s["subject"], r["subject"])),
                None
            )
            if candidate is not None:
                s["marks"] = candidate
                recovered = True

    new_entries = [
        r for r in other_subjects
        if r.get("marks") is not None and r.get("subject")
        and not any(_subjects_match(s["subject"], r["subject"]) for s in subjects)
    ]
    if new_entries:
        recovered = True
        used = set()
        reordered = []
        for r in other_subjects:
            if not r.get("subject"):
                continue
            match_idx = next(
                (i for i, s in enumerate(subjects)
                 if i not in used and _subjects_match(s["subject"], r["subject"])),
                None
            )
            if match_idx is not None:
                reordered.append(subjects[match_idx])
                used.add(match_idx)
            elif r.get("marks") is not None:
                reordered.append(dict(r))
        for i, s in enumerate(subjects):
            if i not in used:
                reordered.append(s)
        subjects[:] = reordered

    return recovered

"""
def _enhance_marks_table_crop(image):   
    h, w = image.shape[:2]
    crop = image[int(0.30 * h):int(0.72 * h), 0:w]
    ch, cw = crop.shape[:2]
    big = cv2.resize(crop, (cw * 3, ch * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

"""
def _enhance_marks_table_crop(image):
    h, w = image.shape[:2]
    crop = image[int(0.30 * h):int(0.72 * h), 0:w]
    ch, cw = crop.shape[:2]
    scale = min(HIGH_RES_MAX_DIM / max(ch, cw), 3.0)
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def _looks_implausible_name(value):
    if not value:
        return True
    cleaned = re.sub(r"[^A-Z]", "", value.upper())
    return len(cleaned) <= 3

def _student_name_duplicates_parent(result):
   
    student = result.get("student_name")
    if not student:
        return False
    student_clean = re.sub(r"[^A-Z]", "", student.upper())
    if not student_clean:
        return False
    for f in ("father_name", "mother_name"):
        other = result.get(f)
        if not other:
            continue
        other_clean = re.sub(r"[^A-Z]", "", other.upper())
        if other_clean and student_clean == other_clean:
            return True
    return False

def _name_looks_merged(value):
    
    if not value:
        return False
    v = value.strip()
    return " " not in v and len(re.sub(r"[^A-Z]", "", v.upper())) >= 8


async def _recover_missing_marksheet_marks(result: dict, image, texts=None, raw_ocr_data=None) -> dict:
    """Redo OCR at higher resolution to fill in missing marks/name fields, only when something is actually missing."""
    if not isinstance(result, dict):
        return result

    subjects = result.get("subjects") or []
    univ_upper = (result.get("university") or "").upper()
    is_vtu = "VISVESVARAYA" in univ_upper or univ_upper == "VTU"
    name_fields = ("student_name",) if is_vtu else ("student_name", "father_name", "mother_name")
    missing_marks = [s for s in subjects if s.get("marks") is None]
    missing_name_fields = [
        f for f in name_fields
        if _looks_implausible_name(result.get(f))
    ]
    
    if "student_name" in name_fields and "student_name" not in missing_name_fields \
            and _student_name_duplicates_parent(result):
        missing_name_fields.append("student_name")
    total_marks = result.get("total_marks")
    known_sum = sum(s["marks"] for s in subjects if s.get("marks") is not None)
    
    total_mismatch = (
        total_marks is not None and subjects
        and not missing_marks
        and known_sum != total_marks
    )
    if not missing_marks and not missing_name_fields and not total_mismatch:
        return result

    texts2 = None
    try:
        retry_start = time.perf_counter()
        texts2, raw2 = await run_in_thread(run_ocr, image, HIGH_RES_MAX_DIM)
        retry_result = await run_in_thread(marksheet.extract, texts2, image, raw2)
        print(f"High-res retry OCR Time: {(time.perf_counter() - retry_start):.2f} sec "
              f"(recovering marks={[s['subject'] for s in missing_marks]}, fields={missing_name_fields})")

        for f in name_fields:
            current = result.get(f)
            retried = retry_result.get(f)
            if not current and retried:
                result[f] = retried
            elif current and retried and current != retried:
                cur_clean = re.sub(r"[^A-Z ]", "", current.upper()).strip()
                ret_clean = re.sub(r"[^A-Z ]", "", retried.upper()).strip()
                if cur_clean != ret_clean and (cur_clean.endswith(ret_clean) or ret_clean.endswith(cur_clean)):
                    if len(ret_clean) > len(cur_clean) and len(ret_clean) >= 3:
                        result[f] = retried
                elif _looks_implausible_name(current) and not _looks_implausible_name(retried):
                    result[f] = retried
                elif f == "student_name":
                    cur_clean_name = re.sub(r"[^A-Z]", "", (current or "").upper())
                    ret_clean_name = re.sub(r"[^A-Z]", "", (retried or "").upper())
                    current_is_parent_dup = any(
                        cur_clean_name and cur_clean_name == re.sub(r"[^A-Z]", "", (src.get(pf) or "").upper())
                        for pf in ("father_name", "mother_name")
                        for src in (result, retry_result)
                    )
                    if current_is_parent_dup:
                        retried_is_parent_dup = any(
                            ret_clean_name and ret_clean_name == re.sub(r"[^A-Z]", "", (src.get(pf) or "").upper())
                            for pf in ("father_name", "mother_name")
                            for src in (result, retry_result)
                        )
                        if not retried_is_parent_dup:
                            result[f] = retried 
                        
          
        
        
        retry_subjects = retry_result.get("subjects") or []
        _recover_from(subjects, retry_subjects)

        still_missing = [s for s in subjects if s.get("marks") is None]
        if still_missing:
            crop_start = time.perf_counter()
            enhanced_crop = _enhance_marks_table_crop(image)
            texts3, raw3 = await run_in_thread(run_ocr, enhanced_crop, HIGH_RES_MAX_DIM)
            crop_result = await run_in_thread(marksheet.extract, texts3, enhanced_crop, raw3)
            crop_subjects = crop_result.get("subjects") or []
            print(f"Enhanced-crop retry OCR Time: {(time.perf_counter() - crop_start):.2f} sec "
                  f"(still missing: {[s['subject'] for s in still_missing]}, crop found: {crop_subjects})")
            _recover_from(subjects, crop_subjects)

        still_missing = [s for s in subjects if s.get("marks") is None]
        if len(still_missing) < len(missing_marks) and result.get("total_marks") is None:
            result["total_marks"] = sum(s["marks"] for s in subjects if s.get("marks") is not None)

        still_missing = [s for s in subjects if s.get("marks") is None]
        total = result.get("total_marks")
        if len(still_missing) == 1 and total is not None:
            known_sum = sum(s["marks"] for s in subjects if s.get("marks") is not None)
            diff = total - known_sum
            if 0 <= diff <= 200:
                still_missing[0]["marks"] = diff

        still_missing_fields = [
            f for f in name_fields
            if _looks_implausible_name(result.get(f))
        ]
        if still_missing_fields and texts and texts2:
            img_h = image.shape[0] if image is not None else 0
            img_w = image.shape[1] if image is not None else 0
            merged_a = marksheet._group_texts_by_y(raw_ocr_data, image_height=img_h, image_width=img_w) if raw_ocr_data else texts
            merged_b = marksheet._group_texts_by_y(raw2, image_height=img_h, image_width=img_w) if raw2 else texts2
            known = [
                result.get(f) for f in name_fields
                if not _looks_implausible_name(result.get(f))
            ]
            consensus = marksheet.recover_names_by_consensus(merged_a, merged_b, still_missing_fields, known)
            #print(f"DEBUG consensus recovery: still_missing_fields={still_missing_fields}, "
                  #f"known={known}, consensus_returned={consensus}")
            for f, v in consensus.items():
                if _looks_implausible_name(result.get(f)):
                    result[f] = v
                    #print(f"DEBUG consensus recovery: applied {f}={v!r}")
                
    except Exception as e:
        logger.warning(f"High-res marksheet retry skipped: {e}")

    return result

    

# ── Validation helpers ──
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/bmp", "image/tiff",
    "image/webp", "image/jpg",
}

def _is_blank_image(image, threshold=12.0):
    """Detect near-uniform (blank) images by checking pixel variance."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Sample center region to avoid border artifacts
        h, w = gray.shape
        crop = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        return float(np.std(crop)) < threshold
    except Exception:
        return False


#  Core processing (inner function, wrapped with timeout) 
async def _process_upload(file: UploadFile) -> dict:
    t_start = time.perf_counter()

    # ── 1. File-level validation ──
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail="Unsupported file type. Please upload a JPG, PNG, BMP, TIFF, or WebP image.",
        )

    #  Read file into memory (no disk I/O) 
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Empty file. Please upload a valid document image.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File too large ({len(contents) // (1024*1024)}MB). Maximum allowed size is 20MB.",
        )

    # ── 2. Image decode ──
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=422,
            detail="Invalid or corrupted file. Please upload a valid image (JPG, PNG, BMP, TIFF, or WebP).",
        )

    # ── 3. Blank image check ──
    if _is_blank_image(image):
        raise HTTPException(
            status_code=422,
            detail="The uploaded image appears to be blank. Please upload a document image with visible content.",
        )

    # OCR (CPU-bound  thread) 
    ocr_start = time.perf_counter()

    texts, raw_ocr_data = await run_in_thread(run_ocr, image)
    print(
    f"OCR Time: {(time.perf_counter() - ocr_start):.2f} sec"
    )

    # ── 4. Minimum text check ──
    if len(texts) < 3:
        raise HTTPException(
            status_code=422,
            detail="Unable to detect sufficient text in the image. Please upload a clearer document image.",
        )

    #  Classify 
    #doc_type = classify_document(texts)
    cls_start = time.perf_counter()

    doc_type = classify_document(texts)

    # Fallback: only pay the rotation-correction cost when classification
    # fails, instead of on every upload.
    if doc_type == "Unknown":
        #rint("Classification failed, retrying with orientation correction")
        image = correct_orientation(image)
        texts, raw_ocr_data = await run_in_thread(run_ocr, image)
        doc_type = classify_document(texts)

    print("DOCUMENT TYPE =", doc_type)
    print(
    f"Classification Time: {(time.perf_counter() - cls_start):.2f} sec"
    )

    KNOWN_DOC_TYPES = {"Aadhaar", "PAN", "Driving License", "Marksheet"}
    if doc_type not in KNOWN_DOC_TYPES:
        raise HTTPException(
            status_code=422,
            detail="Invalid document. Please upload a supported document (Aadhaar, PAN, Driving License, or Marksheet).",
        )

    job_id = uuid.uuid4().hex

    # Pre-compute shared Aadhaar crop (avoids duplicate crop_embedded_aadhaar_card calls)
    _aadhaar_is_new = False
    _aadhaar_crop = None
    if doc_type == "Aadhaar":
        _aadhaar_is_new = _is_new_style_aadhaar(image, texts)
        if _aadhaar_is_new:
            _aadhaar_crop = await run_in_thread(crop_embedded_aadhaar_card, image)

    #  Parallel: field extraction + asset extraction 
    async def extract_fields_task():
        try:
            if doc_type == "Aadhaar":

             field_start = time.perf_counter()

             field_texts = texts
             if _aadhaar_is_new and _aadhaar_crop is not None:
                 crop_texts, _ = await run_in_thread(run_ocr, _aadhaar_crop)
                 if crop_texts:
                     field_texts = crop_texts

             result = await run_in_thread(
              aadhaar.extract_aadhaar_fields,
              field_texts
             )

             print(
             f"Aadhaar Field Extraction Time = {time.perf_counter() - field_start:.2f} sec"
             )

             return result
             
            elif doc_type == "PAN":
                return await run_in_thread(pan.extract, texts, raw_ocr_data)
            elif doc_type == "Driving License":
                return await run_in_thread(dl.extract, texts, image)
            elif doc_type == "Marksheet":
                result = await run_in_thread(marksheet.extract, texts, image, raw_ocr_data)
                return await _recover_missing_marksheet_marks(result, image, texts, raw_ocr_data)
                
        except Exception as e:
            logger.error(f"Field extraction failed for doc_type={doc_type}: {e}", exc_info=True)
        return {}

    async def extract_assets_task():
        async def run_asset(func, *args):
            try:
                return await run_in_thread(func, *args)
            except Exception as e:
                logger.warning(f"Asset extraction skipped ({func.__name__}): {e}")
                return None

        face_out = os.path.join(OUTPUT_DIR, f"face_{job_id}.jpg")
        sig_out  = os.path.join(OUTPUT_DIR, f"signature_{job_id}.jpg")

        if doc_type == "Aadhaar":

         face_start = time.perf_counter()

         face_source_image = _aadhaar_crop if (_aadhaar_is_new and _aadhaar_crop is not None) else image

         results = await asyncio.gather(
         run_asset(extract_aadhaar_face, face_source_image, face_out)
         )

         print(
         f"Aadhaar Face Extraction Time = {time.perf_counter() - face_start:.2f} sec"
         )

         return results[0], None
        elif doc_type == "PAN":
            results = await asyncio.gather(
                run_asset(extract_pan_face, image, face_out),
                run_asset(extract_pan_signature, image, sig_out),
            )
            return results[0], results[1]
        elif doc_type == "Driving License":
            results = await asyncio.gather(
                run_asset(extract_dl_face, image, face_out),
                run_asset(extract_dl_signature, image, sig_out),
            )
            return results[0], results[1]
        elif doc_type == "Marksheet":
            results = await asyncio.gather(run_asset(extract_marksheet_face, image, face_out))
            return results[0], None
        
        return None, None

    extract_start = time.perf_counter()

    data, (face_path, signature_path) = await asyncio.gather(
    extract_fields_task(),
    extract_assets_task(),
    )

    print(
    f"Extraction Time: {(time.perf_counter() - extract_start):.2f} sec"
    )
    #  VTU marksheet: no face 
    if doc_type == "Marksheet" and isinstance(data, dict):
        if data.get("university") == "VISVESVARAYA TECHNOLOGICAL UNIVERSITY":
            if face_path and os.path.exists(face_path):
                try:
                    os.remove(face_path)
                except Exception:
                    pass
                face_path = None
        if "INDIAN CERTIFICATE OF SECONDARY EDUCATION" in (data.get("board") or ""):
            if face_path and os.path.exists(face_path):
                try:
                    os.remove(face_path)
                except Exception:
                    pass
                face_path = None
    print(f"TOTAL TIME = {time.perf_counter() - t_start:.2f} sec")
    return {
        "document_type": doc_type,
        "extracted_fields": data,
        "face_image": face_path,
        "signature_image": signature_path,
        "job_id": job_id,
    }


@app.post("/upload", response_model=OCRResponse, response_model_exclude_none=True)
async def upload_document(file: UploadFile = File(...)):
    _metrics["requests_total"] += 1
    t0 = time.perf_counter()

    try:
        result = await asyncio.wait_for(_process_upload(file), timeout=REQUEST_TIMEOUT)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _metrics["total_latency_ms"] += elapsed_ms
        logger.info(
            f"doc_type={result.get('document_type')} "
            f"job_id={result.get('job_id')} "
            f"latency_ms={elapsed_ms:.0f}"
        )
        return result

    except asyncio.TimeoutError:
        _metrics["requests_error"] += 1
        msg = f"Processing timed out. Please try again with a smaller or clearer image."
        logger.error(f"Request timed out after {REQUEST_TIMEOUT}s")
        return JSONResponse(
            {"detail": msg},
            status_code=408,
        )
    except HTTPException as exc:
        _metrics["requests_error"] += 1
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    except Exception as exc:
        _metrics["requests_error"] += 1
        logger.exception(f"Unhandled error during /upload: {exc}")
        return JSONResponse(
            {"detail": "An unexpected error occurred while processing your document. Please try again."},
            status_code=500,
        )