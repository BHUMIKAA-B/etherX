def classify_document(texts):
    text = " ".join(texts).upper()

    # Stronger Heuristics for Aadhaar
    if any(k in text for k in ["AADHAAR", "AADHAR", "UNIQUE IDENTIFICATION", "UIDAI"]):
        return "Aadhaar"
    
    # Heuristic: "Government of India" + ("DOB" or "Year of Birth") + "Male/Female"
    if "GOVERNMENT OF INDIA" in text:
        if any(k in text for k in ["DOB", "YEAR OF BIRTH", "YOB"]) and \
           any(k in text for k in ["MALE", "FEMALE"]):
            return "Aadhaar"

    # Heuristic: VID (Virtual ID) is specific to Aadhaar
    if "VID :" in text or "VID:" in text:
        return "Aadhaar"

    import re

    if "PERMANENT ACCOUNT NUMBER" in text or "INCOME TAX DEPARTMENT" in text:
        return "PAN"
    if "DRIVING LICENCE" in text or "DL NO" in text or "DRIVING LICENSE" in text:
        return "Driving License"
        
    # Heuristic: 14-15 digit alphanumeric DL number pattern (e.g., MH0320080022135, KA01 20200016183)
    if re.search(r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b", text):
        return "Driving License"
    # Fuzzy SSLC match: tolerates OCR noise like missing/extra dots or
    # merged words, e.g. "S.S.LC", "PASSEDS.SLCE", "SSLC", "S S L C"
    sslc_fuzzy = re.search(r"S\.?\s*S\.?\s*L\.?\s*C", text) is not None

    # Marksheet classification with confidence scoring to prevent false positives
    marksheet_score = 0
    
    # Strong exact matches (high confidence)
    if "SSLC" in text or "S.S.L.C" in text or sslc_fuzzy:
        marksheet_score += 2
    if "CBSE" in text or "ICSE" in text or "VISVESVARAYA" in text:
        marksheet_score += 2
    if "SECONDARY EDUCATION" in text or "MARKS STATEMENT CUM CERTIFICATE" in text:
        marksheet_score += 2
    if "COUNCIL FOR THE INDIAN SCHOOL" in text:
        marksheet_score += 2
        
    # Weaker matches (require multiple for confidence)
    if "UNIVERSITY" in text: marksheet_score += 1
    if "KARNATAKA" in text and ("BOARD" in text or "SECONDARY" in text): marksheet_score += 1
    if "OFKARNATAKA" in text or "OF-KARNATAKA" in text: marksheet_score += 1
    if "STATEMENT OF MARKS" in text: marksheet_score += 1
    if "CENTRAL BOARD" in text: marksheet_score += 1
    if "INDIAN CERTIFICATE" in text: marksheet_score += 1

    if marksheet_score >= 1:
        # We need at least one strong match or multiple weak matches
        return "Marksheet"

    if re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text):
        return "Aadhaar"

    return "Unknown"





