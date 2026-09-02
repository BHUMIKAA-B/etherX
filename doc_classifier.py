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

    #if "UNIVERSITY" in text or "MARKS" in text or "RESULT" in text:
    if (
    "UNIVERSITY" in text
    or "VISVESVARAYA" in text
    or "SSLC" in text
    or "S.S.L.C" in text
    or "OFKARNATAKA" in text
    or "OF-KARNATAKA" in text
    or sslc_fuzzy
    or "SECONDARY EDUCATION" in text
   
    or ("KARNATAKA" in text and ("BOARD" in text or "SECONDARY" in text))
    # CBSE (e.g. "CENTRAL BOARD OF SECONDARY EDUCATION", "MARKS STATEMENT CUM CERTIFICATE")
    or "CBSE" in text
    or "CENTRAL BOARD OF SECONDARY EDUCATION" in text
    or "CENTRAL BOARD" in text
    or "MARKS STATEMENT CUM CERTIFICATE" in text
    # ICSE (e.g. "COUNCIL FOR THE INDIAN SCHOOL CERTIFICATE EXAMINATIONS",
    # "INDIAN CERTIFICATE OF SECONDARY EDUCATION", "STATEMENT OF MARKS")
    or "ICSE" in text
    or "INDIAN CERTIFICATE OF SECONDARY EDUCATION" in text
    or "INDIAN CERTIFICATE" in text
    or "COUNCIL FOR THE INDIAN SCHOOL" in text
    or "STATEMENT OF MARKS" in text
     ):
    
        return "Marksheet"

    if re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text):
        return "Aadhaar"

    return "Unknown"





