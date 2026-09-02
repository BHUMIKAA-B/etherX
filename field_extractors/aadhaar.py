import re

def clean_name(text):
    if not text:
        return text
    # Split merged initial at the end: "NameM" -> "Name M"
    # Look for Lowercase followed by Uppercase at the VERY END.
    text = re.sub(r"([a-z])([A-Z])$", r"\1 \2", text)
    return text.title()

def extract_name(text_lines):
    # Common non-name words to filter out (Lower case)
    # Added explicit UI terms to prevent screenshot noise
    invalid_keywords = {
        "government", "india", "unique", "identification", "authority",
        "aadhaar", "dob", "year", "yob", "male", "female", "father", 
        "mother", "husband", "wife", "address", "s/o", "d/o", "w/o",
        "vill", "dist", "state", "pin", "help", "download", "generated", "issue",
        "rural", "urban", "road", "cross", "main", "street", "lane", "layout",
        "nagar", "colony", "block", "sector", "phase", "near", "opp", "behind",
        "district", "disict", "bengaluru", "bangalore", "vid",
        # UI/App Text to ignore
        "crop", "adjust", "edit", "share", "send", "delete", "lens", "more", "favourite",
        "file", "scan", "page", "search", "menu", "home", "back"
    }

    lines = [line.strip() for line in text_lines]
    
    header_index = -1
    footer_index = -1
    
    # Scan for Header
    for i, line in enumerate(lines):
        l_lower = line.lower()
        if "government" in l_lower or "india" in l_lower or "aadhaar" in l_lower:
            # Update header index, but keep pushing it down if we see multiple headers
            # (e.g. Gov of India top line)
            header_index = i
        # Stop looking for header if we hit data-like fields
        if any(x in l_lower for x in ["dob", "year", "male", "female", "s/o", "w/o"]):
            break
            
    # Scan for Footer (Anchor) - First strong data field
    for i, line in enumerate(lines):
        l_lower = line.lower()
        # DOB/Gender are strongest anchors for Name (Name is usually just above)
        if any(x in l_lower for x in ["dob", "date of birth", "year of birth", "yob", "male", "female"]):
            footer_index = i
            break
            
    # STRATEGY 1: "To" Pattern (Address Side)
    # If we see "To", the name is almost certainly the next line.
    for i, line in enumerate(lines):
        if line.lower() in ["to", "to,"]:
            # Check next 3 lines
            for offset in range(1, 4):
                if i + offset >= len(lines): break
                cand = lines[i + offset]
                if _is_valid_name(cand, invalid_keywords):
                    return clean_name(cand)

   
    if header_index != -1 and footer_index != -1:
        # We have both bounds. Scan between them.
        if header_index < footer_index:
            # Standard Top-Down: Header (0) ... Name ... Footer (10)
            # Scan Upwards from Footer to Header
            scan_range = range(footer_index - 1, header_index, -1)
        else:
            # Reversed/Bottom-Up: Footer (2) ... Name ... Header (10)
            # Scan "Downwards" (in list) from Footer to Header
            scan_range = range(footer_index + 1, header_index, 1)
            
        for i in scan_range:
            line = lines[i]
            if _is_valid_name_candidate(line, invalid_keywords):
                return clean_name(line)
                
    elif footer_index != -1:
        # Only Footer found (Header missing/unrecognized).
        # Assume Standard Order: Name is ABOVE Footer.
        # Scan 5 lines above Footer.
        start = max(0, footer_index - 6)
        for i in range(footer_index - 1, start - 1, -1):
            line = lines[i]
            if _is_valid_name_candidate(line, invalid_keywords):
                 return clean_name(line)
                 
        # Fallback: Sometimes Name is BELOW Footer if completely inverted?
        # Check 2 lines below just in case.
        end = min(len(lines), footer_index + 3)
        for i in range(footer_index + 1, end):
            line = lines[i]
            if _is_valid_name_candidate(line, invalid_keywords):
                 return clean_name(line)

    return None

def _is_valid_name_candidate(line, invalid_keywords):
    # Immediate skip checks
    if not line: return False
    if len(line) < 3: return False
    if any(char.isdigit() for char in line): return False 
    
    # Keyword check
    lower = line.lower()
    if any(k in lower for k in invalid_keywords): return False
    
    # Explicit Label Handling "Name: X"
    if lower.startswith("name"):
        return False # The label line itself is not a name. 
        
    return _is_valid_name(line, invalid_keywords)

def _is_valid_name(text, invalid_keywords):
    # Helper to validate a name candidate
    # 1. Length check
    if len(text) < 3: return False
    
    # 2. Keyword check (Redundant but safe)
    text_lower = text.lower()
    if any(k in text_lower for k in invalid_keywords): return False
    
    # 3. Content check
    # Must contain mostly letters. No numbers.
    # Allow . and spaces
    if re.search(r"[0-9]", text): return False
    
    # 4. Format check
    # Names usually start with Uppercase.
    # Reject "ecoe" (OCR noise)
    if not text[0].isupper(): return False
    
    # 5. Char Set Check (Allow A-Z, space, dot, hyphen, apostrophe)
    # Regex: Must start with a letter. Can contain letters, spaces, dots, hyphens, apostrophes.
    if not re.match(r"^[A-Za-z][A-Za-z\s\.\-\']+$", text):
        return False
        
    # 6. Minimum Letter Count
    # Reject "A. ." or similar noise
    letter_count = sum(c.isalpha() for c in text)
    if letter_count < 3: return False
        
    return True

def extract_dob(text_lines):
    # Dates on these lines are card-metadata, not the holder's DOB
    SKIP_KEYWORDS = ["issue", "generated", "download", "gov", "india", "print", "update"]
    DATE_SKIP_KEYWORDS = ["issue", "generated", "download", "print", "update"]

    # 1️⃣ Highest priority: DOB label line (same line OR next line if OCR splits)
    # Handles: "DOB: 01/01/1990", "Date of Birth: ...", Hindi "जन्म तिथि/DOB:"
    for i, line in enumerate(text_lines):
        l = line.lower()
        if any(x in l for x in ["dob", "d0b", "date of birth", "\u091c\u0928\u094d\u092e", "year of birth"]):
            # Same line
            match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", line)
            if match:
                return match.group()
            # Next line (OCR sometimes splits label and date)
            if i + 1 < len(text_lines):
                match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", text_lines[i + 1])
                if match:
                    return match.group()
            # Year-only DOB: "DOB: 1995"
            match = re.search(r"\b(19|20)\d{2}\b", line)
            if match:
                return match.group()

    # Helper to check if a date candidate should be skipped
    def should_skip_date(idx, candidate_line):
        if any(k in candidate_line.lower() for k in SKIP_KEYWORDS):
            return True
        if idx > 0:
            prev_l = text_lines[idx-1].lower()
            if any(k in prev_l for k in DATE_SKIP_KEYWORDS):
                return True
        return False

    # 2️⃣ Date near Gender line — skip any metadata date lines
    for i, line in enumerate(text_lines):
        l = line.lower()
        if "female" in l or "male" in l:
            for j in range(max(0, i - 3), min(len(text_lines), i + 4)):
                candidate = text_lines[j]
                if should_skip_date(j, candidate):
                    continue
                match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", candidate)
                if match:
                    return match.group()

    # 3️⃣ Generic scan — skip metadata date lines, checking both current and previous lines for specific context
    for i, line in enumerate(text_lines):
        if should_skip_date(i, line):
            continue

        match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", line)
        if match:
            return match.group()

    # 4️⃣ Fallback: "Year of Birth" / "YOB" label
    for line in text_lines:
        l = line.lower()
        if "year" in l or "yob" in l or "birth" in l:
            match = re.search(r"(19|20)\d{2}", line)
            if match:
                return match.group()

    # 5️⃣ Last resort: standalone year near Gender line
    for i, line in enumerate(text_lines):
        l = line.lower()
        if "female" in l or "male" in l:
            for j in range(max(0, i - 1), min(len(text_lines), i + 2)):
                candidate = text_lines[j]
                if should_skip_date(j, candidate):
                    continue
                y_match = re.search(r"\b(19|20)\d{2}\b", candidate)
                if y_match:
                    if len(candidate) < 20:
                        return y_match.group()

    return None

def extract_gender(text_lines):
    for line in text_lines:
        l = line.lower()
        if "female" in l:
            return "Female"
        if "male" in l:
            return "Male"
    return None

def extract_aadhaar_number(text_lines):
    for line in text_lines:
        # Aadhaar: 12 digits (with OR without spaces)
        match = re.search(r"\b\d{12}\b", line.replace(" ", ""))
        if match:
            return match.group()
    return None

def extract_aadhaar_fields(text_lines):
    dob_value = extract_dob(text_lines)
    
    res = {
        "name": extract_name(text_lines),
        "date_of_birth": None,
        "year_of_birth": None,
        "gender": extract_gender(text_lines),
        "aadhaar_number": extract_aadhaar_number(text_lines),
    }

    if dob_value:
        if len(dob_value) == 4 and dob_value.isdigit():
            res["year_of_birth"] = dob_value
        else:
            res["date_of_birth"] = dob_value

    return res