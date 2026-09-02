import re
import math
import difflib

def _recover_name_spaces(text):
    if not text or " " in text:
        return text
    
    if len(text) > 4 and text[-1] == text[-2]:
        return text[:-1] + " " + text[-1]
    return text

def extract(texts, raw_data=None):
    # Normalize OCR lines
    lines = [t.strip() for t in texts if t.strip()]

    pan_number = None
    dob = None
    dob_index = -1
    dob_box = None

    # Step 1: Find PAN number and DOB
    for i, line in enumerate(lines):
        if not pan_number:
            pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", line)
            if pan_match:
                pan_number = pan_match.group()

        if not dob:
            dob_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", line)
            if dob_match:
                dob = dob_match.group()
                dob_index = i
                if raw_data and i < len(raw_data):
                    dob_box = raw_data[i].get("box")

    # Grouping logic: If raw_data exists, group blocks that are on the same horizontal line
    sorted_blocks = []
    if raw_data:
        # Sort by Y-coordinate of top-left corner
        for d in raw_data or []:
            box = d.get("box")
            if box:
                # Get average Y and min X
                avg_y = sum(p[1] for p in box) / 4
                avg_x = sum(p[0] for p in box) / 4
                min_x = min(p[0] for p in box)
                min_y = min(p[1] for p in box)
                # Rough width/height
                box_w = max(p[0] for p in box) - min(p[0] for p in box)
                box_h = max(p[1] for p in box) - min(p[1] for p in box)
                sorted_blocks.append({"text": d["text"], "y": avg_y, "x": avg_x, "box": box, "w": box_w, "h": box_h})

        # Grouping logic for names: PAN cards can be horizontal or vertical (rotated)
    if sorted_blocks:
        # Detect if text is mostly vertical (width < height)
        vertical_count = sum(1 for b in sorted_blocks if b["h"] > b["w"] * 1.5)
        is_rotated = vertical_count > len(sorted_blocks) * 0.3
        
        if is_rotated:
            
            income_tax_x = None
            for b in sorted_blocks:
                if "INCOME" in b["text"].upper() or "ACCOUNT" in b["text"].upper():
                    income_tax_x = b["x"]
                    break
            
            
            reverse_x = True
            if income_tax_x is not None:
                # Compare to median X of all blocks
                all_x = sorted([b["x"] for b in sorted_blocks])
                median_x = all_x[len(all_x)//2]
                if income_tax_x < median_x:
                    reverse_x = False
            
            sorted_blocks.sort(key=lambda x: x["x"], reverse=reverse_x)
        else:
            sorted_blocks.sort(key=lambda x: x["y"])

    # Collect name candidates and label positions
    name_candidates = []
    labels = {} # type -> index in sorted_blocks
    
    prefixes = [
        r"^NAME[:\s]*", r"^FATHER['S]*\s*NAME[:\s]*", r"^S/O[:\s]*", r"^D/O[:\s]*", r"^W/O[:\s]*"
    ]
    label_patterns = {
        "name": [r"NAME\b", r"T\/NAME\b"],
        "father": [r"FATHER\b", r"FATHERS\s*NAME\b", r"faTa\b"]
    }

    
    FUZZY_LABELS = {"name": "NAME", "father": "FATHER"}
    FUZZY_THRESHOLD = 0.6

    candidate_list = [b["text"] for b in sorted_blocks] if sorted_blocks else lines

    for i, line in enumerate(candidate_list):
        clean_line = line.strip().upper()
        is_pure_label = False

        # Check for labels (exact substring match first)
        for l_type, p_list in label_patterns.items():
            for p in p_list:
                if re.search(p, clean_line, re.IGNORECASE):
                    labels[l_type] = i

        
        if len(clean_line) <= 10 and " " not in clean_line:
            for l_type, ref_word in FUZZY_LABELS.items():
                if l_type in labels:
                    continue  # already found via exact match
                ratio = difflib.SequenceMatcher(None, clean_line, ref_word).ratio()
                if ratio >= FUZZY_THRESHOLD:
                    labels[l_type] = i
                   
                    is_pure_label = True

        if is_pure_label:
            continue

        # Strip prefixes
        for p in prefixes:
            clean_line = re.sub(p, "", clean_line, flags=re.IGNORECASE).strip()
        
        # Candidate filtering
        if re.fullmatch(r"[A-Z0-9 \.]{2,}", clean_line):
            # Exclude headers and noise
            excluded = ["GOVERNMENT", "INCOME", "TAX", "DEPARTMENT", "SIGNATURE", 
                        "PERMANENT", "ACCOUNT", "NUMBER", "CARD", "INDIA", "HRROR", "ERROR", "HRD"]
            
            if any(k in clean_line for k in excluded):
                continue
            
           
            if pan_number and clean_line.replace(" ", "") == pan_number:
                continue
            if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", clean_line.replace(" ", "")):
                continue

            # Additional check for names: must contain at least one vowel (mostly)
            # or be a recognized initial pattern like "A K"
            if not re.search(r"[AEIOUY]", clean_line) and len(clean_line) > 3:
                continue

            
            if " " not in clean_line and len(clean_line) < 3:
                continue
            if len(clean_line) < 2:
                continue            
            recovered = _recover_name_spaces(clean_line).upper()
            name_candidates.append({"text": recovered, "index": i})
  
    name_candidates.sort(key=lambda c: 0 if " " in c["text"] else 1)

    # Step 3: Assign names
    name = None
    father_name = None   
    header_idx = None
    for i, line in enumerate(candidate_list):
        if re.search(r"INCOME\s*TAX|GOVT\.?\s*OF\s*INDIA|GOVERNMENT\s*OF\s*INDIA", line.upper()):
            header_idx = i  # keep the LAST header line found

    bound_end = dob_index if dob_index != -1 else len(candidate_list)

    if header_idx is not None and bound_end > header_idx:
        in_bounds = [c for c in name_candidates if header_idx < c["index"] < bound_end]
        in_bounds.sort(key=lambda c: c["index"])
        if len(in_bounds) >= 2:
            name = in_bounds[0]["text"].title()
            father_name = in_bounds[1]["text"].title()
        elif len(in_bounds) == 1:
            name = in_bounds[0]["text"].title()

    
    if not name or not father_name:
        name_val = None
        father_val = None

        if "name" in labels:
            idx = labels["name"]
            nearest = sorted(name_candidates, key=lambda c: abs(c["index"] - idx))
            if nearest:
                name_val = nearest[0]["text"]
                name_candidates = [c for c in name_candidates if c["text"] != name_val]

        if "father" in labels:
            idx = labels["father"]
            nearest = sorted(name_candidates, key=lambda c: abs(c["index"] - idx))
            if nearest:
                father_val = nearest[0]["text"]
                name_candidates = [c for c in name_candidates if c["text"] != father_val]

        if name_val and father_val:
            name = name or name_val.title()
            father_name = father_name or father_val.title()
        elif len(name_candidates) >= 2 or (name_val or father_val):
            if not name_val:
                name_val = name_candidates[0]["text"] if name_candidates else None
                name_candidates = name_candidates[1:]
            if not father_val:
                father_val = name_candidates[0]["text"] if name_candidates else None

            name = name or (name_val.title() if name_val else None)
            father_name = father_name or (father_val.title() if father_val else None)
    
    # Final cleanup: recover spaces 
    if name: name = _recover_name_spaces(name.upper()).title()
    if father_name: father_name = _recover_name_spaces(father_name.upper()).title()

    return {
        "name": name,
        "father_name": father_name,
        "pan_number": pan_number,
        "date_of_birth": dob
    }

