import cv2
import re
import difflib

#print("[marksheet.py] PUC name/father/mother/register fix loaded (v2)")

try:
    from ocr_engine import get_ocr
except ImportError:
    def get_ocr():
        return None

def detect_rows(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > image.shape[1] * 0.5:
            rows.append((y, y + h))

    rows = sorted(rows, key=lambda x: x[0])
    cropped = []

    for i in range(len(rows) - 1):
        y1 = rows[i][1]
        y2 = rows[i + 1][0]

        if y2 - y1 > 25:
            cropped.append(image[y1:y2, :])
    return cropped

def read_row(row_img):
    ocr = get_ocr()
    if ocr is None: return ""    
    result = ocr.ocr(row_img)

    if not result or result[0] is None:
        return ""
    return " ".join([line[1][0] for line in result[0]]).upper()
    return {"subject": subject, "marks": marks}

def _repair_spacing(text):
    if not text: return text
    joins = {
        "CAMBRIDGEINSTITUTE": "CAMBRIDGE INSTITUTE",
        "OFTECHNOLOGY": "OF TECHNOLOGY",
        "COMPUTERORGANIZATION": "COMPUTER ORGANIZATION",
        "PROFESSIONALCOMMUNICATON": "PROFESSIONAL COMMUNICATION",
        "ELEDRO": "ELECTRO",
        "LABORTORY": "LABORATORY",
        "SOMNG": "SOLVING",
        "INTODION": "INTRODUCTION",
        "COMMUNICATON": "COMMUNICATION",
        "ETIC": "ETHIC",
        "DISARETE": "DISCRETE",
        "MATHIEMATICS": "MATHEMATICS",
        "FUNDARNENTALS": "FUNDAMENTALS",
        "PROGRANMING": "PROGRAMMING",
        "DIGTA": "DIGITAL",
        "ELEDRONICS": "ELECTRONICS",
        "UNX": "UNIX",
        "DISCRETEMATHEMATICS": "DISCRETE MATHEMATICS",
        "CPROGRAMMING": "C PROGRAMMING",
        "HNDI": "HINDI",
        "HIND1": "HINDI",
        "HINDI3": "HINDI",
        "INNIH": "HINDI",
        "IHNNI": "HINDI",
    }
    for join, split in joins.items():
        text = text.replace(join, split)
    
    text = re.sub(r"([A-Z])(OF|AND|FOR|THE)\b", r"\1 \2", text)
    text = re.sub(r"([A-Z])(INSTITUTE|COLLEGE|TECHNOLOGY|UNIVERSITY)\b", r"\1 \2", text)
    text = text.replace("DISCRETEMATHEMATICS", "DISCRETE MATHEMATICS")    
    return _clean_val(text)

def parse_subject(text, **kwargs):   
    GENERIC_IGNORE = ["BOARD", "UNIVERSITY", "RESULT", "DATE", "CODE", "ROLL", "REGISTER", "GRANT", "MARK", "MARKS", "TOTAL", "TOTALMARKS", "GRANDTOTAL", "PAGE", "YEAR", "MONTH", "STATEMENT", "MEDIUM", "INSTRUCTION", "CLASS", "DECLARED", "DISTINCTION", "PERCENT", "PERCENTAGE", "CERTIFIRATE"]
    SPECIFIC_IGNORE = ["NAME", "BIRTH", "GENDER", "SEX", "DOB", "PLACE", "SECRETARY", "BENGALURU", "RURAL", "DIRECTOR", "OFFICE", "EXAMINATION", "SECONDARY", "KARNATAKA", "DEPARTMENT", "K.S.E.E.B", "KSEEB", "K.P.B", "KPU", "GOVERNMENT", "EDUCATION", "STUDENT", "RETAKE", "DONE", "STATEMENT", "BELOW", "ABOVE", "FIRSTCLASS", "SECONDCLASS", "PASSIN", "ATTEMPT", "SOUTH", "NORTH", "EAST", "WEST", "REG NO", "USN", "SERIAL", "REGISTER NO", "GEM"]
    
    VTU_CODE_REGEX = r"\b\d{1,2}[A-Z]{2,5}\d{1,3}[A-Z]?\b"
    up_text = text.upper()
    has_vtu_code = re.search(VTU_CODE_REGEX, up_text)
    CORE_WHITELIST = ["KANNADA", "SANSKRIT", "HINDI", "ENGLISH", "MATHEMATICS", "SCIENCE", "SOCIAL", "SOCIAL SCIENCE", "SOCAL", "KANN"]

    is_core = any(cw in up_text for cw in CORE_WHITELIST)
    if is_core:

        if any(h in up_text for h in ["MEDIUM", "INSTRUCTION", "SCHOOL", "COLLEGE", "ACADEMY", "INSTITUTE"]):

            if not any(lp in up_text for lp in ["FIRST LANGUAGE", "SECOND LANGUAGE", "THIRD LANGUAGE", "THRDLANGUAGE"]):
                 return None
    
    if is_core:
         pass
    elif any(re.search(rf"\b{w}\b", up_text) for w in GENERIC_IGNORE) or any(w in up_text for w in SPECIFIC_IGNORE) or "GOVERNMENT" in up_text:

        if not has_vtu_code:
            return None

    parts = re.split(r"(\b\d{2,3}\b)", text, maxsplit=1)
    subject_part = parts[0]
    remaining_part = "".join(parts[1:])
    
    all_nums = [int(m) for m in re.findall(r"\b\d+\b", text)]

    if not all_nums and not any(cw in up_text for cw in CORE_WHITELIST): 
        return None
    res_meta = {}
    marks = all_nums[-1] if all_nums else None
    if len(all_nums) >= 9:
        nums = all_nums[-9:]
        obt_ext = nums[2]
        obt_int = nums[5]
        obt_tot = nums[8]
        
        expected_tot = obt_ext + obt_int
        
        if abs(obt_tot - expected_tot) <= 5:
            marks = obt_tot
        else:

            if expected_tot > 30 and expected_tot < 151:
                marks = expected_tot

                res_meta = {"vtu_mismatch": True}
            else:
                marks = obt_tot
    elif len(all_nums) >= 3:

        marks_nums = [int(m) for m in re.findall(r"\b\d+\b", remaining_part)]
        if marks_nums:
            if len(marks_nums) >= 3:

                if 0 <= marks_nums[2] <= 125:
                   marks = marks_nums[2]
            elif len(marks_nums) == 2:

                marks = marks_nums[1]
            else:
                marks = marks_nums[0]
            
        for i in range(len(all_nums)-1):
            for j in range(i+1, len(all_nums)-1):
                if all_nums[i] + all_nums[j] == all_nums[-1]:
                    marks = all_nums[-1]
                    break
    text = re.sub(r"\b[0-9]{0,2}[A-Z]{2,5}[0-9]{1,3}[A-Z]{0,1}\b", "", subject_part, count=1).strip()
   
    subject = re.sub(r"\d+", "", text)
    subject = re.sub(r"[^A-Z ]", "", subject)
    subject = _repair_spacing(subject)

    words = [w for w in subject.split() if w.strip()]
    PREFIXES = ["LANGUAGE", "FIRST", "SECOND", "THIRD", "PAPER", "PART", "CCE", "REGULAR", "FRESH", "PRIVATE", "NSR", "NSPR"]
    while words:
        w0 = words[0].upper()
        if w0 in PREFIXES:
            words = words[1:]
            continue
        if len(words) > 1 and len(w0) <= 4 and w0.isalpha() and w0 not in ["ARTS", "URDU", "UNIX", "YOGA", "MATH", "ART"]:
             if w0 in ["MCA", "BE", "CS", "IS", "EE", "EC", "ME"]:
                words = words[1:]
                continue
        break
    
    subject = re.sub(r"\s+", " ", subject).strip()    
    subject = re.sub(r"\s(?:[ABSFIP][\+\#]?|PASS|FAIL)$", "", subject)    
    subject_upper = subject.upper()
    if "KANNADA" in subject_upper: subject = "KANNADA"
    elif "SANSKRIT" in subject_upper: subject = "SANSKRIT"
    elif "ENGLISH" in subject_upper: subject = "ENGLISH"
    elif "MATHEMATICS" in subject_upper: subject = "MATHEMATICS"
    elif "SOCIAL" in subject_upper: subject = "SOCIAL SCIENCE"
    elif "SCIENCE" in subject_upper: subject = "SCIENCE"
    elif "HINDI" in subject_upper: subject = "HINDI"
    

    if any(p in subject_part.upper() for p in ["FIRST", "SECOND", "THIRD"]):

        for p in ["FIRST", "SECOND", "THIRD"]:
            if p in subject_part.upper() and p not in subject.upper():
                subject = f"{p} LANGUAGE {subject}"
                break

    subject = re.sub(r"\s+", " ", subject).strip()
    return {"subject": subject, "marks": marks, "meta": res_meta}

def _reconcile_marks(subjects, grand_total):
    if not subjects or not grand_total or not isinstance(grand_total, (int, float)):
        return subjects
        
    try:
        current_sum = sum(s.get("marks", 0) or 0 for s in subjects)
        diff = current_sum - grand_total        
        if diff == 0:
            return subjects            

        if abs(diff) in [12, 11, 10, 9, 8, 2, 1]:

            certainty_map = {"Low": 0, "Medium": 1, "High": 2, None: 0}
            sorted_subjects = sorted(subjects, key=lambda x: certainty_map.get(x.get("certainty") or x.get("meta", {}).get("certainty")))            
            theory_subs = [s for s in sorted_subjects if (s.get("marks", 0) or 0) > 60]
            lab_subs = [s for s in sorted_subjects if (s.get("marks", 0) or 0) <= 60]
        
        if abs(diff) == 12:
            if theory_subs and lab_subs:
                theory_subs[0]["marks"] -= (10 if diff > 0 else -10)
                lab_subs[0]["marks"] -= (2 if diff > 0 else -2)
                return subjects
        
        if abs(diff) == 10 and theory_subs:
            theory_subs[0]["marks"] -= diff
            return subjects
            
        if abs(diff) == 2 and lab_subs:
            lab_subs[0]["marks"] -= diff
            return subjects
            

        if abs(diff) <= 3:
            subjects[0]["marks"] -= diff
            return subjects
            
        return subjects
    except:
        pass
    return subjects

def _extract_vtu_blocks(lines):    
    VTU_CODE_REGEX = r"\b(\d{1,2}[A-Z]{2,5}\d{1,3}[A-Z]?|[A-Z]{2,6}[0-9OIL]{2,3}[A-Z]?)\b"

    def _vtu_code_match(text):
        m = re.search(VTU_CODE_REGEX, text)
        if m and any(c.isdigit() for c in m.group(1)):
            return m
        return None

    FOOTER_KEYWORDS = ["GRAND TOTAL", "GRANDTOTAL", "RESULT OF THE SEMESTER", "RESUT", "DATE!", "REGISTRAR", "NOTE", "ABSENT", "N-NOT EIGE", "SGPA", "CGPA", "CUMULATIVE", "REPEATED EXAM"]
    UNIV_NOISE = ["TITLE", "MAX", "MIN", "OBTAINED", "RESULT", "CREDITS", "GRADE", "PASS", "FAIL", "TOTAL", "EXTERNAL", "INTERNAL", "ASSESSMENT", "EXAMINATION", "MARKS", "SUBJECT", "OBTAL", "OBTALNEC", "MAZ", "RESUT", "RESUT OT THE", "LETTER", "POINT", "EARNED", "ASSIGNED", "COURSE", "CXG", "SGPA", "CGPA", "REGISTERED", "CUMULATIVE", "REPEATED", "EXAM", "MEDIUM", "INSTRUCTIONENGLISH", "CODE", "NAME", "ANNOUNCED", "UPDATED"]

    def _clean_subject_title(block_text_upper, code):
        title = block_text_upper.replace(code, "")
        title = re.sub(r"\d+", "", title)
        title = title.replace("-", " ")
        title = re.sub(r"[^A-Z ]", "", title)
        title = _repair_spacing(title)
        words = [w for w in title.split() if w.strip()]
        words = [w for w in words if w not in UNIV_NOISE and (len(w) > 1 or w in "CIVX") and not any(f in w for f in FOOTER_KEYWORDS)]
        if words and words[0].isdigit():
            words = words[1:]
        return " ".join(words).strip()

    subjects_a = []
    lines_list = list(lines)
    i_a = 0
    while i_a < len(lines_list):
        line = lines_list[i_a]
        line_up = line.upper()
        if any(f in line_up for f in FOOTER_KEYWORDS):
            break

        code_match = _vtu_code_match(line_up)
        if not code_match:
            i_a += 1
            continue
        code = code_match.group(1)
        after_code = line_up[code_match.end():]
        after_code_no_date = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", after_code)
        nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", after_code_no_date)]
        result_match = re.search(r"\b([PFAW])\b", after_code_no_date)
        total = None
        certainty = "High"
        if len(nums) == 3:
            internal, external, total = nums
            if abs((internal + external) - total) > 2:
                total = None
            elif not result_match:
                certainty = "Medium"
        elif len(nums) == 2 and result_match:
            total = nums[-1]
            certainty = "Medium"

        if total is not None:
            title = _clean_subject_title(line_up, code)

            j = i_a + 1
            while j < len(lines_list):
                nxt = lines_list[j].strip().upper()
                if not nxt:
                    j += 1
                    continue

                if _vtu_code_match(nxt):
                    break

                if re.search(r"\b\d{2,3}\b", nxt):
                    break

                if any(f in nxt for f in FOOTER_KEYWORDS):
                    break
                if any(k in nxt for k in ["PASS", "FAIL", "ABSENT", "WITHHELD", "NOMENCLATURE", "ABBREVIATION"]):
                    break

                extra = re.sub(r"[^A-Z\s]", " ", nxt).strip()
                extra_words = [w for w in extra.split() if w not in UNIV_NOISE and len(w) > 1]
                if extra_words:
                    title = (title + " " + " ".join(extra_words)).strip()
                j += 1

            if title:
                subjects_a.append({
                    "subject": title,
                    "marks": total,
                    "certainty": certainty,
                })
        i_a += 1

    if subjects_a:
        return subjects_a
    subjects = []
    blocks = []
    current_block = []
    for line in lines:
        line_up = line.upper()
        if any(f in line_up for f in FOOTER_KEYWORDS):
            break

        if _vtu_code_match(line_up):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        elif current_block:
            current_block.append(line)
    if current_block:
        blocks.append(current_block)

    full_text_up = " ".join(lines).upper()
    is_grade_card = any(k in full_text_up for k in ["GRADE CARD", "SGPA", "CGPA", "CREDIT"])

    for block in blocks:
        block_text = " ".join(block).upper()
        match = _vtu_code_match(block_text)
        code = match.group(1) if match else ""

        final_title = _clean_subject_title(block_text, code)
        if not final_title:
            continue

        code_pos = block_text.find(code)
        post_code_text = block_text[max(0, code_pos + len(code)):]
        all_nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", post_code_text)]

        marks = None
        certainty = "Low"

        candidates = []
        if not is_grade_card and len(all_nums) >= 8:
            for j in range(len(all_nums) - 8):
                subset = all_nums[j:j+9]
                sum_val = subset[2] + subset[5]
                total_val = subset[8]
                total_max = subset[6]
                max_sum = subset[0] + subset[3]
                if total_max in [100, 125, 150, 175, 200, 80, 50, 40, 30]:
                    if sum_val == total_val and total_val > 0:
                        score = 100
                        if total_val < total_max:
                            score += 80
                        elif total_val == total_max and total_max >= 100:
                            score -= 90
                        if max_sum == total_max:
                            score += 100
                        candidates.append({"marks": total_val, "score": score, "certainty": "High"})
                    elif max_sum == total_max and sum_val > 0 and sum_val <= total_max:
                        score = 150
                        if sum_val < total_max:
                             score += 50
                        candidates.append({"marks": sum_val, "score": score, "certainty": "Medium"})
                    elif abs(sum_val - total_val) <= 12 and total_val > 5 and total_val <= total_max:
                        score = 50
                        if total_val < total_max:
                            score += 20
                        if max_sum == total_max:
                            score += 30
                        candidates.append({"marks": total_val, "score": score, "certainty": "Medium"})

        if candidates:
            best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]
            marks = best["marks"]
            certainty = best["certainty"]

        if is_grade_card and len(all_nums) >= 2:
            for j in range(len(all_nums) - 2):
                c, ce, val3 = all_nums[j], all_nums[j+1], all_nums[j+2]
                if 1 <= c <= 5 and 1 <= ce <= 5:
                     marks = val3
                     certainty = "High"
                     if j + 3 < len(all_nums):
                          val4 = all_nums[j+3]
                          if 0 <= val4 <= 10:
                               marks = val4
                     break
            if marks is None:
                valid_pts = [n for n in all_nums if 0 <= n <= 10]
                if valid_pts:
                    marks = valid_pts[-1]
                    certainty = "Low"

        if marks is None and not is_grade_card and len(all_nums) >= 6:
            for k in range(len(all_nums) - 5):
                 e_mx, e_obt, i_mx, i_obt = all_nums[k+1], all_nums[k+2], all_nums[k+4], all_nums[k+5]
                 if e_mx in [100, 125, 75, 50, 40] and i_mx in [50, 25, 20]:
                      marks = e_obt + i_obt
                      certainty = "Low"
                      break

        if marks is None and len(all_nums) >= 3:
            best_mx = -1
            for k in range(len(all_nums) - 2):
                mx, mn, obt = all_nums[k], all_nums[k+1], all_nums[k+2]
                if mx in [150, 125, 100, 80, 50, 40] and mn in [75, 50, 44, 40, 35, 25, 20]:
                    if obt < mx:
                        if mx >= best_mx:
                            marks = obt
                            best_mx = mx
                            certainty = "Low"

        if marks is not None:
             subjects.append({"subject": final_title, "marks": marks, "certainty": certainty})

    return subjects

def _extract_puc_subjects(lines):   
    subjects = []
    seen_subjects = set()
    HEADER_NOISE = {
        "MAX", "MIN", "MARKS", "OBTAINED", "THEORY", "EXAM", "INTERNAL",
        "TOTAL", "GRAND", "PART", "LANGUAGE", "OPTIONALS", "SCHOLASTIC",
        "GOVERNMENT", "KARNATAKA", "EXAMINATION", "BOARD", "CERTIFICATE",
        "CERTIFY", "REGISTER", "YEAR", "MEDIUM", "CANDIDATE", "DISTINCTION",
        "WORDS", "CLASS", "CHAIRPERSON", "COLLEGE", "DETAILS", "RESULT",
        "MORYUN", "YEAROF", "TOSAL", "ROGEODRIDO", "ROUDEOND", "ROGOOR",
        "ROND", "ROOD", "AIOR", "OXSZOR", "RON", "OTR", "EOTRD",
        "MAXMARKS", "PARTLL", "PARTHL", "PARTH", "DECLARED", "REGISTERNO",
    }

    PUC_SUBJECTS_WHITELIST = {
        "ENGLISH", "HINDI", "KANNADA", "SANSKRIT", "URDU", "TAMIL", "TELUGU",
        "MATHEMATICS", "MATHEMATIC", "MATHS", "PHYSICS", "CHEMISTRY", "BIOLOGY", "COMPUTER",
        "HISTORY", "GEOGRAPHY", "ECONOMICS", "POLITICAL", "SOCIOLOGY",
        "ACCOUNTANCY", "BUSINESS", "BUSI", "STATISTICS", "PSYCHOLOGY", "HOME",
        "LOGIC", "EDUCATION", "MUSIC", "ARTS", "COMMERCE", "ELECTRONICS",
    }

    WORD_DIGITS = {
        "ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
        "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9",

        "ZCRO": "0", "ZQRO": "0", "ZER0": "0",
        "EICHT": "8", "EIGHТ": "8", "ELGHT": "8", "E1GHT": "8",
        "FOIR": "4", "FEUR": "4",
        "FIVF": "5", "F1VE": "5",
        "S1X": "6", "SLX": "6",
        "THRCE": "3", "THREF": "3",
        "NIME": "9", "NlNE": "9",
    }
    valid_th_maxes  = {80, 100, 150, 90, 70, 60, 50, 40, 30}
    valid_int_maxes = {20, 25, 50, 30, 10, 15}

    def _try_puc_pattern(nums):
        for k in range(len(nums) - 5):
            th_max, th_obt   = nums[k],   nums[k+1]
            int_max, int_obt = nums[k+2], nums[k+3]
            tot_max, tot_obt = nums[k+4], nums[k+5]
            if (th_max in valid_th_maxes and int_max in valid_int_maxes
                    and abs((th_max + int_max) - tot_max) <= 2
                    and 0 <= th_obt  <= th_max
                    and 0 <= int_obt <= int_max
                    and abs((th_obt + int_obt) - tot_obt) <= 2):
                return tot_obt
        return None

    def _try_puc_pattern_partial(nums):
        if len(nums) != 5:
            return None

        best = None
        for missing_slot in range(6):
            slots = [None] * 6
            pos = 0
            for slot in range(6):
                if slot == missing_slot:
                    continue
                slots[slot] = nums[pos]
                pos += 1
            th_max, th_obt, int_max, int_obt, tot_max, tot_obt = slots

            if th_max is not None and th_max not in valid_th_maxes:
                continue
            if int_max is not None and int_max not in valid_int_maxes:
                continue
            if th_max is not None and int_max is not None and tot_max is not None:
                if abs((th_max + int_max) - tot_max) > 2:
                    continue
            if th_obt is not None and th_max is not None and not (0 <= th_obt <= th_max):
                continue
            if int_obt is not None and int_max is not None and not (0 <= int_obt <= int_max):
                continue
            if tot_obt is not None and tot_max is not None and not (0 <= tot_obt <= tot_max):
                continue

            if tot_obt is not None:
                candidate = tot_obt
            elif th_obt is not None and int_obt is not None:
                candidate = th_obt + int_obt
            else:
                continue

            score = sum(x is not None for x in (th_max, int_max, tot_max))
            if tot_obt is not None:
                score += 1
            if best is None or score > best[1]:
                best = (candidate, score)

        return best[0] if best else None

    def _fuzzy_subject_match(word):
        if len(word) < 5:
            return None
        matches = difflib.get_close_matches(word, PUC_SUBJECTS_WHITELIST, n=1, cutoff=0.8)
        return matches[0] if matches else None

    def _has_subject_kw(text):
        up = text.upper()
        if any(kw in up for kw in PUC_SUBJECTS_WHITELIST):
            return True

        return any(_fuzzy_subject_match(w) for w in re.findall(r"[A-Z]+", up))

    MONTH_TOKENS = {
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    }

    _WORD_DIGIT_RUN_RE = re.compile(
        r"^(?:" + "|".join(sorted(WORD_DIGITS, key=len, reverse=True)) + r"){2,}$"
    )

    def _subject_label_prefix(text):
        m = re.match(r"^[^\d]*", text)
        return m.group() if m else text

    def _clean_subj(raw_text):
        s = re.sub(r"\d+", "", raw_text.upper())
        s = re.sub(r"[^A-Z\s]", " ", s)
        words = [
            w for w in s.split()
            if w
            and w not in HEADER_NOISE
            and w not in MONTH_TOKENS
            and w not in WORD_DIGITS
            and not _WORD_DIGIT_RUN_RE.match(w)
        ]

        words = [
            w if w in PUC_SUBJECTS_WHITELIST else (_fuzzy_subject_match(w) or w)
            for w in words
        ]
        s = " ".join(words).strip()
        if s == "MATHEMATIC":
            s = "MATHEMATICS"
        if s.startswith("POLITICAL") and "SCIENCE" not in s:
            s = "POLITICAL SCIENCE"
        if s.startswith("BUSI") and s != "BUSINESS STUDIES":

            s = "BUSINESS STUDIES"
        return s

    def _words_to_number(text_upper):
        tokens = re.findall(r"[A-Z]+", text_upper)
        digits = "".join(WORD_DIGITS[t] for t in tokens if t in WORD_DIGITS)
        return int(digits) if digits else None   
    all_text_upper = " ".join(lines).upper()
    candidate_regs = re.findall(r"\b(\d{5,7})\b", all_text_upper)
    fused_regs = re.findall(r"(\d{5,7})(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)", all_text_upper)
    candidate_regs.extend(fused_regs)

    reg_no = None
    if candidate_regs:
        from collections import Counter
        counts = Counter(candidate_regs)
        most_common, freq = counts.most_common(1)[0]
        if freq >= 2:
            reg_no = most_common
   
    FUSED_REG_MONTH_RE = re.compile(
        r"\d{4,7}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z0-9]{2,6}"
    )

    MONTH_YEAR_RE = re.compile(
        r"(?:\d{4,7})?"
        r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\d]{2,4}"
    )

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        line_upper = line.upper()

        if not _has_subject_kw(line_upper):
            i += 1
            continue
       
        is_core = _has_subject_kw(line_upper)
        next_is_fused = (
            i + 1 < len(lines) and
            bool(FUSED_REG_MONTH_RE.match(lines[i + 1].strip().upper()))
        )

        line_for_nums = line
        has_fused = bool(FUSED_REG_MONTH_RE.search(line_upper))
        if has_fused:
            line_for_nums = FUSED_REG_MONTH_RE.sub(" ", line_upper)
        elif reg_no and reg_no in line:
            line_for_nums = line.replace(reg_no, " ")

        nums_excl_reg = [int(n) for n in re.findall(r"\b\d{1,3}\b", line_for_nums)]
        has_month_year = bool(MONTH_YEAR_RE.search(line_upper)) or has_fused

        if has_fused or (reg_no and reg_no in line) or has_month_year:

            if len(nums_excl_reg) >= 2:
                mx, obt = nums_excl_reg[0], nums_excl_reg[1]
                if mx in (50, 70, 75, 80, 90, 100, 125, 150) and 0 <= obt <= mx:
                    words_val = _words_to_number(line_upper)
                    if words_val is None and i + 1 < len(lines) and not _has_subject_kw(lines[i + 1].upper()):
                        words_val = _words_to_number(lines[i + 1].upper())
                    if words_val is not None and words_val != obt:
                        obt = words_val
                    text_part = _subject_label_prefix(line_for_nums)
                    subj = _clean_subj(text_part)
                    if subj and _has_subject_kw(subj) and subj not in seen_subjects:
                        seen_subjects.add(subj)
                        subjects.append({"subject": subj, "marks": obt, "certainty": "High"})
                    i += 1
                    continue
            words_val = _words_to_number(line_upper)
            if words_val is None and i + 1 < len(lines) and not _has_subject_kw(lines[i + 1].upper()):
                words_val = _words_to_number(lines[i + 1].upper())
            recovered = None
            consumed_next = False
            if words_val is not None and 0 <= words_val <= 100:
                recovered = words_val
            elif len(nums_excl_reg) >= 2:
                dup = next((n for n in nums_excl_reg
                            if nums_excl_reg.count(n) >= 2 and 0 <= n <= 100), None)
                if dup is not None:
                    recovered = dup
                else:
                    small = [n for n in nums_excl_reg if 0 <= n <= 100]
                    if small:
                        recovered = small[-1]
            elif len(nums_excl_reg) == 1 and 0 <= nums_excl_reg[0] <= 100:
                recovered = nums_excl_reg[0]
            elif not nums_excl_reg and i + 1 < len(lines):
                nxt_line = lines[i + 1].strip()
                if not _has_subject_kw(nxt_line.upper()):
                    nxt_nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", nxt_line)]
                    small = [n for n in nxt_nums if 0 <= n <= 100]
                    if small:
                        dup = next((n for n in small if small.count(n) >= 2), None)
                        recovered = dup if dup is not None else small[-1]
                        consumed_next = True

            if recovered is not None:
                text_part = _subject_label_prefix(line_for_nums)
                subj = _clean_subj(text_part)
                if subj and _has_subject_kw(subj) and subj not in seen_subjects:
                    seen_subjects.add(subj)
                    subjects.append({"subject": subj, "marks": recovered, "certainty": "Low"})
                i += 2 if consumed_next else 1
                continue

        if next_is_fused and is_core:

            collect_nums = []
            collect_words = ""
            j = i + 2
            while j < len(lines) and len(collect_nums) < 3:
                tok = lines[j].strip()
                tok_nums = re.findall(r"\b\d{1,3}\b", tok)
                if tok_nums:
                    collect_nums.extend(int(n) for n in tok_nums)
                elif re.search(r"[A-Z]{3,}", tok.upper()):

                    if not collect_words:
                        collect_words = tok.upper()
                    elif _words_to_number(tok.upper()) is not None:
                        collect_words = tok.upper()
                    else:
                        break
                j += 1

            if collect_nums:

                obt = collect_nums[-1] if len(collect_nums) == 1 else collect_nums[1] if len(collect_nums) >= 2 else None
                mx = collect_nums[0] if len(collect_nums) >= 2 else None
                if mx and mx in (50, 70, 75, 80, 90, 100, 125, 150) and obt is not None and 0 <= obt <= mx:

                    words_val = _words_to_number(collect_words) if collect_words else None
                    if words_val is not None and words_val != obt:
                        obt = words_val
                    subj = _clean_subj(_subject_label_prefix(line))
                    if subj and _has_subject_kw(subj) and subj not in seen_subjects:
                        seen_subjects.add(subj)
                        subjects.append({"subject": subj, "marks": obt, "certainty": "High"})
                    i += 1
                    continue
            if len(nums_excl_reg) >= 2:
                mx, obt = nums_excl_reg[0], nums_excl_reg[1]
                if mx in (50, 70, 75, 80, 90, 100, 125, 150) and 0 <= obt <= mx:

                    words_val = _words_to_number(line_upper)
                    if words_val is None and i + 1 < len(lines) and not _has_subject_kw(lines[i + 1].upper()):
                        words_val = _words_to_number(lines[i + 1].upper())
                    if words_val is not None and words_val != obt:

                        obt = words_val

                    text_part = _subject_label_prefix(line)
                    subj = _clean_subj(text_part)
                    if subj and _has_subject_kw(subj) and subj not in seen_subjects:
                        seen_subjects.add(subj)
                        subjects.append({"subject": subj, "marks": obt, "certainty": "High"})
                    i += 1
                    continue

        line_nums = [int(m) for m in re.findall(r"\b\d{1,3}\b", line)]
        if len(line_nums) >= 6:
            text_part = re.sub(r"\d+", "", line)
            text_part = re.sub(r"[^A-Za-z\s]", " ", text_part).strip()
            if _has_subject_kw(text_part):
                mark = _try_puc_pattern(line_nums)
                if mark is not None:
                    subj = _clean_subj(text_part)
                    if subj and _has_subject_kw(subj) and subj not in seen_subjects:
                        seen_subjects.add(subj)
                        subjects.append({"subject": subj, "marks": mark, "certainty": "High"})
            i += 1
            continue

        words_upper = re.findall(r"[A-Z]+", line_upper)
        if words_upper and all(w in HEADER_NOISE for w in words_upper):
            i += 1
            continue

        window_nums = list(line_nums)
        for look in range(i + 1, min(i + 9, len(lines))):
            look_line = lines[look].strip()
            if _has_subject_kw(look_line.upper()):
                break
            if look_line and not re.search(r"\d", look_line):
                if len(re.sub(r"[^A-Za-z]", "", look_line)) >= 4:
                    break
            window_nums.extend(int(m) for m in re.findall(r"\b\d{1,3}\b", look_line))
            if len(window_nums) >= 6:
                break

        mark = _try_puc_pattern(window_nums) if len(window_nums) >= 6 else None
        certainty = "High"
        if mark is None and len(window_nums) == 5:
            mark = _try_puc_pattern_partial(window_nums)
            if mark is not None:
                certainty = "Medium"
        if mark is None and len(window_nums) >= 2:
            t, iv = window_nums[0], window_nums[1]
            if 10 <= t <= 100 and 10 <= iv <= 50:
                mark = t + iv
                certainty = "Low"

        if mark is not None:
            subj = _clean_subj(line)
            if subj and _has_subject_kw(subj) and subj not in seen_subjects:
                seen_subjects.add(subj)
                subjects.append({"subject": subj, "marks": mark, "certainty": certainty})

        i += 1

    return subjects

CORE_10TH_SUBJECTS = [
    "KANNADA", "SANSKRIT", "HINDI", "ENGLISH", "MATHEMATICS", "MATHS",
    "SCIENCE", "SOCIAL SCIENCE", "SOCIAL STUDIES", "SOCIAL", "URDU",
    "TAMIL", "TELUGU", "MARATHI", "COMPUTER", "COMPUTER APPLICATIONS",
    "PHYSICAL EDUCATION", "ART EDUCATION", "PHYSICS", "CHEMISTRY", "BIOLOGY",
]

def _is_recognized_subject(name):
    name_u = name.upper()
    for core in CORE_10TH_SUBJECTS:
        if core in name_u:
            return True

        if len(core) >= 5:
            window = name_u[:len(core) + 3]
            if difflib.SequenceMatcher(None, window, core).ratio() >= 0.72:
                return True
    return False
LANGUAGE_PREFIXES_10TH = ["FIRST LANGUAGE", "SECOND LANGUAGE", "THIRD LANGUAGE"]

HEADER_NOISE_10TH = [
    "SUBJECT", "MARKS", "MAX", "MIN", "OBTAINED", "GRADE", "RESULT",
    "TOTAL", "GRAND TOTAL", "CLASS OBTAINED", "DISTINCTION", "REGISTER",
    "BOARD", "EXAMINATION", "CERTIFICATE", "GOVERNMENT", "SECRETARY",
    "MEDIUM OF INSTRUCTION", "DATE OF BIRTH", "MOTHER", "FATHER",
    "NAME OF THE SCHOOL", "PASS IN EXAMINATION", "AGGREGATE", "SCHOOL",
    "CERTIFY", "GUARDIAN",

    "DATED", "PRESCRIBED", "CO-SCHOLASTIC", "CO SCHOLASTIC", "GRADING",
    "DISCIPLINE AREA", "ACHIEVEMENTS", "ABBREVIATION", "CONTROLLER",
    "SIGNATURE", "BY THE", "ISSUED", "AS PER FORMAT",
]

RESULT_CLASS_WORDS = {
    "FIRST", "SECOND", "THIRD", "PASS", "FAIL", "DISTINCTION",
    "FIRSTCLASS", "SECONDCLASS", "THIRDCLASS",
}

def _is_10th_header_or_noise(line_upper):

    has_early_core = any(c in line_upper for c in CORE_10TH_SUBJECTS)

    if "%" in line_upper and not has_early_core:

        return True

    if any(h in line_upper for h in HEADER_NOISE_10TH):
        if has_early_core:
            return False
        if not any(p in line_upper for p in LANGUAGE_PREFIXES_10TH):
            return True
    return False

def _clean_10th_subject_name(raw_text):
    text = re.sub(r"\d+", "", raw_text)
    text = re.sub(r"[^A-Za-z\s:]", " ", text)
    text = text.replace(":", " ")
    text = re.sub(r"\s+", " ", text).strip().upper()

    prefix = None
    for p in LANGUAGE_PREFIXES_10TH:
        if text.startswith(p):
            prefix = p
            text = text[len(p):].strip()
            break

    if "SOCIAL" in text:
        text = "SOCIAL SCIENCE"
    elif "SCIENCE" in text:
        text = "SCIENCE"
    elif "MATHEMATICS" in text or text == "MATHS":
        text = "MATHEMATICS"
    elif "KANNADA" in text:
        text = "KANNADA"
    elif "SANSKRIT" in text:
        text = "SANSKRIT"
    elif "HINDI" in text:
        text = "HINDI"
    elif "ENGLISH" in text:
        text = "ENGLISH"

    if prefix and prefix.split()[0] not in text:
        text = f"{prefix} {text}".strip()

    return text

_NUM_WORD_ONES = {
    "ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
    "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10,
    "ELEVEN": 11, "TWELVE": 12, "THIRTEEN": 13, "FOURTEEN": 14,
    "FIFTEEN": 15, "SIXTEEN": 16, "SEVENTEEN": 17, "EIGHTEEN": 18,
    "NINETEEN": 19,
}
_NUM_WORD_TENS = {
    "TWENTY": 20, "THIRTY": 30, "FORTY": 40, "FIFTY": 50,
    "SIXTY": 60, "SEVENTY": 70, "EIGHTY": 80, "NINETY": 90,
}

def _english_number_words_to_int(text_upper):
    tokens = re.findall(r"[A-Z]+", text_upper)
    value = 0
    matched = False
    for tok in tokens:
        if tok == "HUNDRED" and matched:
            value *= 100
        elif tok in _NUM_WORD_TENS:
            value += _NUM_WORD_TENS[tok]
            matched = True
        elif tok in _NUM_WORD_ONES:
            value += _NUM_WORD_ONES[tok]
            matched = True
        elif tok == "AND":
            continue
        elif matched:
            break
    return value if matched else None

def _pick_obtained_mark(nums, has_grade=False, words_val=None):

    if words_val is not None and 0 <= words_val <= 200 and len(nums) < 3:
        return words_val

    if not nums:
        return None

    if all(n == 0 for n in nums):
        return None

    if len(nums) == 1:
        candidate = nums[0]
        
        if candidate in (100, 125, 150):
            return None
        return candidate if 0 <= candidate <= 200 else None

    if len(nums) == 2:
        mx, obt = nums[0], nums[1]
        if 0 <= obt <= 200 and (mx == 0 or obt <= mx + 2):
            return obt
        return obt if 0 <= obt <= 200 else None

    if len(nums) == 3:
        mx, mn, obt = nums[0], nums[1], nums[2]
        if 0 <= obt <= 200 and (mx == 0 or obt <= mx + 2):
            return obt
        return obt if 0 <= obt <= 200 else None

    mx, mn, obt = nums[-3], nums[-2], nums[-1]
    if 0 <= obt <= 200 and (mx == 0 or obt <= mx + 2):
        return obt
    return nums[-1] if 0 <= nums[-1] <= 200 else None

def _extract_10th_board_subjects(lines):
    final_subjects = []
    pending_prefix = None

    after_additional_marker = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        line_upper = line.upper()

        if "ADDITIONAL SUBJECT" in line_upper:
            after_additional_marker = True
            i += 1
            continue

        if _is_10th_header_or_noise(line_upper):
            i += 1
            continue

        bare_prefix = next(
            (p for p in LANGUAGE_PREFIXES_10TH
             if line_upper.rstrip(":").strip() == p), None
        )
        if bare_prefix:
            pending_prefix = bare_prefix
            i += 1
            continue

        line_clean = line
        noise_markers = ["NO CLASS", "NC ABOVE", "DISTINCTION", "FIRST CLASS",
                         "SECOND CLASS", "PASS IN", "30% MIN"]
        for nm in noise_markers:
            if nm.upper() in line.upper():
                cut = line.upper().index(nm.upper())
                line_clean = line[:cut].strip()
                break

        if "%" in line_clean:
            pct_pos = line_clean.index("%")
            if pct_pos > 15:
                line_clean = line_clean[:pct_pos].strip()

        
        code_prefix_match = re.match(r"^\s*\d{2,4}\s+(?=[A-Za-z])", line_clean)
        if code_prefix_match:
            line_clean = line_clean[code_prefix_match.end():]

        nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", line_clean)]
        has_grade = bool(re.search(r"\b[ABCDEFSP][+\-#]?\b", line_upper))

        parts = re.split(r"(\b\d{2,3}\b)", line_clean, maxsplit=1)
        subject_text = parts[0]
        words_val = _english_number_words_to_int(line_clean[len(subject_text):].upper())

        is_core = any(c in line_upper for c in CORE_10TH_SUBJECTS)

        if not subject_text.strip() or not re.search(r"[A-Za-z]{3,}", subject_text):
            i += 1
            continue

        if not is_core and not nums:
            i += 1
            continue

        subject = _clean_10th_subject_name(subject_text)

        if pending_prefix and pending_prefix.split()[0] not in subject:
            subject = f"{pending_prefix} {subject}".strip()
        pending_prefix = None

        core_matches = [c for c in CORE_10TH_SUBJECTS if c in subject_text.upper()]
        if len(core_matches) >= 2 and nums:

            first_subj_text = subject_text.upper()
            for cm in core_matches[1:]:

                idx_cut = first_subj_text.find(cm)
                if idx_cut > 0:
                    first_subj_text = first_subj_text[:idx_cut]
            first_subj = _clean_10th_subject_name(first_subj_text)
            if first_subj and len(first_subj) >= 2:
                existing = next((s for s in final_subjects
                                  if s["subject"].upper() == first_subj.upper()), None)
                if not existing:
                    final_subjects.append({"subject": first_subj, "marks": None,
                                            "_from_additional": after_additional_marker})

        if not subject or len(subject) < 2:
            i += 1
            continue

        if subject.replace(" ", "") in RESULT_CLASS_WORDS:
            i += 1
            continue

        marks = _pick_obtained_mark(nums, has_grade=has_grade, words_val=words_val)

        if marks is None and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            next_upper = next_line.upper()
            next_nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", next_line)]
            next_has_text = bool(re.search(r"[A-Za-z]{3,}", next_line))
            next_is_core = any(c in next_upper for c in CORE_10TH_SUBJECTS)

            if next_is_core and not next_nums:

                if i + 2 < len(lines):
                    line2 = lines[i + 2].strip()
                    nums2 = [int(n) for n in re.findall(r"\b\d{1,3}\b", line2)]
                    has_text2 = bool(re.search(r"[A-Za-z]{3,}", line2))
                    if nums2 and not has_text2:

                        existing = next((s for s in final_subjects
                                          if s["subject"].upper() == subject.upper()), None)
                        if not existing:
                            final_subjects.append({"subject": subject, "marks": None,
                                                    "_from_additional": after_additional_marker})
                        i += 1
                        continue

            if next_is_core and next_nums:

                existing = next((s for s in final_subjects
                                  if s["subject"].upper() == subject.upper()), None)
                if not existing:
                    final_subjects.append({"subject": subject, "marks": None,
                                            "_from_additional": after_additional_marker})
                i += 1
                continue

            if next_nums and not next_has_text:

                marks = _pick_obtained_mark(next_nums)
            elif next_nums:
                next_words_val = _english_number_words_to_int(next_line.upper())
                marks = _pick_obtained_mark(next_nums, words_val=next_words_val)

        if marks is not None and len(nums) == 2:
            mx, candidate = nums[0], nums[1]
            if mx in (100, 125, 150) and candidate in (35, 44, 40, 28, 30):
                marks = None

        
        if subject.upper() in ("THIRD LANGUAGE", "SECOND LANGUAGE", "FIRST LANGUAGE"):
            if nums and all(n in (100, 125, 150, 80, 50) for n in nums):
                pending_prefix = subject.upper()
                i += 1
                continue

        existing = next((s for s in final_subjects
                          if s["subject"].upper() == subject.upper()), None)
        if existing:
            if marks is not None and existing.get("marks") is None:
                existing["marks"] = marks
        else:
            final_subjects.append({"subject": subject, "marks": marks,
                                    "_from_additional": after_additional_marker})

        i += 1

    full_names = {s["subject"].upper() for s in final_subjects if s.get("marks") is not None}
    final_subjects = [
        s for s in final_subjects
        if s.get("marks") is not None
        or not any(s["subject"].upper() in fn for fn in full_names)
    ]

    none_subjects = [s for s in final_subjects if s.get("marks") is None]
    known_subjects = [s for s in final_subjects if s.get("marks") is not None]

    if none_subjects and known_subjects:
        total_candidate = None
        for l in lines:
            lu = l.upper()
            if "TOTAL" in lu and "IN WORDS" not in lu and "%" not in lu:
                t_nums = [int(n) for n in re.findall(r"\b\d{3,}\b", l)]

                valid = [n for n in t_nums if 100 <= n <= 1500]
                if valid:
                    total_candidate = valid[-1]
                    break

        if total_candidate:
            known_sum = sum(s["marks"] for s in known_subjects)
            diff = total_candidate - known_sum

            if len(none_subjects) == 1 and 0 <= diff <= 200:

                none_subjects[0]["marks"] = diff

    
    final_subjects = [
        s for s in final_subjects
        if _is_recognized_subject(s["subject"]) or s.get("_from_additional")
    ]

    return final_subjects

ICSE_TOP_SUBJECTS = [
    ("HISTORY", "HISTORY, CIVICS & GEOGRAPHY"),
    ("ENGLISH", "ENGLISH"),
    ("HINDI", "HINDI"),
    ("MATHEMATICS", "MATHEMATICS"),
    ("SCIENCE", "SCIENCE"),
    ("ECONOMIC", "ECONOMIC APPLICATIONS"),
    ("COMPUTER APPLICATIONS", "COMPUTER APPLICATIONS"),
    ("COMMERCIAL APPLICATIONS", "COMMERCIAL APPLICATIONS"),
    ("TECHNICAL DRAWING", "TECHNICAL DRAWING APPLICATIONS"),
    ("PHYSICAL EDUCATION", "PHYSICAL EDUCATION"),
    ("ENVIRONMENTAL APPLICATIONS", "ENVIRONMENTAL APPLICATIONS"),
    ("ECONOMICS APPLICATIONS", "ECONOMICS APPLICATIONS"),
    ("SANSKRIT", "SANSKRIT"),
    ("FRENCH", "FRENCH"),
    ("MALAYALAM", "MALAYALAM"),
    ("TAMIL", "TAMIL"),
    ("TELUGU", "TELUGU"),
    ("KANNADA", "KANNADA"),
    ("MARATHI", "MARATHI"),
    ("BENGALI", "BENGALI"),
    ("GUJARATI", "GUJARATI"),
    ("PUNJABI", "PUNJABI"),
    ("URDU", "URDU"),
    ("ODIA", "ODIA"),
    ("ASSAMESE", "ASSAMESE"),
    ("ARABIC", "ARABIC"),
    ("GERMAN", "GERMAN"),
    ("SPANISH", "SPANISH"),
]

ICSE_CHILD_KEYWORDS = [
    "ENGLISH LANGUAGE", "LITERATURE IN ENGLISH", "HISTORY & CIVICS",
    "HISTORY AND CIVICS", "GEOGRAPHY", "PHYSICS", "CHEMISTRY", "BIOLOGY",
]

ICSE_HEADER_NOISE = [
    "SUBJECTS", "TOTAL MARKS", "PERCENTAGE MARKS", "MAX. MARKS", "MAX MARKS",
    "INTERNAL ASSESSMENT", "SUPW", "COMMUNITY SERVICE", "GRADE",
    "STATEMENT OF MARKS", "COUNCIL FOR", "CERTIFICATE OF SECONDARY EDUCATION",
    "UNIQUE ID", "DAUGHTER OF", "SON OF", "RESULT", "DATE OF BIRTH",
    "DATE OF DECLARATION", "REGN", "PASS MARK", "NO DIVISIONS",
]

_ICSE_NUMBER_WORDS = {
    "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
    "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
    "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY", "THIRTY",
    "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY", "HUNDRED", "AND",
}

def _is_icse_aggregate_line(line_upper):
    tokens = re.findall(r"[A-Z]+|\d+", line_upper)
    if not tokens:
        return False
    return all(t.isdigit() or t in _ICSE_NUMBER_WORDS for t in tokens)

def _is_icse_percentage_line(line_upper):
   
    tokens = re.findall(r"[A-Z]+|\d+", line_upper)
    if not tokens:
        return False
    has_digit = any(t.isdigit() for t in tokens)
    has_word = any(t in _ICSE_NUMBER_WORDS for t in tokens)
    return has_digit and has_word

_ICSE_WORD_ALT = "|".join(sorted(_ICSE_NUMBER_WORDS, key=len, reverse=True))
_ICSE_PERCENT_NUM_RE = re.compile(
    r"(\d{1,3})(?:\s+\b(?:" + _ICSE_WORD_ALT + r")\b)+\s*$"
)

def _extract_icse_percentage_value(line_upper):
    
    m = _ICSE_PERCENT_NUM_RE.search(line_upper)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d{1,3}", line_upper)
    return int(nums[-1]) if nums else None

def _extract_icse_subjects(lines):
   
    final_subjects = []
    current = None
    locked_ids = set()

    def _find(name):
        return next((s for s in final_subjects if s["subject"] == name), None)

    def _lookahead_subject(idx):
        for j in range(idx + 1, min(idx + 4, len(lines))):
            nxt = lines[j].strip().upper()
            if not nxt:
                continue
            if any(h in nxt for h in ICSE_HEADER_NOISE):
                continue
            hit = next(
                (canon for key, canon in ICSE_TOP_SUBJECTS if nxt.startswith(key)),
                None,
            )
            if hit is not None:
                return hit
            if re.search(r"[A-Za-z]{3,}", nxt):
                return None
        return None

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        up = line.upper()

        if any(h in up for h in ICSE_HEADER_NOISE):
            continue

       
        top_hit = next(
            (canon for key, canon in ICSE_TOP_SUBJECTS if up.startswith(key)),
            None,
        )
        if top_hit is not None:
            entry = _find(top_hit)
            if not entry:
                entry = {"subject": top_hit, "marks": None}
                final_subjects.append(entry)
            current = entry

        if _is_icse_percentage_line(up):

            val = _extract_icse_percentage_value(up)
            target = current
            if top_hit is None and (target is None or id(target) in locked_ids):
                ahead = _lookahead_subject(idx)
                if ahead is not None:
                    entry = _find(ahead)
                    if not entry:
                        entry = {"subject": ahead, "marks": None}
                        final_subjects.append(entry)
                    target = entry
                    current = entry
            if val is not None and target is not None and id(target) not in locked_ids:
                target["marks"] = val
                locked_ids.add(id(target))
            continue

        if _is_icse_aggregate_line(up):

            if current is not None and id(current) not in locked_ids:
                nums = [int(n) for n in re.findall(r"\d{1,3}", up)]
                if nums:
                    current["marks"] = nums[0]
            continue

        if not re.search(r"[A-Za-z]{3,}", line):
            continue

        if top_hit is None and any(k in up for k in ICSE_CHILD_KEYWORDS):

            continue

        if top_hit is None:
            continue

        if id(current) not in locked_ids:
            nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", up)]
            if nums:
                cand = _pick_obtained_mark(nums)
                if cand is not None:
                    current["marks"] = cand

    return final_subjects

def extract_subjects_from_text(lines, raw_data=None, image_width=0, board_type=None):
    if board_type == "VTU":
        return _extract_vtu_blocks(lines)

    if board_type in ("KARNATAKA PU BOARD", "KARNATAKA PU BOARD (NEW)"):
        return _extract_puc_subjects(lines)

    if board_type == "ICSE":
        return _extract_icse_subjects(lines)

    return _extract_10th_board_subjects(lines)

def extract_subjects(image, lines=None):

    if not lines:
        return []
        
    return extract_subjects_from_text(lines)

def extract_total(lines):

    processed_lines = []
    for line in lines:
        processed_lines.append(re.sub(r"(\d{3,4})(\d{3})(\d{3})", r"\1 \2 \3", line.upper()))

    for i, line in enumerate(processed_lines):
        if "TOTAL" in line or "GRAND" in line or re.search(r"T[O0A]TAL", line):

            if any(hw in line for hw in ["MAX", "MIN", "SUBJECT", "SCHOLASTIC", "INTERNAL", "EXTERNAL"]):
                continue
                

            nums = [int(n) for n in re.findall(r"\b\d{3,4}\b", line) if 100 <= int(n) <= 1500]
            if nums:

                 return nums[-1]
            

            for j in range(i + 1, min(i + 3, len(processed_lines))):
                next_nums = [int(n) for n in re.findall(r"\b\d{3,4}\b", processed_lines[j])
                             if 200 <= int(n) <= 1500]
                if next_nums:
                    return next_nums[-1]

    full_text = " ".join(processed_lines)
    nums = re.findall(r"\b\d{3,4}\b", full_text)
    nums = [int(n) for n in nums if 200 <= int(n) <= 1500]

    if len(nums) >= 2:
        from collections import Counter
        counts = Counter(nums)
        most_common, freq = counts.most_common(1)[0]
        if freq >= 2:
            return most_common
        return nums[-1]
    
    return nums[0] if nums else None

def extract_result(lines):

    for t in lines:
        t = t.upper()

        if "PASS" in t:
            return "PASS"
        if "FAIL" in t:
            return "FAIL"

    return None

def _group_texts_by_y(raw_data, image_height=0, image_width=0):
    if not raw_data:
        return []

    y_threshold = 15
    if image_height > 0:
        y_threshold = max(4, int(image_height * 0.01))

    x_threshold = 1000
    if image_width > 0:
        x_threshold = max(50, int(image_width * 0.20))

    sorted_data = sorted(raw_data, key=lambda x: x["box"][0][1])
    
    grouped_lines = []
    current_group = [sorted_data[0]]
    
    for i in range(1, len(sorted_data)):
        prev_item = current_group[-1]
        curr_item = sorted_data[i]
        

        if abs(curr_item["box"][0][1] - prev_item["box"][0][1]) <= y_threshold:
            current_group.append(curr_item)
        else:

            current_group.sort(key=lambda x: x["box"][0][0])
            
            chunk = [current_group[0]]
            for j in range(1, len(current_group)):
                prev_x_end = chunk[-1]["box"][1][0]
                curr_x_start = current_group[j]["box"][0][0]
                
                if (curr_x_start - prev_x_end) > x_threshold:
                    grouped_lines.append(" ".join([item["text"] for item in chunk]).upper())
                    chunk = [current_group[j]]
                else:
                    chunk.append(current_group[j])
            
            grouped_lines.append(" ".join([item["text"] for item in chunk]).upper())
            current_group = [curr_item]
            

    current_group.sort(key=lambda x: x["box"][0][0])
    chunk = [current_group[0]]
    for j in range(1, len(current_group)):
        if (current_group[j]["box"][0][0] - chunk[-1]["box"][1][0]) > x_threshold:
            grouped_lines.append(" ".join([item["text"] for item in chunk]).upper())
            chunk = [current_group[j]]
        else:
            chunk.append(current_group[j])
    grouped_lines.append(" ".join([item["text"] for item in chunk]).upper())
    
    return grouped_lines

def detect_board(lines):
    text = " ".join(lines).upper()

    if ("SCHOOL EXAMINATION AND ASSESSMENT" in text
            or "EXAMINATION AND ASSESSMENT BOARD" in text
            or "KSEAB" in text):
        return "KARNATAKA PU BOARD (NEW)"

    if ("PRE-UNIVERSITY" in text or "PRE UNIVERSITY" in text
            or "PUCOLLEGE" in text or "PU COLLEGE" in text
            or ("PUC" in text and "EXAMINATION" in text)):
        return "KARNATAKA PU BOARD"
    if "DEPARTMENT OF PRE-UNIVERSITY" in text:
        return "KARNATAKA PU BOARD"

    
    if "CBSE" in text or "CENTRAL BOARD" in text:
        return "CBSE"

    if "ICSE" in text or "INDIAN CERTIFICATE" in text or "COUNCIL FOR THE INDIAN SCHOOL" in text:
        return "ICSE"

    sslc_fuzzy = re.search(r"S\.?\s*S\.?\s*L\.?\s*C", text) is not None

    if (
        "SECONDARY EDUCATION" in text
        or "SSLC" in text
        or sslc_fuzzy
        or ("KARNATAKA" in text and ("BOARD" in text or "SECONDARY" in text))

        or ("GOVERNMENT OF KARNATAKA" in text and "KANNADA" in text)
    ):
        return "KARNATAKA STATE BOARD"

    if "UNIVERSITY" in text or "VTU" in text or "VISVESVARAYA" in text:
        return "VTU"

    return None

def extract_register_info(lines):
    full_text = " ".join(lines).upper()
    

    reg_patterns = [
        r"(?:[A-Z0-9\-/]*\s?)?R[ECG]G?IST?ER(?:\.?\s?N[O0]\.?)?[:\-\s]*([A-Z0-9\-\/]{5,20})",

        r"(?:[A-Z0-9\-/]*\s?)?R[O0]L{0,2}\.?\s?N[O0]\.?[:\-\s]*([A-Z0-9\-\/]{5,20})",
        r"\bUSN[:\-\s]*([A-Z0-9\-\/]{5,20})",
        r"\bSEAT\s?N[O0]\.?[:\-\s]*([A-Z0-9\-\/]{5,20})",

        r"\bUNIQUE\s?I?D[:\-\s]*([A-Z0-9\-\/]{5,20})",
    ]
    

    USN_REGEX = r"[1I][A-Z]{2}\d{2}[A-Z]{2,3}[A-Z0-9]{2,3}"
    
    for pattern in reg_patterns:
        match = re.search(pattern, full_text)
        if match:
            val = match.group(1).strip()

            if not re.search(r"\d", val):
                continue

            if re.match(USN_REGEX, val):

                if val.startswith('I'):
                    val = '1' + val[1:]
                return val, "USN", None
            return val, "REG_NO", None

    sslc_reg_match = re.search(r"\b(20\d{9})\b", full_text)
    if sslc_reg_match:
        return sslc_reg_match.group(1), "REG_NO", None

    puc_fused = re.findall(
        r"(\d{5,7})\s?(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)",
        full_text
    )
    if puc_fused:
        from collections import Counter
        most_common_reg = Counter(puc_fused).most_common(1)[0][0]
        return most_common_reg, "REG_NO", None

    usn_match = re.search(rf"\b({USN_REGEX})\b", full_text)
    if usn_match:
        val = usn_match.group(1)
        if val.startswith('I'):
            val = '1' + val[1:]
        return val, "USN", None

    header_raw = " ".join(lines[:15]).upper()
    MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
    
    standalone_match = re.search(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{5,15}\b", header_raw)
    if standalone_match:
        cand = standalone_match.group(0)

        if not any(m in cand for m in MONTHS) and not re.match(r"^(?:20\d{2}|19\d{2})$", cand):
            return cand, "REG_NO", None

    sem_match = re.search(r"(?:SEM(?:ESTER)?|SESSION)\s?[:\-]?\s?\b([0-9IVX]{1,2})\b", full_text)
    semester = sem_match.group(1) if sem_match else None
    if semester and semester.isdigit() and int(semester) > 12:
        semester = None

    return None, None, semester

def extract_gpa(lines):
    full_text = " ".join(lines).upper()
    
    sgpa = None
    cgpa = None
    

    sgpa_match = re.search(r"SGPA\s?[:\-]?\s?(\d{1,2}\.\d{1,3})", full_text)
    if sgpa_match:
        sgpa = sgpa_match.group(1)
        
    cgpa_match = re.search(r"CGPA\s?[:\-]?\s?(\d{1,2}\.\d{1,3})", full_text)
    if cgpa_match:
        cgpa = cgpa_match.group(1)
        

    if not sgpa or not cgpa:

        decimals = re.findall(r"\b\d{1,2}\.\d{1,3}\b", full_text)
        if "SGPA" in full_text and "CGPA" in full_text and len(decimals) >= 2:

             if not sgpa: sgpa = decimals[-2]
             if not cgpa: cgpa = decimals[-1]
             
    return sgpa, cgpa

_MONTH_YEAR_RE = re.compile(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s?20\d{2}")

_CONSENSUS_STOPWORDS = {
    "MONTH", "YEAR", "NAME", "CANDIDATE", "CANDIDATES", "REGISTER",
    "MOTHER", "FATHER", "MOTHERS", "FATHERS", "HUSBAND", "CERTIFICATE",
    "EXAMINATION", "EDUCATION", "DEPARTMENT", "GOVERNMENT", "DETAILS",
    "PASSED", "COMPLETED", "KARNATAKA", "UNIVERSITY", "SECOND", "COURSE",
}

_CONSENSUS_BOILERPLATE_SUBSTRINGS = (
    "THIS IS TO", "FOLLOWING DETAILS", "SECOND YEAR", "PRE-UNIVERSITY",
    "PRE UNIVERSITY", "SCHOOL EXAMINATION", "MARKS STATEMENT",
)

def _consensus_header_zone(lines):
    for i, line in enumerate(lines):
        if _has_puc_subject_kw(line.upper()):
            return lines[:i]
    return lines[:20]

def _has_puc_subject_kw(text_upper):
    subjects = {
        "ENGLISH", "HINDI", "KANNADA", "SANSKRIT", "URDU", "TAMIL", "TELUGU",
        "MATHEMATICS", "MATHS", "PHYSICS", "CHEMISTRY", "BIOLOGY", "COMPUTER",
        "HISTORY", "GEOGRAPHY", "ECONOMICS", "POLITICAL", "SOCIOLOGY",
        "ACCOUNTANCY", "BUSINESS", "ELECTRONICS",
    }
    return any(s in text_upper for s in subjects)

def _longest_common_token_run(tokens_a, tokens_b):
    best = []
    for i in range(len(tokens_a)):
        for j in range(len(tokens_b)):
            k = 0
            while (i + k < len(tokens_a) and j + k < len(tokens_b)
                   and tokens_a[i + k] == tokens_b[j + k]):
                k += 1
            if k > len(best):
                best = tokens_a[i:i + k]
    return best

def _consensus_student_name(zone_a, zone_b, known_values=None):
    known_upper = [v.strip().upper() for v in (known_values or []) if v]

    def already_known(text):
        return any(difflib.SequenceMatcher(None, text, k).ratio() >= 0.8 for k in known_upper)

    def pre_monthyear_tokens(zone):
        for line in zone:
            up = line.upper()
            m = _MONTH_YEAR_RE.search(up)
            if m:
                toks = re.findall(r"[A-Z]{1,15}", up[:m.start()])
                return [t for t in toks if t not in _CONSENSUS_STOPWORDS]
        return []

    toks_a = pre_monthyear_tokens(zone_a)
    toks_b = pre_monthyear_tokens(zone_b)
    if not toks_a or not toks_b:
        return None

    run = _longest_common_token_run(toks_a, toks_b)
    if run and sum(len(t) for t in run) >= 3:
        candidate = " ".join(run)
        if already_known(candidate):
            return None
        return candidate
    return None

def _consensus_candidate_lines(zone, start_after_monthyear=True):
    start_idx = 0
    if start_after_monthyear:
        for i, line in enumerate(zone):
            if _MONTH_YEAR_RE.search(line.upper()):
                start_idx = i + 1
                break
    cands = []
    for line in zone[start_idx:]:
        up = line.strip().upper()
        if not up or len(up) > 35:
            continue
        if _MONTH_YEAR_RE.search(up):
            continue
        if any(b in up for b in _CONSENSUS_BOILERPLATE_SUBSTRINGS):
            continue
        letters_only = re.sub(r"[^A-Z\s]", "", up).strip()
        letters_only = re.sub(r"\s+", " ", letters_only)
        if len(letters_only) < 3:
            continue
        if any(w in letters_only.replace(" ", "") for w in _CONSENSUS_STOPWORDS):
            continue
        cands.append(letters_only)
    return cands

def _consensus_parent_names(zone_a, zone_b, known_values):
    cand_a = _consensus_candidate_lines(zone_a)
    cand_b = _consensus_candidate_lines(zone_b)
    known_upper = [v.strip().upper() for v in known_values if v]

    def already_known(text):
        return any(difflib.SequenceMatcher(None, text, k).ratio() >= 0.8 for k in known_upper)

    stable = []
    used_b = set()
    for ca in cand_a:
        if len(ca) < 4 or not any(v in ca for v in "AEIOU") or already_known(ca):
            continue
        for j, cb in enumerate(cand_b):
            if j in used_b:
                continue
            if ca == cb:
                used_b.add(j)
                stable.append(ca)
                break
    return stable

def recover_names_by_consensus(lines_a, lines_b, missing_fields, known_values=None):
    known_values = known_values or []
    out = {}
    zone_a = _consensus_header_zone(lines_a)
    zone_b = _consensus_header_zone(lines_b)

    if "student_name" in missing_fields:
        name = _consensus_student_name(zone_a, zone_b, known_values)
        if name:
            out["student_name"] = name

    remaining = [f for f in ("mother_name", "father_name") if f in missing_fields]
    if remaining:
        stable = _consensus_parent_names(zone_a, zone_b, known_values + [out.get("student_name")])
        for field, value in zip(remaining, stable):
            out[field] = value

    return out

_MARKSHEET_HEADER_VOCAB = (
    "SUBJECTS", "SUBJECT", "LANGUAGES", "LANGUAGE", "OPTIONALS", "OPTIONAL",
    "PARTICULARS", "REGISTER", "MONTH", "YEAR", "MARKS", "OBTAINED",
    "FIGURES", "WORDS", "TOTAL", "MAXIMUM", "BOARD", "UNIVERSITY",
    "COLLEGE", "DEPARTMENT", "EXAMINATION", "CERTIFICATE", "CANDIDATE",
    "STUDENT", "PRINCIPAL", "DIRECTOR", "SIGNATURE",
)

def _is_probable_header_noise(text):

    tokens = [t for t in re.findall(r"[A-Z]{3,}", text.upper()) if len(t) >= 6]
    for tok in tokens:
        for header_word in _MARKSHEET_HEADER_VOCAB:
            if difflib.SequenceMatcher(None, tok, header_word).ratio() >= 0.6:
                return True
    return False

def _strip_leading_short_noise_token(text):
    words = text.split()
    if len(words) >= 2 and len(words[0]) <= 2 and len(words[-1]) >= 4:
        return " ".join(words[1:])
    return text

_FUZZY_PARENT_LABEL_DENYLIST = {"MONTH"}

def _fuzzy_token_hit(tok, target, cutoff=0.72):
    if not tok or not target or tok[0] != target[0]:
        return False
    if tok in _FUZZY_PARENT_LABEL_DENYLIST:
        return False
    return difflib.SequenceMatcher(None, tok, target).ratio() >= cutoff

def _is_fuzzy_parent_label(line_upper):
    if re.search(r"\bM[O0]B?ERS\b|\bF[A4]THERS?\b", line_upper):
        return True
    for tok_match in re.finditer(r"[A-Z]{4,}", line_upper):
        tok = tok_match.group(0)
        if any(_fuzzy_token_hit(tok, t) for t in ("FATHER", "FATHERS", "MOTHER", "MOTHERS")):
            return True
    return False

def _find_parent_kw_end(line_upper, literal_pattern, fuzzy_targets):
    m = re.search(literal_pattern, line_upper)
    if m:
        return m.end()
    for tok_match in re.finditer(r"[A-Z]{4,}", line_upper):
        if any(_fuzzy_token_hit(tok_match.group(0), t) for t in fuzzy_targets):
            return tok_match.end()
    return None

def _strip_parent_label(remainder):
    name_kw = re.search(r"NA[MN]E?", remainder[:40])
    if name_kw:
        remainder = remainder[name_kw.end():]
    else:
        remainder = re.sub(r"^['\s]*S?\s*", "", remainder)
    return re.sub(r"^[:\-\s]+", "", remainder).strip()


def _trim_fused_parent_name(text):
    words = text.split()
    if len(words) <= 3:
        return text
    return " ".join(words[-2:])

def extract_personal_info(lines):
    #print("DEBUG extract_personal_info received lines:")
    #for _i, _l in enumerate(lines):
        #print(f"  [{_i}] {_l!r}")

    full_text = "\n".join(lines).upper()
    
    name = None
    father = None
    mother = None
    
    NOISE = ["ADDRESS", "PHOTO", "DATE", "PLACE", "SIGNATURE", "OFFICE", "STAMP", "COLLEGE", "INSTITUTE", "UNIVERSITY", "USN", "US NO", "REGISTER", "FATHWER", "MOTHWR", "STDONT"]

    for i, line in enumerate(lines):
        line = line.upper()
        

        name_fuzzy_hit = (
            "NAME" in line or "NME" in line
            or re.search(r"\bNAM\b|NAM$", line)
        )
        parent_label_hit = any(
            kw in line for kw in
            ["FATHER", "MOTHER", "FATHWER", "MOTHWR", "SCHOOL", "BOARD",
             "UNIVERSITY", "COLLEGE", "INSTITUTE", "OFFICE", "SUBJECT"]
        ) or _is_fuzzy_parent_label(line)
        if name_fuzzy_hit and not parent_label_hit:
            
            labels = ["NAME OF THE STUDENT", "NAME OF THE CANDIDATE", "STUDENT NAME", "CANDIDATE'S NAME", "CANDIDATE NAME", "NME OF THE STUDENT", "NME OF THE CANDIDATE", "NME CT HE STDONT", "NME CT HE STUDENT"]
            

            label_found = None
            for label in labels:
                if label in line:
                    label_found = label
                    break
            
            cand = ""
            if label_found:

                parts = line.split(label_found)
                rem_after = parts[-1].replace(":", "").strip()
                rem_before = parts[0].strip() if len(parts) > 1 else ""

                if len(rem_after) >= 3 and re.search(r"[A-Za-z]{2,}", rem_after):
                    cand = rem_after
                elif len(rem_before) >= 3 and re.search(r"[A-Za-z]{2,}", rem_before):

                    cand = rem_before
                elif i + 1 < len(lines):

                    nxt = re.sub(r"^[\s:.\-,]+", "", lines[i+1].strip())

                    nxt_up = nxt.upper()
                    is_clean_name = (
                        len(nxt) >= 3
                        and re.search(r"[A-Za-z]{2,}", nxt)
                        and not re.search(r"[a-z]{4,}", nxt)
                        and not any(noise in nxt_up for noise in
                                    ["DATE", "RESULT", "MARKS", "TOTAL", "REGISTER", "MONTH"])
                    )
                    if is_clean_name:
                        cand = nxt

                if not cand:
                    for back in range(1, min(4, i + 1)):
                        prev = lines[i - back].strip()
                        prev_up = prev.upper()

                        if any(kw in prev_up for kw in
                               ["GOVERNMENT", "KARNATAKA", "BOARD", "DEPARTMENT",
                                "CERTIFICATE", "EXAMINATION", "REGISTER", "DATE",
                                "RESULF", "RESULT", "MARKS", "TOTAL", "MOTHER",
                                "FATHER", "FATHER'S", "MOTHER'S"]):
                            continue
                        if (re.match(r"^[A-Za-z\s\.]{3,40}$", prev)
                                and len(prev.split()) <= 6
                                and len(prev) >= 3):
                            cand = re.sub(r"^[^A-Za-z]+", "", prev).strip()

                            cand = _strip_leading_short_noise_token(cand)
                            break
            else:

                match = re.search(
                    r"(?:NAME|NAM)\s?[:\-]?\s?([A-Z][A-Z\s]{2,40}?)(?=\s+(?:MARKS|TOTAL|RESULT|INTERNAL|EXTERNAL|SUBJECT|SEMESTER|DATE)\b|$)",
                    line
                )
                if match:
                    cand = match.group(1).strip()

                if not cand and len(line) <= 25 and i + 1 < len(lines):
                    nxt = re.sub(r"^[\s:.\-,]+", "", lines[i + 1].strip())
                    nxt_up = nxt.upper()
                    is_clean_name = (
                        len(nxt) >= 3
                        and re.search(r"[A-Za-z]{2,}", nxt)
                        and not re.search(r"[a-z]{4,}", nxt)
                        and not any(noise in nxt_up for noise in
                                    ["DATE", "RESULT", "MARKS", "TOTAL", "REGISTER", "MONTH",
                                     "MOTHER", "FATHER", "NAME"])
                    )
                    if is_clean_name:
                        cand = nxt
            
            if cand:

                cand = re.sub(r"^[\s:.\-,]+", "", cand).strip()

                cand = re.sub(r"^[LIJ](?=[A-Z]{8,})", "", cand).strip()

                for n in NOISE:
                    if n in cand:
                        cand = cand.split(n)[0].strip()
                
                """
                is_priority_hit = bool(label_found) or any(
                    lbl in line.upper() for lbl in ["NAME:", "NAME OF", "NME:"]
                )
                name_is_priority = bool(name) and name.endswith("@@")
                if len(cand) >= 3 and "GOVERNMENT" not in line.upper():
                    if not name:
                        should_set = True
                    elif name_is_priority:
                        should_set = is_priority_hit
                    else:
                        should_set = is_priority_hit or (len(cand) > len(name) and "AND" not in cand)
                    if should_set:
                        name = cand
                        if is_priority_hit:

                             name = f"{cand}@@"
                """
                is_priority_hit = bool(label_found) or any(
                     lbl in line.upper() for lbl in ["NAME:", "NAME OF", "NME:"]
                )
                name_is_priority = bool(name) and name.endswith("@@")
                if len(cand) >= 3 and "GOVERNMENT" not in line.upper():
                    
                    should_set = (not name) or is_priority_hit
                    if should_set:
                        name = cand
                        if is_priority_hit:

                             name = f"{cand}@@"   

        if not name and "CERTIFY" in line and "THAT" in line:
            cert_match = re.search(
                r"CERTIFY(?:ING)?\s+THAT\s+(?!THE\b)([A-Z][A-Z\s]{2,40}?)"
                r"(?=\s+(?:ROLL|MOTHER|FATHER|DATE|SCHOOL|HAS|IS)\b|$)",
                line
            )
            if cert_match:
                cand = cert_match.group(1).strip()
                if len(cand) >= 3:
                    name = cand

        father_kw_end = _find_parent_kw_end(line, r"FATHER|HUSBAND|FATHWER", ("FATHER", "FATHERS"))
        if father_kw_end is not None:
            remainder = line[father_kw_end:]
            remainder = _strip_parent_label(remainder)
            for n in NOISE:
                if n in remainder:
                    remainder = remainder.split(n)[0].strip()

            cand = ""
            if len(remainder) >= 3 and re.search(r"[A-Z]{3,}", remainder):
                cand = remainder
            elif i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                for n in NOISE:
                    if n in nxt.upper():
                        nxt = nxt[:nxt.upper().index(n)].strip()
                if (len(nxt) >= 3 and re.search(r"[A-Za-z]{2,}", nxt)
                        and "MOTHER" not in nxt.upper()
                        and not _is_probable_header_noise(nxt)):
                    cand = nxt

            if not cand:
                for back in range(1, min(4, i + 1)):
                    prev = lines[i - back].strip()
                    prev_up = prev.upper()
                    if any(kw in prev_up for kw in
                           ["GOVERNMENT", "KARNATAKA", "BOARD", "DEPARTMENT",
                            "CERTIFICATE", "EXAMINATION", "REGISTER", "DATE",
                            "MOTHER", "FATHER", "NAME", "CANDIDATE"]):
                        continue
                    if (re.match(r"^[A-Za-z0-9\s\.]{3,40}$", prev)
                            and len(prev.split()) <= 6 and len(prev) >= 3):
                        trimmed = _trim_fused_parent_name(prev)
                        if (re.search(r"[A-Za-z]{3,}", trimmed)
                                and not _is_probable_header_noise(trimmed)):
                            cand = trimmed
                            break

            if cand:
                father = _trim_fused_parent_name(cand)

        mother_kw_end = _find_parent_kw_end(line, r"MOTHER", ("MOTHER", "MOTHERS"))
        if mother_kw_end is not None:
            remainder = line[mother_kw_end:]
            remainder = _strip_parent_label(remainder)
            for n in NOISE:
                if n in remainder:
                    remainder = remainder.split(n)[0].strip()

            cand = ""
            if len(remainder) >= 3 and re.search(r"[A-Z]{3,}", remainder):
                cand = remainder
            elif i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                for n in NOISE:
                    if n in nxt.upper():
                        nxt = nxt[:nxt.upper().index(n)].strip()
                if (len(nxt) >= 3 and re.search(r"[A-Za-z]{2,}", nxt)
                        and "FATHER" not in nxt.upper()
                        and not _is_probable_header_noise(nxt)):
                    cand = nxt

            if not cand:
                for back in range(1, min(4, i + 1)):
                    prev = lines[i - back].strip()
                    prev_up = prev.upper()
                    if any(kw in prev_up for kw in
                           ["GOVERNMENT", "KARNATAKA", "BOARD", "DEPARTMENT",
                            "CERTIFICATE", "EXAMINATION", "REGISTER", "DATE",
                            "MOTHER", "FATHER", "NAME", "CANDIDATE"]):
                        continue
                    if (re.match(r"^[A-Za-z0-9\s\.]{3,40}$", prev)
                            and len(prev.split()) <= 6 and len(prev) >= 3):
                        trimmed = _trim_fused_parent_name(prev)
                        if (re.search(r"[A-Za-z]{3,}", trimmed)
                                and not _is_probable_header_noise(trimmed)):
                            cand = trimmed
                            break

            if cand:
                mother = _trim_fused_parent_name(cand)

    if father is None or mother is None:
        def _strip_leading_honorific(text):
            text = text.strip()
            if not text:
                return text
            if " " in text[:6]:
                parts = text.split(None, 1)
                if len(parts[0]) <= 4:
                    return parts[1] if len(parts) > 1 else ""
                return text
            for n in (3, 4):
                if len(text) > n + 3 and text[n].isalpha():
                    return text[n:]
            return text

        for idx, raw in enumerate(lines):
            up = raw.strip().upper()
            if re.search(r"\b(SON OF|DAUGHTER OF)\b", up) and idx + 1 < len(lines):
                if mother is None:
                    cand = _strip_leading_honorific(lines[idx + 1].strip().upper())
                    if len(cand) >= 3 and re.search(r"[A-Z]{2,}", cand):
                        mother = cand
                if father is None and idx + 2 < len(lines):
                    cand = _strip_leading_honorific(lines[idx + 2].strip().upper())
                    if len(cand) >= 3 and re.search(r"[A-Z]{2,}", cand):
                        father = cand
                break

        for raw in lines:
            up = raw.strip().upper()
            if father is None and re.match(r"^SHRI\b", up):
                cand = re.sub(r"^SHRI\b[:\-\s]*", "", up).strip()
                if len(cand) >= 3 and re.search(r"[A-Z]{2,}", cand):
                    father = cand
            if mother is None and re.match(r"^SMT\b", up):
                cand = re.sub(r"^SMT\b[:\-\s]*", "", up).strip()
                if len(cand) >= 3 and re.search(r"[A-Z]{2,}", cand):
                    mother = cand

    #print(f"DEBUG extract_personal_info result: name={name!r} father={father!r} mother={mother!r}")
    return name, father, mother

def _clean_val(val):
    if not val: return val
    if not isinstance(val, str): return val
    return re.sub(r"\s+", " ", val).strip()

def _expand_university_name(name):
    if not name: return name
    name = _clean_val(name).upper()
    
    mapping = {
        "VTU": "VISVESVARAYA TECHNOLOGICAL UNIVERSITY",
        "CBSE": "CENTRAL BOARD OF SECONDARY EDUCATION",
        "ICSE": "INDIAN CERTIFICATE OF SECONDARY EDUCATION",
        "SSLC": "KARNATAKA STATE SECONDARY EDUCATION EXAMINATION BOARD",
        "PUC": "DEPARTMENT OF PRE-UNIVERSITY EDUCATION, KARNATAKA",
        "KARNATAKA STATE BOARD": "KARNATAKA STATE SECONDARY EDUCATION EXAMINATION BOARD",
        "KARNATAKA PU BOARD": "DEPARTMENT OF PRE-UNIVERSITY EDUCATION, KARNATAKA",
        "KARNATAKA PU BOARD (NEW)": "KARNATAKA SCHOOL EXAMINATION AND ASSESSMENT BOARD, BENGALURU",
    }
    
    return mapping.get(name, name)

def extract_college(lines, skip_name=None):

    full_lines_upper = [l.upper() for l in lines]
    COLLEGE_LABEL_BLOCKLIST = [
        "GOVERNMENT OF", "KARNATAKA SCHOOL", "CERTIFY", "CANDIDATE",
        "OBTALN", "OBTAÍN", "OBTA1N", "CLASS OBTAIN", "TWO HUNDRED",
        "THREE HUNDRED", "FOUR HUNDRED", "ONLY", "TOTAL", "MARKS",
    ]

    for idx, lu in enumerate(full_lines_upper):
        if "COLLEGE DETAILS" in lu or "COLLEGE CODE" in lu:
            for check_idx in [idx - 1, idx + 1]:
                if 0 <= check_idx < len(lines):
                    cand = lines[check_idx].strip()
                    cand_up = cand.upper()

                    if (len(cand) > 5
                            and not re.match(r'^[\d\s,/\.]+$', cand)
                            and not any(bk in cand_up for bk in COLLEGE_LABEL_BLOCKLIST)
                            and re.search(r"[A-Za-z]{4,}", cand)):
                        return _clean_college_name(cand)

    header_area = lines[:15]
    
    COLLEGE_KEYWORDS = ["COLLEGE", "INSTITUTE", "TECHNOLOGY", "ACADEMY", "POLYTECHNIC", "SCHOOL", "VIDYALAYA", "UNIVERSITY", "ENGINEERING", "CAMPUS"]
    UNIVERSITY_IDENTIFIERS = ["VISVESVARAYA", "TECHNOLOGICAL", "BELAGAVI", "STATE BOARD", "PRE-UNIVERSITY", "EXAMINATION AND ASSESSMENT"]
    SKIP_KEYWORDS = [
        "REGISTER", "MARKS", "STATEMENT", "GRADE", "USN", "ROLL", "DATE", "RESULT", "PAGE",
        "B.E.", "B.TECH", "M.TECH", "BACHELOR", "MASTER", "DEGREE", "BRANCH", "COURSE", "SCHEME",
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
        "CERTIFICATE OF", "EXAMINATION", "CERTIFY", "BOARD OF SECONDARY", "CENTRAL BOARD",
        "SEAT NUMBER", "SEAT NO", "STUDENT NAME", "PROVISIONAL RESULT", "SEMESTER",
        "CLASS OBTAIN", "CLASS OBTALN", "OBTALN", "OBTAÍN", "OBTA1N",
    ]
    
    CITIES = ["MANGALURU", "MANGALORE", "BELAGAVI", "BELGAUM", "BENGALURU", "BANGALORE", "MYSORE", "MYSURU", "HUBLI", "DHARWAD", "KALABURAGI", "GULBARGA", "SHIVAMOGGA"]

    
    BOARD_ACRONYMS_SAFE_IN_NAME = {"ICSE", "CBSE", "SSLC", "PUC", "VTU"}

    for i, line in enumerate(header_area):
        line_upper = line.upper()

        if (skip_name and skip_name.upper() in line_upper
                and skip_name.upper() not in BOARD_ACRONYMS_SAFE_IN_NAME):
            continue
            
        if any(u in line_upper for u in UNIVERSITY_IDENTIFIERS):
             if not any(k in line_upper for k in ["COLLEGE", "INSTITUTE", "ACADEMY", "ENGINEERING"]):
                 continue

        if any(s in line_upper for s in ["B.E.", "B.TECH", "BACHELOR", "BRANCH", "COURSE", "SCHEME"]):
            continue

        if any(kw in line_upper for kw in COLLEGE_KEYWORDS):
            if any(s in line_upper for s in SKIP_KEYWORDS):
                continue
                
            if len(line.strip()) > 8:
                name = line
                if i + 1 < len(header_area):
                    next_line = header_area[i+1].strip()
                    next_upper = next_line.upper()
                    if any(c in next_upper for c in CITIES) or re.search(r"\b\d{6}\b", next_line):
                        name = name.strip().rstrip(",")
                        next_line = next_line.strip().lstrip(",")
                        name += f", {next_line}"
                
                return _clean_college_name(name)

def _clean_college_name(name):
    if not name: return name
    name = re.sub(r"^(?:NAME\sO[FT]\sTHE\s)?(?:COLLEGE|INSTITUTE|OFFICE|DEPARTMENT|SCHOOL)[:\-\s]*", "", name, flags=re.IGNORECASE).strip()
    
    code_match = re.search(r"\d{3,6}-\s*", name)
    if code_match and code_match.start() <= 15:
        name = name[code_match.end():]
    
    name = re.sub(r"([A-Za-z])(\d+)([A-Za-z])", r"\1 \2 \3", name) 
    name = re.sub(r"^OF\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^(?=[A-Z0-9#\-\.]*\d)[A-Z0-9#\-\.]{4,}(?::?\s*[A-Z0-9#\-\.]+)?\s*,\s*", "", name)
    name = re.sub(r"^.*?DRDADDRND.*?(?:,\s*|\s+)", "", name, flags=re.IGNORECASE) 
    name = _repair_spacing(name)
    return _clean_val(name)
                
    return None

def _split_fused_trailing_initials(name):
    if not name or " " in name:
        return name
    up = name.upper()
    if not up.isalpha():
        return name
    if len(up) >= 7:
        stem = up[:-2]
        if len(stem) >= 5:
            return f"{stem} {up[-2]} {up[-1]}"
    if len(up) >= 6:
        stem = up[:-1]
        endings = ("AY", "AR", "AS", "AN", "AM", "AL", "AD", "YA", "TH", "SH")
        if len(stem) >= 5 and not any(up.endswith(s) for s in endings):
            return f"{stem} {up[-1]}"
    return up
"""
KNOWN_PARENT_NAME_FIXES = {
    "ARJUNNS": "ARJUN N S",
    "MEGHANACD": "MEGHANA C D",
    "YASHASWINIM": "YASHASWINI M",
    "MAHADEVCHIKRAY": "MAHADEV CHIKRAY",
    "SUJITHMR": "SUJITH M R",
    "SRINIVASRC": "SRINIVAS R C",
    "RAMESHKUMARMR": "RAMESH KUMAR M R",
    "SEETHALAKSHMIAS": "SEETHALAKSHMI A S",
}
"""
def _format_parent_name(name):
    if not name:
        return name
    name = name.replace("@@", "")
    name = _clean_val(name)
    return name

def _format_name(name):
    if not name: return name
    name = name.replace("@@", "")
    name = _clean_val(name)  
    name = _clean_val(name)

    return name

def extract(lines, image, raw_ocr_data=None):
    try:
        height = image.shape[0] if image is not None else 0
        width = image.shape[1] if image is not None else 0
        if raw_ocr_data:

            lines = _group_texts_by_y(raw_ocr_data, image_height=height, image_width=width)
        
        board = detect_board(lines)

        subjects = extract_subjects_from_text(lines, raw_data=raw_ocr_data, image_width=width, board_type=board)

        has_total_label = any(
            ("TOTAL" in l.upper() or "GRAND" in l.upper())
            and not ("SUBJECT" in l.upper() and "CODE" in l.upper())
            and re.search(r"\b\d{3,}\b", l)
            for l in lines
        )

        if board in ("VTU", "CBSE", "ICSE") and not has_total_label and subjects:
            total = sum(s["marks"] for s in subjects if s.get("marks") is not None) or None
        else:
            total = extract_total(lines)

        is_grade_card = any(k in " ".join(lines).upper() for k in ["GRADE CARD", "SGPA", "CGPA", "CREDIT"])
        if total and subjects and not is_grade_card:
            subjects = _reconcile_marks(subjects, total)

        board = detect_board(lines)

        reg_val, reg_type, semester = extract_register_info(lines)
        name, father, mother = extract_personal_info(lines)
        sgpa, cgpa = extract_gpa(lines)
        
        if any(
                s["subject"].upper().startswith(("FIRST LANGUAGE", "SECOND LANGUAGE", "THIRD LANGUAGE"))
                for s in subjects
            ):
                board = "KARNATAKA STATE BOARD"

        
        college = None if board == "VTU" else extract_college(lines, skip_name=board)
        if board == "VTU":
            father = None
            mother = None

        data = {
            "student_name": _format_name(name),
            "father_name": _format_parent_name(father),
            "mother_name": _format_parent_name(mother),
            "college_name": college,
            "subjects": subjects,
            "total_marks": total,
            "sgpa": sgpa,
            "cgpa": cgpa,
            "university": _expand_university_name(board) if board and "UNIVERSITY" in board.upper() or board == "VTU" else None
        }

        final_board = _clean_val(board)
        is_university = False
        
        if reg_type == "USN" or (final_board and ("UNIVERSITY" in final_board or "VTU" in final_board)):
            data["university"] = "VTU"
            is_university = True
            if final_board and "VTU" not in final_board and "UNIVERSITY" not in final_board:
                 data["board"] = final_board
        else:
             if final_board:

                 if "BOARD" in final_board or final_board in ["SSLC", "PUC"]:
                      data["board"] = final_board
                      data["university"] = None
                 elif "UNIVERSITY" in final_board:
                      data["university"] = final_board
                      is_university = True
                 else:
                      data["board"] = final_board

        if reg_val:

            reg_key = "usn" if is_university else "register number"
            data[reg_key] = reg_val
        if semester:
            data["semester"] = semester

        if "university" in data:
            data["university"] = _expand_university_name(data["university"])
        if "board" in data:
            data["board"] = _expand_university_name(data["board"])

        for k, v in data.items():
            if isinstance(v, str):
                data[k] = _clean_val(v)
            if k == "subjects" and isinstance(v, list):
                for s in v:
                    s["subject"] = _clean_val(s["subject"])

                    if "meta" in s:
                        del s["meta"]
                    if "certainty" in s:
                        del s["certainty"]
                

        if "subjects" in data and board not in ("VTU", "KARNATAKA PU BOARD", "KARNATAKA PU BOARD (NEW)", "ICSE"):
            data["subjects"] = [
                s for s in data["subjects"]
                if _is_recognized_subject(s["subject"]) or s.get("_from_additional")
            ]
            data["subjects"] = [s for s in data["subjects"] if s["subject"].upper() != "GEM"]

        if "subjects" in data and isinstance(data["subjects"], list):
            for s in data["subjects"]:
                if "_from_additional" in s:
                    del s["_from_additional"]

        return {k: v for k, v in data.items() if v is not None}

    except Exception as e:
        #print("Marksheet Extraction Error:", e)

        return {
            "board": None,
            "subjects": [],
            "result": None,
            "total_marks": None
        }
