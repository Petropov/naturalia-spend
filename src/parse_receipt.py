from __future__ import annotations
import io, re, math, subprocess, tempfile, shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from pdfminer.high_level import extract_text

# --- heuristics / constants
AMOUNT_RE = re.compile(r'(?<!\d)(?:€\s*)?-?\s*\d{1,3}(?:[ .]\d{3})*(?:[.,]\d{2})?\s*€?(?!\d)')
TOTAL_HINTS_POS = (
    "total ttc","total à payer","total a payer","total a régler","total a regler",
    "montant total","reste a payer","reste à payer","total","paiement","a payer","à payer"
)
TOTAL_HINTS_NEG = ("sous-total","sous total","subtotal","total articles","total ligne","total remises")

DATE_RES = [
    re.compile(r'(\d{2})/(\d{2})/(\d{4})'),      # dd/mm/yyyy
    re.compile(r'(\d{2})/(\d{2})/(\d{2})'),      # dd/mm/yy
    re.compile(r'(\d{4})-(\d{2})-(\d{2})'),      # yyyy-mm-dd
]

# --- amount parsing robust to French formats and OCR artifacts
def _normalize_amount(s: str) -> float | None:
    import math, re
    s0 = s.strip().replace('€', '').replace('\u00a0', ' ').strip()
    # e.g. "14 61" or "14 61" -> "14.61"
    if re.match(r'^\d+(?:\s|\u00A0){1,2}\d{2}$', s0):
        s0 = s0.replace('\u00A0', ' ')
        parts = s0.split()
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 2:
            s0 = parts[0] + '.' + parts[1]
        else:
            s0 = s0.replace(' ', '')
    else:
        s0 = s0.replace(' ', '')
        if ',' in s0 and '.' in s0:
            s0 = s0.replace('.', '').replace(',', '.')
        elif ',' in s0 and '.' not in s0:
            s0 = s0.replace(',', '.')
    try:
        v = float(s0)
    except:
        # fallback: plain 3–4 digits like '1101' -> 11.01 when plausible
        if re.fullmatch(r'\d{3,4}', s0):
            try:
                v = float(int(s0)) / 100.0
                if 0 < v <= 300:
                    return round(v, 2)
            except:
                return None
        return None
    return round(v, 2) if math.isfinite(v) else None

# --- text sources
def _text_via_pdfminer(b:bytes)->str:
    try:
        return extract_text(io.BytesIO(b)) or ""
    except:
        return ""

def _text_via_pdftotext(b:bytes)->str:
    if not shutil.which("pdftotext"):
        return ""
    with tempfile.TemporaryDirectory() as td:
        p=Path(td,"in.pdf"); p.write_bytes(b)
        out=Path(td,"out.txt")
        subprocess.run(["pdftotext","-layout",str(p),str(out)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out.read_text(encoding="utf-8", errors="ignore") if out.exists() else ""

def _text_via_ocr(b:bytes)->str:
    # TSV line-level OCR at 400 DPI, fra+eng
    td = tempfile.mkdtemp(prefix="ocrpdf_")
    try:
        pdf = Path(td,"in.pdf"); pdf.write_bytes(b)
        subprocess.run(["pdftoppm","-r","400","-png",str(pdf), Path(td,"p").as_posix()],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lines_all: list[str] = []
        for img in sorted(Path(td).glob("p-*.png")):
            out = subprocess.run(
                ["tesseract", str(img), "stdout", "-l", "fra+eng", "--oem","1","--psm","6","tsv"],
                text=True, capture_output=True, check=False
            ).stdout.splitlines()
            # tsv header then rows: we care about word-level (level==5)
            rows = [ln.split("\t") for ln in out[1:] if ln.strip()]
            by_line: dict[int, list[str]] = {}
            for cols in rows:
                if len(cols) < 12: continue
                if cols[0] != '5':  # word level
                    continue
                try:
                    lid = int(cols[4])
                except:
                    continue
                text = cols[11]
                by_line.setdefault(lid, []).append(text)
            ordered = [" ".join(words).strip() for _,words in sorted(by_line.items()) if words]
            lines_all.extend([ln for ln in ordered if ln])
        return "\n".join(lines_all)
    finally:
        shutil.rmtree(td, ignore_errors=True)

# --- field extraction
def _extract_date(text: str) -> str | None:
    t = text.replace("\u00a0", " ")
    for rx in DATE_RES:
        m = rx.search(t)
        if not m:
            continue
        try:
            a,b,c = m.groups()
            if len(c)==4 and len(a)==2:             # dd/mm/yyyy
                d = datetime(int(c), int(b), int(a))
            elif len(a)==4:                          # yyyy-mm-dd
                d = datetime(int(a), int(b), int(c))
            else:                                    # dd/mm/yy
                year = 2000+int(c) if int(c) < 80 else 1900+int(c)
                d = datetime(year, int(b), int(a))
            return d.date().isoformat()
        except:
            pass
    return None

def _extract_total(text: str) -> float | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # prefer hint lines
    for ln in lines:
        low = ln.lower()
        if any(h in low for h in TOTAL_HINTS_NEG):
            continue
        if any(h in low for h in TOTAL_HINTS_POS):
            cand = None
            for m in AMOUNT_RE.finditer(ln):
                cand = m.group(0)
            if cand:
                v = _normalize_amount(cand)
                if v is not None:
                    return v
    # fallback: last reasonable amount (≤ 300)
    cands = []
    for m in AMOUNT_RE.finditer(text):
        v = _normalize_amount(m.group(0))
        if v is not None:
            cands.append(v)
    if not cands:
        return None
    reasonable = [v for v in cands if 0 < v <= 300]
    return (reasonable[-1] if reasonable else cands[-1])

def _extract_store(t:str)->str|None:
    lines=[ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in lines[:8]:
        if "naturalia" in ln.lower():
            return ln.strip()
    return None

def _extract_postcode(t:str)->str|None:
    m=re.search(r'\b\d{5}\b', t)
    return m.group(0) if m else None

def _extract_time(t:str)->str|None:
    m=re.search(r'\b(\d{2})(?:[:h])(\d{2})(?::(\d{2}))?\b', t)
    if not m: return None
    hh,mm,ss=m.group(1),m.group(2),m.group(3) or '00'
    return f"{hh}:{mm}:{ss}"

def _num(s:str):
    s=s.replace("€","").replace("\u00a0"," ").strip()
    if " " in s and s.split()[-1].isdigit() and len(s.split()[-1])==2:
        parts=s.split(); s=parts[0]+"."+parts[1]
    s=s.replace(" ","")
    if "," in s and "." in s: s=s.replace(".","").replace(",",".")
    elif "," in s: s=s.replace(",",".")
    try: return round(float(s),2)
    except: return None



def _extract_items(t:str):
    import re as _re
    # normalize lines
    lines = [_re.sub(r'[·•]|\u2022','', ln) for ln in t.splitlines()]
    clean = [_re.sub(r'\.{2,}', ' ', ln).strip() for ln in lines if ln.strip()]
    items = []

    # patterns
    AMT_ONLY = _re.compile(r'^-?\d{1,3}(?:[ .]\d{3})*(?:[., ]\d{2})\s*€?$', _re.U)
    AMT_TAIL = _re.compile(r'(?P<amt>-?\d{1,3}(?:[ .]\d{3})*(?:[., ]\d{2})|-?\d+(?:[., ]\d{2}))\s*€?$', _re.U)
    MULT     = _re.compile(r'^(?:(?P<name>.+?)\s+)?(?P<qty>\d+)\s*[xX×]\s*(?P<unit>\d+[\s., ]\d{2})\s*(?:=|→)?\s*(?P<total>\d+[\s., ]\d{2})\s*€?$', _re.U)

    BAD_WORDS = (
        'total','sous total','ttc','paiement','règlement','reglement','tva','carte','visa','mastercard',
        'caiss','client','bonjour','merci','rendu','espèces','especes','reste a payer','reste à payer',
        'nombre articles','========','====','---','__','cb','amex','total hors avantages','mes remises',
        'total des remises','vente à emporter','vente a emporter','total tva','cb emv','reste  a  payer'
    )
    TAX_KEYS = ('h.t.','ht','t.v.a.','tva','t.t.c','ttc','total ttc','total ht')

    def is_header(name:str)->bool:
        low = name.lower().strip()
        if '%' in low:                   # any percent line -> tax line
            return True
        low_nop = low.replace('.', '').replace(':','')
        if low in BAD_WORDS or low_nop in TAX_KEYS: return True
        if any(k in low for k in BAD_WORDS): return True
        if len(name) <= 1: return True
        if all(ch in "=*_- .:/\\" for ch in low): return True
        return False

    def to_num(s):
        s = s.replace('€','').replace('\\u00a0',' ').strip()
        m = _re.search(r'(-?\\d{1,3}(?:[ .]\\d{3})*(?:[., ]\\d{2})|-?\\d+(?:[., ]\\d{2}))\\s*$', s)
        if not m:
            return None
        s = m.group(1)
        s = s.replace(' ', '')
        if ',' in s and '.' in s: s = s.replace('.', '').replace(',', '.')
        elif ',' in s: s = s.replace(',', '.')
        try:
            return round(float(s), 2)
        except:
            if _re.fullmatch(r'\\d{3,4}', s):   # 1101 -> 11.01
                return round(float(int(s))/100.0, 2)
            return None

    # Pass 1: multiplier lines and single-line 'name ... amount€'
    for raw in clean:
        m = MULT.match(raw)
        if m:
            nm = (m.group('name') or '').strip()
            qty = int(m.group('qty')) if m.group('qty') else 1
            unit = to_num(m.group('unit'))
            tot  = to_num(m.group('total'))
            if nm and tot is not None and 0 < tot <= 300 and not is_header(nm):
                items.append({'product_raw': nm, 'qty': qty, 'unit_price': unit, 'line_total': tot})
            continue
        mt = AMT_TAIL.search(raw)
        if mt:
            nm = raw[:mt.start()].strip()
            if nm and not is_header(nm):
                tot = to_num(mt.group('amt'))
                if tot is not None and 0 < tot <= 300:
                    items.append({'product_raw': nm, 'qty': 1, 'unit_price': None, 'line_total': tot})

    # Pass 2: amount-only lines -> pair with previous non-header line
    last_good = None
    for raw in clean:
        if AMT_ONLY.match(raw):
            tot = to_num(raw)
            if tot is None or not (0 < tot <= 300):
                continue
            if last_good and not is_header(last_good) and not AMT_ONLY.match(last_good):
                nm = last_good.strip()
                if not is_header(nm):
                    items.append({'product_raw': nm, 'qty': 1, 'unit_price': None, 'line_total': tot})
        else:
            last_good = raw

    # Dedup (name,total)
    seen=set(); uniq=[]
    for it in items:
        key=(it['product_raw'].lower().strip(), it['line_total'])
        if key in seen: continue
        seen.add(key); uniq.append(it)
    items = uniq

    # Backfill unit price
    for it in items:
        if it.get('unit_price') is None and it.get('qty'):
            it['unit_price'] = round(it['line_total']/it['qty'], 2)

    for i,it in enumerate(items,1): it['line_no']=i
    return items

def parse_attachment(b:bytes,fname:str)->Dict[str,Any]:
    # source A: text layer
    txt_a=_text_via_pdfminer(b)
    if not txt_a.strip(): txt_a=_text_via_pdftotext(b)
    # source B: OCR (line TSV)
    txt_b=_text_via_ocr(b)

    # extract from A
    items_a = _extract_items(txt_a) if txt_a else []
    total_a = _extract_total(txt_a) if txt_a else None
    date_a  = _extract_date(txt_a)  if txt_a else None

    # extract from B
    items_b = _extract_items(txt_b) if txt_b else []
    total_b = _extract_total(txt_b) if txt_b else None
    date_b  = _extract_date(txt_b)  if txt_b else None

    # pick the richer source (more items); tie-break by having a valid total
    use_b = False
    if len(items_b) > len(items_a):
        use_b = True
    elif len(items_b) == len(items_a):
        if (total_b is not None) and (total_a is None):
            use_b = True

    if use_b:
        txt, items, total, date = txt_b, items_b, total_b, date_b
    else:
        txt, items, total, date = txt_a, items_a, total_a, date_a

    # confidence heuristic
    conf=0.9 if (txt and (total is not None or items)) else (0.5 if txt else 0.2)

    return {
        "date": date,
        "total": total or 0.0,
        "items": items,
        "confidence": conf,
        "store": _extract_store(txt) if txt else None,
        "postcode": _extract_postcode(txt) if txt else None,
        "time": _extract_time(txt) if txt else None,
        "text_sample": "\n".join((txt or "").splitlines()[:20]),
    }
