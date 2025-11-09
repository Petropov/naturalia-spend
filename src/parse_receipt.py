from __future__ import annotations
import io, re, math, subprocess, tempfile, shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from pdfminer.high_level import extract_text
AMOUNT_RE = re.compile(r'(?<!\d)(?:€\s*)?-?\s*\d{1,3, 'product_norm': normalize_name(item.get('product_raw',''))}(?:[ .]\d{3, 'product_norm': normalize_name(item.get('product_raw',''))})*(?:[.,]\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})?\s*€?(?!\d)')
TOTAL_HINTS_POS = ("total ttc","total à payer","total a payer","total a régler","total a regler","montant total","total","paiement","a payer")
TOTAL_HINTS_NEG = ("sous-total","sous total","subtotal","total articles","total ligne","total remises")
DATE_RES = [re.compile(r'(\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})/(\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})/(\d{4, 'product_norm': normalize_name(item.get('product_raw',''))})'), re.compile(r'(\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})/(\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})/(\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})'), re.compile(r'(\d{4, 'product_norm': normalize_name(item.get('product_raw',''))})-(\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})-(\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})')]
def _normalize_amount(s: str) -> float | None:
    import math, re
    s0 = s.strip().replace('€', '').replace('\u00a0', ' ').strip()
    # e.g. "14 61" -> "14.61"
    if re.match(r'^\d+(?:\s|\u00A0){1,2, 'product_norm': normalize_name(item.get('product_raw',''))}\d{2, 'product_norm': normalize_name(item.get('product_raw',''))}$', s0):
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
        # fallback: plain 3–4 digits like '1101' -> 11.01 if plausible
        if re.fullmatch(r'\d{3,4, 'product_norm': normalize_name(item.get('product_raw',''))}', s0):
            try:
                v = float(int(s0)) / 100.0
                if 0 < v <= 300:
                    return round(v, 2)
            except:
                return None
        return None
    return round(v, 2) if math.isfinite(v) else None
def _text_via_pdfminer(b:bytes)->str:
    try: return extract_text(io.BytesIO(b)) or ""
    except: return ""
def _text_via_pdftotext(b:bytes)->str:
    if not shutil.which("pdftotext"): return ""
    with tempfile.TemporaryDirectory() as td:
        p=Path(td,"in.pdf"); p.write_bytes(b)
        out=Path(td,"out.txt")
        subprocess.run(["pdftotext","-layout",str(p),str(out)],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return out.read_text(encoding="utf-8",errors="ignore") if out.exists() else ""
def _text_via_ocr(b:bytes)->str:
    import subprocess, tempfile, shutil
    from pathlib import Path
    td = tempfile.mkdtemp(prefix="ocrpdf_")
    try:
        pdf = Path(td,"in.pdf"); pdf.write_bytes(b)
        subprocess.run(["pdftoppm","-r","400","-png",str(pdf), Path(td,"p").as_posix()],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lines_all = []
        for img in sorted(Path(td).glob("p-*.png")):
            out = subprocess.run(
                ["tesseract", str(img), "stdout", "-l", "fra+eng", "--oem","1","--psm","6","tsv"],
                text=True, capture_output=True, check=False
            ).stdout.splitlines()
            words = [ln.split("	") for ln in out[1:] if ln.strip()]
            rows = {, 'product_norm': normalize_name(item.get('product_raw',''))}
            for cols in words:
                if len(cols) >= 12 and cols[0] == '5':  # word
                    try:
                        lid = int(cols[4])
                    except:
                        continue
                    rows.setdefault(lid, []).append(cols[11])
            lines = [" ".join(v).strip() for _,v in sorted(rows.items()) if v]
            lines_all.extend([ln for ln in lines if ln])
        return "\n".join(lines_all)
    finally:
        shutil.rmtree(td, ignore_errors=True)
def _extract_date(t:str)->str|None:
    t=t.replace("\u00a0"," ")
    for rx in DATE_RES:
        m=rx.search(t)
        if not m: continue
        try:
            a,b,c=m.groups()
            if len(c)==4:
                d=datetime(int(c),int(b),int(a))
            elif len(a)==4:
                d=datetime(int(a),int(b),int(c))
            else:
                year=2000+int(c) if int(c)<80 else 1900+int(c)
                d=datetime(year,int(b),int(a))
            return d.date().isoformat()
        except: pass
    return None
def _extract_total(t:str)->float|None:
    lines=[ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in lines:
        low=ln.lower()
        if any(h in low for h in TOTAL_HINTS_NEG): continue
        if any(h in low for h in TOTAL_HINTS_POS):
            cand=None
            for m in AMOUNT_RE.finditer(ln): cand=m.group(0)
            if cand:
                v=_normalize_amount(cand)
                if v is not None: return v
    cands=[]
    for m in AMOUNT_RE.finditer(t):
        raw=m.group(0)
        v=_normalize_amount(raw)
        if v is not None: cands.append(v)
    if not cands: return None
    reasonable=[v for v in cands if 0<v<=300]
    if reasonable: return reasonable[-1]
    return cands[-1]
def _extract_store(t:str)->str|None:
    lines=[ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in lines[:8]:
        if "naturalia" in ln.lower():
            return ln.strip()
    return None
def _extract_postcode(t:str)->str|None:
    m=re.search(r'\b\d{5, 'product_norm': normalize_name(item.get('product_raw',''))}\b', t)
    return m.group(0) if m else None
def _extract_time(t:str)->str|None:
    m=re.search(r'\b(\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})(?:[:h])(\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})(?::(\d{2, 'product_norm': normalize_name(item.get('product_raw',''))}))?\b', t)
    if not m: return None
    hh,mm,ss=m.group(1),m.group(2),m.group(3) or '00'
    return f"{hh, 'product_norm': normalize_name(item.get('product_raw',''))}:{mm, 'product_norm': normalize_name(item.get('product_raw',''))}:{ss, 'product_norm': normalize_name(item.get('product_raw',''))}"
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
    lines = [re.sub(r'[·•]|\u2022','', ln) for ln in t.splitlines()]
    clean = [re.sub(r'\.{2,, 'product_norm': normalize_name(item.get('product_raw',''))}', ' ', ln).strip() for ln in lines if ln.strip()]
    items = []
    AMT_ONLY = re.compile(r'^-?\d{1,3, 'product_norm': normalize_name(item.get('product_raw',''))}(?:[ .]\d{3, 'product_norm': normalize_name(item.get('product_raw',''))})?[\s.,]\d{2, 'product_norm': normalize_name(item.get('product_raw',''))}\s*€?$', re.U)
    AMT_TAIL = re.compile(r'(-?\d{1,3, 'product_norm': normalize_name(item.get('product_raw',''))}(?:[ .]\d{3, 'product_norm': normalize_name(item.get('product_raw',''))})*|-?\d+)[\s.,]\d{2, 'product_norm': normalize_name(item.get('product_raw',''))}\s*€?$', re.U)
    MULT = re.compile(r'^(?:(?P<name>.+?)\s+)?(?P<qty>\d+)\s*[xX×]\s*(?P<unit>\d+[\s.,]\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})\s*(?:=|→)?\s*(?P<total>\d+[\s.,]\d{2, 'product_norm': normalize_name(item.get('product_raw',''))})\s*€?$', re.U)
    BAD_WORDS = (
        'total','sous total','ttc','paiement','règlement','reglement','tva','carte','visa','mastercard',
        'caiss','client','bonjour','merci','rendu','espèces','especes','reste a payer','a payer',
        'nombre articles','========','====','---','__','cb','amex'
    )
    def is_header(name:str)->bool:
        low = name.lower()
        if any(k in low for k in BAD_WORDS): return True
        if len(name) <= 2: return True
        if all(ch in "=*_- .:/\\" for ch in low): return True
        return False
    def to_num(s):
        s = s.replace('€','').replace('\u00a0',' ').strip()
        if ' ' in s and s.split()[-1].isdigit() and len(s.split()[-1])==2:
            parts = s.split(); s = parts[0]+'.'+parts[1]
        s = s.replace(' ','')
        if ',' in s and '.' in s: s = s.replace('.','').replace(',', '.')
        elif ',' in s: s = s.replace(',', '.')
        try: return round(float(s),2)
        except: return None
    # Pass 1: lines like "NAME .... 3,49" or "2 x 1,99 = 3,98"
    for raw in clean:
        m = MULT.match(raw)
        if m:
            nm = (m.group('name') or '').strip()
            qty = int(m.group('qty')) if m.group('qty') else 1
            unit = to_num(m.group('unit'))
            tot  = to_num(m.group('total'))
            if nm and tot is not None and 0 < tot <= 300 and not is_header(nm):
                items.append({'product_raw': nm, 'qty': qty, 'unit_price': unit, 'line_total': tot, 'product_norm': normalize_name(item.get('product_raw',''))})
            continue
        mt = AMT_TAIL.search(raw)
        if mt:
            nm = raw[:mt.start()].strip()
            tot = to_num(mt.group(0))
            if nm and tot is not None and 0 < tot <= 300 and not is_header(nm):
                items.append({'product_raw': nm, 'qty': 1, 'unit_price': None, 'line_total': tot, 'product_norm': normalize_name(item.get('product_raw',''))})
    # Pass 2: amounts on their own line → pair with nearest previous non-header, non-amount line
    last_good = None
    for raw in clean:
        if AMT_ONLY.match(raw):
            tot = to_num(raw)
            if tot is None or not (0 < tot <= 300): 
                continue
            if last_good and not is_header(last_good) and not AMT_ONLY.match(last_good):
                items.append({'product_raw': last_good.strip(), 'qty': 1, 'unit_price': None, 'line_total': tot, 'product_norm': normalize_name(item.get('product_raw',''))})
        else:
            last_good = raw
    # Deduplicate exact (name,total) pairs
    seen=set(); uniq=[]
    for it in items:
        key = (it['product_raw'].lower().strip(), it['line_total'])
        if key in seen: 
            continue
        seen.add(key); uniq.append(it)
    items = uniq
    # Fill unit_price if missing
    for it in items:
        if it.get('unit_price') is None and it.get('qty'):
            it['unit_price'] = round(it['line_total']/it['qty'], 2)
    for i,it in enumerate(items,1): it['line_no']=i
    return items

def normalize_name(s: str):
    return ' '.join(s.split()).lower()

def parse_attachment(b:bytes,fname:str)->Dict[str,Any]:
    txt=_text_via_pdfminer(b)
    if not txt.strip(): txt=_text_via_pdftotext(b)
    if not txt.strip(): txt=_text_via_ocr(b)
    total=_extract_total(txt) if txt else None
    date=_extract_date(txt) if txt else None
    conf=0.9 if (txt and total is not None) else (0.5 if txt else 0.2)
    return {
        "date": date,
        "total": total or 0.0,
        "items": _extract_items(txt) if txt else [],
        "confidence": conf,
        "store": _extract_store(txt) if txt else None,
        "postcode": _extract_postcode(txt) if txt else None,
        "time": _extract_time(txt) if txt else None,
        "text_sample": "\n".join((txt or "").splitlines()[:20])
    , 'product_norm': normalize_name(item.get('product_raw',''))}
