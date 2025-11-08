import subprocess, tempfile, shutil, re, csv
from pathlib import Path

RAW = Path("raw")
DATA = Path("data"); DATA.mkdir(exist_ok=True)

BAD_WORDS = (
    "total","sous total","ttc","paiement","règlement","reglement","tva","carte","visa","mastercard",
    "caiss","client","bonjour","merci","rendu","espèces","especes","reste a payer","reste à payer",
    "nombre articles","total hors avantages","mes remises","total des remises",
    "vente à emporter","vente a emporter","total tva","cb emv","cb","amex"
)
def is_header(name:str)->bool:
    low=name.lower().strip()
    if "%" in low: return True
    if any(k in low for k in BAD_WORDS): return True
    if len(low)<=1 or all(ch in "=*_- .:/\\" for ch in low): return True
    return False

AMT_TAIL = re.compile(r'(?P<amt>-?\d{1,3}(?:[ .]\d{3})*(?:[., ]\d{2})|-?\d+(?:[., ]\d{2}))\s*€?$', re.U)
AMT_ONLY = re.compile(r'^-?\d{1,3}(?:[ .]\d{3})*(?:[., ]\d{2})\s*€?$', re.U)
MULT_EQ   = re.compile(r'^(?:(?P<name_a>.+?)\s+)?(?P<qty_a>\d+)\s*[xX×]\s*(?P<unit_a>\d+[ .,\d]*\d{2})\s*(?:=|→)\s*(?P<tot_a>\d+[ .,\d]*\d{2})\s*€?$', re.U)
MULT_NAME = re.compile(r'^(?P<qty_b>\d+)\s*[xX×]\s*(?P<name_b>.+?)\s+(?P<unit_b>\d+[ .,\d]*\d{2})\s*€?$', re.U)

def to_num(s:str):
    s = s.replace("€","").replace("\u00a0"," ").strip()
    m = re.search(r'(-?\d{1,3}(?:[ .]\d{3})*(?:[., ]\d{2})|-?\d+(?:[., ]\d{2}))\s*$', s)
    if not m: return None
    s = m.group(1).replace(" ","")
    if "," in s and "." in s: s=s.replace(".","").replace(",",".")
    elif "," in s: s=s.replace(",",".")
    try: return round(float(s),2)
    except:
        if re.fullmatch(r'\d{3,4}', s): return round(int(s)/100.0,2)
        return None

def clean_name(s:str)->str:
    s = s.strip().rstrip("€").strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def ocr_lines(pdf_bytes:bytes, dpi=400, lang="fra+eng", psm="6"):
    td=tempfile.mkdtemp(prefix="ocrpdf_")
    try:
        pdf=Path(td,"in.pdf"); pdf.write_bytes(pdf_bytes)
        subprocess.run(["pdftoppm","-r",str(dpi),"-png",str(pdf),Path(td,"p").as_posix()],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out_lines=[]
        for img in sorted(Path(td).glob("p-*.png")):
            out = subprocess.run(
                ["tesseract", str(img), "stdout", "-l", lang, "--oem","1","--psm", psm, "tsv"],
                text=True, capture_output=True, check=False
            ).stdout.splitlines()
            words=[ln.split("\t") for ln in out[1:] if ln.strip()]
            rows={}
            for cols in words:
                if len(cols)>=12 and cols[0]=='5':
                    try: lid=int(cols[4])
                    except: continue
                    rows.setdefault(lid,[]).append(cols[11])
            ordered=[" ".join(v).strip() for _,v in sorted(rows.items()) if v]
            out_lines.extend([ln for ln in ordered if ln])
        return out_lines
    finally:
        shutil.rmtree(td, ignore_errors=True)

rows=[]
for pdf in sorted(RAW.glob("*.pdf")):
    b = pdf.read_bytes()
    lines = ocr_lines(b)
    clean = [re.sub(r'\.{2,}',' ', ln).strip() for ln in lines if ln.strip()]

    items=[]
    # multipliers with '='
    pending=None
    for ln in clean:
        m = MULT_EQ.match(ln)
        if m:
            nm = (m.group("name_a") or pending or "").strip()
            qty = int(m.group("qty_a"))
            unit= to_num(m.group("unit_a"))
            tot = to_num(m.group("tot_a"))
            if nm and not is_header(nm) and tot is not None and 0 < tot <= 300:
                items.append({"product_raw": clean_name(nm), "qty": qty, "unit_price": unit, "line_total": tot})
            pending=None
            continue
        if not re.search(r'\d', ln):
            pending = ln
        else:
            pending = None

    # multipliers like "2X NAME 2.59"
    for ln in clean:
        m = MULT_NAME.match(ln)
        if not m: continue
        qty  = int(m.group("qty_b"))
        nm   = m.group("name_b").strip()
        unit = to_num(m.group("unit_b"))
        tot  = round(qty * unit, 2) if (unit is not None) else None
        if nm and not is_header(nm) and unit is not None and 0 < (tot or 0) <= 300:
            items.append({"product_raw": clean_name(nm), "qty": qty, "unit_price": unit, "line_total": tot})

    # single-line "NAME ... amount"
    for ln in clean:
        mt = AMT_TAIL.search(ln)
        if not mt: continue
        nm = ln[:mt.start()].strip()
        if not nm or is_header(nm): continue
        amt = to_num(mt.group("amt"))
        if amt is None or not (0 < amt <= 300): continue
        items.append({"product_raw": clean_name(nm), "qty": 1, "unit_price": amt, "line_total": amt})

    # amount-only lines -> previous non-header
    prev=None
    for ln in clean:
        if AMT_ONLY.match(ln):
            amt = to_num(ln)
            if amt is None or not (0 < amt <= 300): continue
            if prev and not is_header(prev) and not AMT_ONLY.match(prev):
                items.append({"product_raw": clean_name(prev), "qty": 1, "unit_price": amt, "line_total": amt})
        else:
            prev = ln

    # dedupe (name,total)
    uniq=[]; seen=set()
    for it in items:
        key=(it["product_raw"].lower(), it["line_total"])
        if key in seen: continue
        seen.add(key); uniq.append(it)

    rid = pdf.stem[:8]
    for i,it in enumerate(uniq,1):
        rows.append([rid, i, it["product_raw"], it["qty"], it["unit_price"], it["line_total"], "EUR"])

out = DATA/"items.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["receipt_id","line_no","product","qty","unit_price","line_total","currency"]); w.writerows(rows)
print(f"Wrote {out} with {len(rows)} lines")
if rows:
    print("Sample:", rows[:5])
