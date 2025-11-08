import re, os
from dateutil import parser as dateparser
from PIL import Image
import pytesseract

AMOUNT = re.compile(r'([0-9]+(?:[.,][0-9]{2}))\s*€')
DATE_RE = re.compile(r'(\d{2}/\d{2}/\d{4})|(\d{2}/\d{2}/\d{2})')
TOTAL_HINTS = ('TOTAL', 'T.T.C', 'A PAYER', 'RESTE A PAYER')
ITEM_LINE = re.compile(r'^\s*(?P<name>.+?)\s+(?P<amount>[0-9]+(?:[.,][0-9]{2}))\s*€\s*$', re.I)
QTY_PRICE = re.compile(r'(?P<name>.+?)\s+(?P<qty>[0-9]+(?:[.,][0-9]{1,3})?)\s*x\s*(?P<unit>[0-9]+(?:[.,][0-9]{2}))\s*€', re.I)

def normalize_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9 %\-_/]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def _parse_text(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    vendor = (lines[0][:60] if lines else 'Naturalia')
    # date
    date_val = None
    m = DATE_RE.search(text)
    if m:
        try: date_val = dateparser.parse(m.group(0), dayfirst=True).date().isoformat()
        except: pass
    # total
    total, conf = None, 0.3
    for ln in reversed(lines):
        if any(k in ln.upper() for k in TOTAL_HINTS):
            m2 = AMOUNT.search(ln)
            if m2:
                total = float(m2.group(1).replace(',', '.'))
                conf = 0.9
                break
    if total is None:
        m3 = AMOUNT.search(text)
        if m3:
            total = float(m3.group(1).replace(',', '.'))
            conf = 0.5
    # items
    items, line_no = [], 0
    for ln in lines:
        m = QTY_PRICE.search(ln)
        if m:
            name = m.group('name').strip()
            qty = float(m.group('qty').replace(',', '.'))
            unit = float(m.group('unit').replace(',', '.'))
            line_no += 1
            items.append({
                'line_no': line_no,
                'product_raw': name,
                'product_norm': normalize_name(name),
                'qty': qty,
                'unit_price': unit,
                'line_total': round(qty * unit, 2)
            })
            continue
        m2 = ITEM_LINE.match(ln)
        if m2:
            name = m2.group('name').strip()
            amt = float(m2.group('amount').replace(',', '.'))
            line_no += 1
            items.append({
                'line_no': line_no,
                'product_raw': name,
                'product_norm': normalize_name(name),
                'qty': '',
                'unit_price': '',
                'line_total': amt
            })
    return {'vendor': vendor or 'Naturalia', 'date': date_val, 'total': total, 'items': items, 'confidence': conf}

def parse_attachment(bytes_data: bytes, fname: str):
    ext = os.path.splitext(fname.lower())[1]
    text = ''
    if ext in ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'):
        from io import BytesIO
        text = pytesseract.image_to_string(Image.open(BytesIO(bytes_data)), lang='fra+eng')
    else:
        try: text = bytes_data.decode('utf-8', errors='ignore')
        except: text = ''
    return _parse_text(text)
