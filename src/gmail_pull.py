from __future__ import print_function
import os, base64, csv, hashlib
from datetime import datetime
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from src.parse_receipt import parse_attachment

def _normalize_name(s: str) -> str:
    return ' '.join((s or '').split()).lower()


LABEL = 'Naturalia'
PROCESSED_LABEL = 'Naturalia/Processed'

DATA_DIR = 'data'
RAW_DIR = 'raw'
RCPT_CSV = f'{DATA_DIR}/receipts.csv'
ITEMS_CSV = f'{DATA_DIR}/items.csv'
HASHES = f'{DATA_DIR}/hashes.txt'

SCOPES_READONLY = ['https://www.googleapis.com/auth/gmail.readonly']
SCOPES_MODIFY   = ['https://www.googleapis.com/auth/gmail.modify']  # needed only if you want to relabel as processed

def ensure():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(RCPT_CSV):
        open(RCPT_CSV,'w').write('receipt_id,date,vendor,store,subtotal,tax,total,currency,source_msg_id,confidence\n')
    if not os.path.exists(ITEMS_CSV):
        open(ITEMS_CSV,'w').write('receipt_id,line_no,product_raw,product_norm,qty,unit_price,line_total,currency\n')
    if not os.path.exists(HASHES): open(HASHES,'w').close()

def load_hashes():
    with open(HASHES) as f: return {x.strip() for x in f}
def add_hash(h):
    with open(HASHES,'a') as f: f.write(h+'\n')
def append_row(path, row):
    import csv
    with open(path,'a',newline='') as f: csv.writer(f).writerow(row)

def get_label_id(svc, name):
    labs = svc.users().labels().list(userId='me').execute().get('labels',[])
    for l in labs:
        if l['name'] == name:
            return l['id']
    # create if missing (only for processed)
    if name == PROCESSED_LABEL:
        created = svc.users().labels().create(userId='me', body={'name': name}).execute()
        return created['id']
    return None

def run(modify_after=False):
    ensure()
    import os
    search_q = os.environ.get('GMAIL_SEARCH_Q')
    scopes = SCOPES_MODIFY if modify_after else SCOPES_READONLY
    creds = Credentials.from_authorized_user_file('token.json', scopes)
    svc = build('gmail','v1',credentials=creds)

    src_label_id = get_label_id(svc, LABEL) if not search_q else None
    dst_label_id = get_label_id(svc, PROCESSED_LABEL) if modify_after else None

    seen = load_hashes()
    import os
    lookback = os.environ.get('LOOKBACK_DAYS','365')
    print('Search query: ', search_q or f'newer_than:{lookback}d')
    page_token = None

    while True:
        # pull recent first; feel free to change the query window
        resp = svc.users().messages().list(
            userId='me',
            labelIds=[src_label_id] if (not search_q and src_label_id) else None,
            q=search_q or f'newer_than:{lookback}d',
            maxResults=200,
            pageToken=page_token
        ).execute()

        msgs = resp.get('messages', [])
        page_token = resp.get('nextPageToken')

        for m in msgs:
            full = svc.users().messages().get(userId='me', id=m['id']).execute()
            payload = full.get('payload', {})
            parts = payload.get('parts') or []
            if not parts and 'body' in payload:
                parts = [payload]

            processed_any = False

            for p in parts:
                filename = (p.get('filename') or '').strip()
                body = p.get('body', {})
                data_b64 = body.get('data')
                att_id = body.get('attachmentId')

                # only PDFs (skip inline logos)
                if filename and not filename.lower().endswith('.pdf'):
                    continue

                if data_b64:
                    blob = base64.urlsafe_b64decode(data_b64)
                elif att_id:
                    att = svc.users().messages().attachments().get(userId='me', messageId=m['id'], id=att_id).execute()
                    blob = base64.urlsafe_b64decode(att['data'])
                    if not filename:
                        filename = f'{m["id"]}-{att_id}.pdf'
                else:
                    continue

                if len(blob) < 4_000:  # skip likely inline images or tiny blobs
                    continue

                h = hashlib.sha256(blob).hexdigest()
                if h in seen:
                    continue
                seen.add(h); add_hash(h)

                raw_path = os.path.join(RAW_DIR, filename or f'{m["id"]}.pdf')
                with open(raw_path,'wb') as f: f.write(blob)

                parsed = parse_attachment(blob, filename or 'blob.pdf')
                receipt_id = f'{m["id"][:8]}-{h[:8]}'

                append_row(RCPT_CSV, [
                    receipt_id, parsed.get('date') or '', 'Naturalia', '', '', '', parsed.get('total') or '',
                    'EUR', full.get('id',''), parsed.get('confidence',0.0)
                ])
                for it in parsed.get('items', []):
                    append_row(ITEMS_CSV, [
                        receipt_id, it['line_no'], it['product_raw'], (it.get('product_norm') or _normalize_name(it.get('product_raw',''))),
                        it['qty'], it['unit_price'], it['line_total'], 'EUR'
                    ])

                print('Ingested', filename or '<inline>', '→', receipt_id)
                processed_any = True

            # move email to Processed once at least one PDF was ingested
            if modify_after and processed_any and dst_label_id:
                svc.users().messages().modify(
                    userId='me',
                    id=m['id'],
                    body={'addLabelIds':[dst_label_id], 'removeLabelIds':[src_label_id]}
                ).execute()

        if not page_token:
            break

if __name__ == '__main__':
    # set to True if you want to move messages to .../Processed (requires gmail.modify scope)
    run(modify_after=False)