from __future__ import print_function
import os, base64, csv, hashlib, json
from typing import Dict, Optional
from urllib import request
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
STATE_FILE = os.path.join(DATA_DIR, 'state', 'last_receipt_fingerprint.json')

SCOPES_READONLY = ['https://www.googleapis.com/auth/gmail.readonly']
SCOPES_MODIFY   = ['https://www.googleapis.com/auth/gmail.modify']  # needed only if you want to relabel as processed

def ensure():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
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


def load_state() -> Optional[Dict[str, str]]:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_state(fingerprint: str, internal_date: int) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump({
            'last_seen_receipt_fingerprint': fingerprint,
            'last_seen_internal_date': internal_date
        }, f, indent=2)

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

def _build_query(search_q: Optional[str], lookback: str, last_seen_internal_date: int) -> str:
    query_parts = []
    if search_q:
        query_parts.append(search_q)
    else:
        normalized_lookback = (lookback or '').strip().lower()
        if normalized_lookback and normalized_lookback not in {'all', '0'}:
            query_parts.append(f'newer_than:{lookback}d')
    if last_seen_internal_date:
        # internalDate is in ms since epoch; Gmail expects seconds for the "after:" operator
        after_ts = (last_seen_internal_date // 1000) + 1
        query_parts.append(f'after:{after_ts}')
    return ' '.join(query_parts)


def _trigger_self_dispatch(lookback: str) -> bool:
    token = os.environ.get('GITHUB_TOKEN')
    repository = os.environ.get('GITHUB_REPOSITORY')
    workflow_ref = os.environ.get('GITHUB_WORKFLOW_REF') or ''
    workflow_file = workflow_ref.split('/')[-1].split('@')[0] if workflow_ref else 'report.yml'
    ref = os.environ.get('GITHUB_REF_NAME') or os.environ.get('GITHUB_REF', 'main').split('/')[-1]

    if not token or not repository:
        print('Warning: unable to auto-dispatch workflow (missing token or repository context)')
        return False

    url = f'https://api.github.com/repos/{repository}/actions/workflows/{workflow_file}/dispatches'
    payload = json.dumps({
        'ref': ref,
        'inputs': {
            'lookback_days': lookback,
            'trigger_reason': 'new_receipt'
        }
    }).encode('utf-8')

    req = request.Request(
        url,
        data=payload,
        headers={
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'naturalia-spend-reporter'
        },
        method='POST'
    )

    try:
        with request.urlopen(req) as resp:
            print('Auto-dispatch response status:', resp.status)
        return True
    except Exception as exc:
        print('Warning: failed to retrigger workflow:', exc)
        return False


def run(modify_after=False):
    ensure()
    search_q = os.environ.get('GMAIL_SEARCH_Q')
    scopes = SCOPES_MODIFY if modify_after else SCOPES_READONLY
    creds = Credentials.from_authorized_user_file('token.json', scopes)
    svc = build('gmail','v1',credentials=creds)

    src_label_id = get_label_id(svc, LABEL) if not search_q else None
    dst_label_id = get_label_id(svc, PROCESSED_LABEL) if modify_after else None

    seen = load_hashes()
    lookback = os.environ.get('LOOKBACK_DAYS','365')

    state = load_state()
    last_seen_internal_date = int(state.get('last_seen_internal_date', 0)) if state else 0
    is_first_run = state is None

    trigger_reason = os.environ.get('TRIGGER_REASON', 'manual')
    allow_retrigger = trigger_reason in {'manual', 'schedule'}

    query = _build_query(search_q, lookback, last_seen_internal_date)
    print('Search query:', query)
    if last_seen_internal_date:
        print('Last seen internal date (ms):', last_seen_internal_date)

    page_token = None
    new_receipts: Dict[str, int] = {}

    while True:
        # pull recent first; feel free to change the query window
        resp = svc.users().messages().list(
            userId='me',
            labelIds=[src_label_id] if (not search_q and src_label_id) else None,
            q=query,
            maxResults=200,
            pageToken=page_token
        ).execute()

        msgs = resp.get('messages', [])
        page_token = resp.get('nextPageToken')

        for m in msgs:
            full = svc.users().messages().get(userId='me', id=m['id']).execute()
            payload = full.get('payload', {})
            internal_date = int(full.get('internalDate', 0))
            fingerprint = full.get('id')

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

            if processed_any:
                # Only mark as new when at least one PDF attachment was ingested
                new_receipts[fingerprint] = internal_date

        if not page_token:
            break

    if not new_receipts:
        print('No new receipts found; exiting without retrigger.')
        return

    # Persist latest seen receipt to avoid re-processing
    latest_fp, latest_date = max(new_receipts.items(), key=lambda kv: kv[1])
    save_state(latest_fp, latest_date)
    print('Updated receipt state with latest internal date:', latest_date)
    print('NEW_RECEIPT_DETECTED')

    if is_first_run:
        print('First run detected; skipping auto-retrigger to avoid loop.')
        return

    if not allow_retrigger:
        print(f'Skipping auto-retrigger; trigger_reason={trigger_reason} is not allowed to relaunch the workflow.')
        return

    _trigger_self_dispatch(lookback)


if __name__ == '__main__':
    # set to True if you want to move messages to .../Processed (requires gmail.modify scope)
    run(modify_after=False)
