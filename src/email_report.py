import base64
import mimetypes
import os
from email.message import EmailMessage
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


def _load_creds() -> Credentials:
    token_path = Path("token.json")
    if not token_path.exists():
        raise FileNotFoundError("token.json not found (did workflow recreate Gmail token?)")
    return Credentials.from_authorized_user_file(str(token_path))


def _attach_file(msg: EmailMessage, path: Path) -> None:
    ctype, encoding = mimetypes.guess_type(str(path))
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)

    data = path.read_bytes()
    msg.add_attachment(
        data,
        maintype=maintype,
        subtype=subtype,
        filename=path.name,
    )


def main() -> None:
    report_to = os.environ.get("REPORT_TO", "").strip()
    if not report_to:
        raise SystemExit("REPORT_TO env var is required (set secrets.REPORT_TO).")

    report_path = Path(os.environ.get("REPORT_PATH", "artifacts/dashboard.html"))
    if not report_path.exists():
        raise SystemExit(f"Report file not found at {report_path}. Ensure dashboard build writes it.")

    subject_base = os.environ.get("REPORT_SUBJECT", "Naturalia report").strip() or "Naturalia report"
    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    today = os.environ.get("TODAY", "").strip()
    subject = subject_base
    if today or run_id:
        subject += f" ({today or 'date-unknown'} · run {run_id or 'n/a'})"

    msg = EmailMessage()
    msg["To"] = report_to
    msg["Subject"] = subject
    msg.set_content(
        "Attached: Naturalia dashboard HTML report.\n\n"
        "If you can't open it directly from your mail client, download the attachment and open in a browser."
    )

    _attach_file(msg, report_path)

    creds = _load_creds()
    service = build("gmail", "v1", credentials=creds)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    print(f"Sent report to {report_to} with attachment {report_path}.")


if __name__ == "__main__":
    main()
