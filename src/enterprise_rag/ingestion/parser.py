from pathlib import Path
import hashlib


def parse_markdown(path: Path) -> tuple[str, str]:
    if path.suffix.lower() != ".md":
        raise ValueError("Only Markdown files supported")

    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()

    text = raw.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if not text.strip():
        raise ValueError("Document is empty")

    return text, content_hash