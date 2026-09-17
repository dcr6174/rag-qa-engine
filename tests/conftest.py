import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


FIXTURE_DOC = """# Deployment Guide

Welcome to the deployment guide. It covers configuration and networking.

## Configuration

Configuration lives in a YAML file. The file is read at startup.

| Setting | Default |
| ------- | ------- |
| workers | 4       |
| debug   | false   |

Edit the file before the first run.

## Networking

Timeout is 30s by default. Increase it for slow upstreams.

```python
client = connect(host, timeout=30)
client.retry(backoff=2)
```

Retries use exponential backoff.
"""


def make_pdf(page_texts):
    """Build a minimal multi-page PDF pypdf can read (no external deps)."""
    objects = []
    pages_kids = " ".join(f"{3+i} 0 R" for i in range(len(page_texts)))
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{pages_kids}] /Count {len(page_texts)} >>")
    for i, text in enumerate(page_texts):
        content_id = 3 + len(page_texts) + i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>"
        )
    for text in page_texts:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = ["%PDF-1.4"]
    offsets = []
    pos = len("%PDF-1.4\n")
    for i, body in enumerate(objects, start=1):
        offsets.append(pos)
        chunk = f"{i} 0 obj\n{body}\nendobj\n"
        out.append(chunk)
        pos += len(chunk)
    xref_pos = pos
    n = len(objects) + 1
    xref = "xref\n0 %d\n0000000000 65535 f \n" % n
    for off in offsets:
        xref += "%010d 00000 n \n" % off
    out.append(xref)
    out.append(f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF")
    return "\n".join(out).encode()
