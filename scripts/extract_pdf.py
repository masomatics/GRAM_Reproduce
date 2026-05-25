"""Extract text from GRAM_Looped.pdf for offline reading."""
import sys
from pathlib import Path

pdf = Path("/work/gj26/b20090/GRAM_Reproduce/documents/GRAM_Looped.pdf")
out = Path("/work/gj26/b20090/GRAM_Reproduce/documents/GRAM_Looped.txt")

try:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf))
    n = len(reader.pages)
    print(f"pypdf: {n} pages", flush=True)
    chunks = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception as e:
            t = f"[extract error: {e}]"
        chunks.append(f"\n===== PAGE {i+1} =====\n{t}")
    out.write_text("".join(chunks))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    sys.exit(0)
except ImportError:
    print("pypdf missing, trying PyPDF2…", flush=True)

try:
    import PyPDF2
    reader = PyPDF2.PdfReader(str(pdf))
    n = len(reader.pages)
    print(f"PyPDF2: {n} pages", flush=True)
    chunks = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception as e:
            t = f"[extract error: {e}]"
        chunks.append(f"\n===== PAGE {i+1} =====\n{t}")
    out.write_text("".join(chunks))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    sys.exit(0)
except ImportError:
    print("PyPDF2 missing, trying pdfminer…", flush=True)

from pdfminer.high_level import extract_text
text = extract_text(str(pdf))
out.write_text(text)
print(f"wrote {out} ({out.stat().st_size} bytes) via pdfminer")
