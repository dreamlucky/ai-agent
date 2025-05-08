from PyPDF2 import PdfReader

def extract_pdf_text(filepath: str, max_chars=3000):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
        if len(text) > max_chars:
            break
    return text[:max_chars]
