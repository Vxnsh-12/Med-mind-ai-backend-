import os

def read_file(filepath):
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    if ext == '.txt':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f: # Fallback
                return f.read()
    elif ext == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError:
             raise ImportError("pypdf is required for PDF files. Please install it using 'pip install pypdf'")
             
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {e}")
    else:
        raise ValueError("Unsupported file format. Please use .txt or .pdf")
