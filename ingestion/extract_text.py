# 1. Import Libraries
from PyPDF2 import PdfReader
import os


# 2. Function to Extract Text from PDFs
def extract_text_from_pdfs(pdf_folder):
    """
    Function to extract text from PDF files.
    
    Returns:
            {"file_1.pdf": "text_1",
            "file_2.pdf": "text_2"}
    """
    text_data = {}
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            reader = PdfReader(path)
            full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            text_data[file] = full_text
    
    return text_data


# 3. Function to Extract Text, with Pages, from PDFs
def extract_text_with_pages(pdf__folder):
    """
    Function to extract text, with pages, from PDF files.
    
    Returns: 
            { "file.pdf": [ {"page_number": 1, "text": "..."},
                            {"page_number": 2, "text": "..."} ] }
    """
    data = {}
    for file in os.listdir(pdf__folder):
        if not file.endswith(".pdf"):
            continue
        
        path = os.path.join(pdf__folder, file)
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append({"page_number": i, "text": txt})
        
        if pages:
            data[file] = pages

    return data
