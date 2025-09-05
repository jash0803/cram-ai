import PyPDF2
import io
from typing import List, Dict, Any
import streamlit as st

class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    def __init__(self):
        self.extracted_texts = {}
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better processing"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def process_pdf(self, pdf_file, filename: str) -> Dict[str, Any]:
        """Process PDF and return structured data"""
        text = self.extract_text_from_pdf(pdf_file)
        chunks = self.chunk_text(text)
        
        return {
            "filename": filename,
            "full_text": text,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "word_count": len(text.split())
        }
